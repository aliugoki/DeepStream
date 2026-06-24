"""
Recognition core: gallery matching and per-track identity stabilization.

Replaces three weaknesses of the legacy probe (utils/probe_git.py):
  * hardcoded RECOGNITION_THRESHOLD = 0.2 (config rec_threshold was ignored)
    with a configurable threshold + a top-1/top-2 margin gate;
  * an O(N) Python loop over gallery keys with a single vectorized matmul;
  * a thread-unsafe clear()+update() gallery hot-reload with an atomic swap;
and adds per-track voting (recognize-once-per-track) with TTL eviction so the
global per-object state no longer grows without bound.

Pure numpy -- fully unit-testable without DeepStream.
"""
import threading
import time
from collections import defaultdict, Counter

import numpy as np


class Gallery:
    """
    Thread-safe gallery of L2-normalized face embeddings.

    Holds a contiguous (N, 512) matrix so matching is a single BLAS matmul.
    Hot-reload swaps the matrix atomically (a single attribute rebind under a
    lock), so the streaming thread never observes a half-updated gallery.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._ids = []

    def replace(self, faces):
        """
        Atomically replace the gallery. ``faces`` is a dict {id: vector} where
        each vector is a (D,)/(D,1)/(1,D) array (L2-normalized or not).
        """
        ids, rows = [], []
        for fid, vec in faces.items():
            if vec is None:
                continue
            v = np.asarray(vec, dtype=np.float32).reshape(-1)
            n = np.linalg.norm(v)
            if n == 0:
                continue
            ids.append(fid)
            rows.append(v / n)
        matrix = np.ascontiguousarray(np.vstack(rows)) if rows else np.zeros((0, 0), np.float32)
        with self._lock:
            self._matrix = matrix
            self._ids = ids

    def snapshot(self):
        with self._lock:
            return self._matrix, self._ids

    def __len__(self):
        with self._lock:
            return len(self._ids)

    def match(self, feature):
        """
        Return (best_id, best_score, margin) for a single probe ``feature``.

        ``margin`` is best_score - second_best_score (1.0 if only one entry,
        -1.0 if the gallery is empty). Cosine similarity == dot product since
        both gallery rows and the probe are L2-normalized.
        """
        matrix, ids = self.snapshot()
        if matrix.shape[0] == 0:
            return None, -1.0, -1.0
        v = np.asarray(feature, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(v)
        if n == 0:
            return None, -1.0, -1.0
        v = v / n
        scores = matrix @ v
        best = int(np.argmax(scores))
        best_score = float(scores[best])
        if scores.shape[0] >= 2:
            part = np.partition(scores, -2)
            second = float(part[-2])
            margin = best_score - second
        else:
            margin = 1.0
        return ids[best], best_score, margin


def accept_match(best_score, margin, threshold, min_margin):
    """A match is accepted only if it clears both the score and margin gates."""
    return best_score >= threshold and margin >= min_margin


class TrackIdentityManager:
    """
    Per-tracker-object identity stabilization (recognize-once-per-track).

    Accumulates accepted matches per object_id across frames and commits an
    identity once it has enough votes, so a single noisy frame can't mislabel a
    person and a committed track stops needing re-embedding. Old tracks are
    evicted by TTL so memory stays bounded over long runs.
    """

    def __init__(self, threshold=0.35, min_margin=0.05, min_votes=3,
                 ttl_seconds=30.0, time_fn=time.monotonic):
        self.threshold = threshold
        self.min_margin = min_margin
        self.min_votes = min_votes
        self.ttl = ttl_seconds
        self._now = time_fn
        self._tracks = {}
        self._last_sweep = self._now()

    def observe(self, object_id, best_id, best_score, margin):
        """
        Record one frame's match for a track and return the currently committed
        identity (or None). Once committed, the identity is sticky for the life
        of the track.
        """
        now = self._now()
        st = self._tracks.get(object_id)
        if st is None:
            st = {"votes": Counter(), "score_sum": defaultdict(float),
                  "committed": None, "last_seen": now}
            self._tracks[object_id] = st
        st["last_seen"] = now

        if st["committed"] is None and best_id is not None and \
                accept_match(best_score, margin, self.threshold, self.min_margin):
            st["votes"][best_id] += 1
            st["score_sum"][best_id] += best_score
            leader, count = st["votes"].most_common(1)[0]
            if count >= self.min_votes:
                st["committed"] = leader

        self._maybe_sweep(now)
        return st["committed"]

    def is_committed(self, object_id):
        st = self._tracks.get(object_id)
        return st is not None and st["committed"] is not None

    def _maybe_sweep(self, now):
        if now - self._last_sweep < self.ttl:
            return
        self._last_sweep = now
        stale = [oid for oid, st in self._tracks.items()
                 if now - st["last_seen"] > self.ttl]
        for oid in stale:
            del self._tracks[oid]

    def __len__(self):
        return len(self._tracks)


def _self_test():
    rng = np.random.default_rng(0)
    base = l2 = lambda x: x / np.linalg.norm(x)
    g = Gallery()
    vecs = {f"id{i}": l2(rng.standard_normal(512)) for i in range(50)}
    g.replace(vecs)
    assert len(g) == 50

    # A probe close to id7 should match id7 with a positive margin.
    probe = l2(vecs["id7"] + 0.01 * rng.standard_normal(512))
    bid, score, margin = g.match(probe)
    assert bid == "id7" and score > 0.9 and margin > 0, (bid, score, margin)

    # Gate rejects low score or low margin.
    assert accept_match(0.5, 0.1, 0.35, 0.05)
    assert not accept_match(0.30, 0.1, 0.35, 0.05)
    assert not accept_match(0.5, 0.01, 0.35, 0.05)

    # Voting commits only after min_votes and is sticky; TTL evicts.
    clock = [0.0]
    mgr = TrackIdentityManager(threshold=0.35, min_margin=0.05, min_votes=3,
                               ttl_seconds=10.0, time_fn=lambda: clock[0])
    assert mgr.observe(1, "alice", 0.8, 0.2) is None
    assert mgr.observe(1, "alice", 0.8, 0.2) is None
    assert mgr.observe(1, "alice", 0.8, 0.2) == "alice"
    # Noisy frame can't flip a committed track.
    assert mgr.observe(1, "bob", 0.9, 0.5) == "alice"
    clock[0] = 100.0
    mgr.observe(2, "carol", 0.8, 0.2)  # triggers sweep of stale track 1
    assert 1 not in mgr._tracks
    print("recognition self-test OK")


if __name__ == "__main__":
    _self_test()
