#!/usr/bin/env python3
"""Fetch recorded footage from a Hikvision NVR/DVR for a time range (ISAPI).

Backfill: when a client's live stream was down (no internet), their on-site NVR
kept recording. This pulls the missed window as file(s), which the backfill
pipeline then processes UNTHROTTLED (NVDEC full speed) — main_enterprise with
[backfill].clip_start = each segment's real start time. A file source is the fast
path; RTSP playback would be paced to real-time.

Hikvision ISAPI flow (HTTP Digest auth):
  1. POST /ISAPI/ContentMgmt/search   (CMSearch) -> recorded segments in the range,
     each with a playbackURI + its true start/end time.
  2. POST /ISAPI/ContentMgmt/download {playbackURI} -> the media file (streamed to disk).

The NVR must be reachable from wherever this runs (client port-forward / VPN / public IP).

CLI:
  python3 tools/nvr_fetch.py --host 1.2.3.4 --port 80 --user admin --password '***' \
      --channel 1 --start 2026-07-10T08:00:00 --end 2026-07-10T09:00:00 --out /workspace/data/backfill
Prints one line per downloaded clip:  <path>\t<start>\t<end>  (feed <start> to [backfill].clip_start).
"""
import argparse
import datetime
import os
import uuid
import xml.etree.ElementTree as ET


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable without a network)
# --------------------------------------------------------------------------- #
def track_id(channel: int, substream: bool = False) -> int:
    """Hikvision track = channel*100 + stream (1=main, 2=sub). Cam1 main=101, cam2 main=201."""
    return int(channel) * 100 + (2 if substream else 1)


def iso_z(dt: datetime.datetime) -> str:
    """Hikvision time format, UTC: 2026-07-10T08:00:00Z."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_search_xml(track: int, start: datetime.datetime, end: datetime.datetime,
                     max_results: int = 400, position: int = 0) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<CMSearchDescription>'
        f'<searchID>{uuid.uuid4()}</searchID>'
        f'<trackList><trackID>{track}</trackID></trackList>'
        '<timeSpanList><timeSpan>'
        f'<startTime>{iso_z(start)}</startTime><endTime>{iso_z(end)}</endTime>'
        '</timeSpan></timeSpanList>'
        f'<maxResults>{max_results}</maxResults>'
        f'<searchResultPostion>{position}</searchResultPostion>'
        '<metadataList><metadataDescriptor>//recordType.meta.std-cgi.com</metadataDescriptor></metadataList>'
        '</CMSearchDescription>'
    )


def build_download_xml(playback_uri: str) -> str:
    return ('<?xml version="1.0" encoding="utf-8"?>'
            f'<downloadRequest><playbackURI>{playback_uri}</playbackURI></downloadRequest>')


def _strip_ns(root):
    for el in root.iter():
        if isinstance(el.tag, str) and '}' in el.tag:
            el.tag = el.tag.split('}', 1)[1]
    return root


def parse_search(xml_text: str) -> list[dict]:
    """Extract [{start, end, playback_uri}] from a CMSearchResult (namespace-agnostic)."""
    root = _strip_ns(ET.fromstring(xml_text))
    out = []
    for item in root.findall(".//searchMatchItem"):
        ts = item.find("./timeSpan")
        out.append({
            "start": ts.findtext("startTime") if ts is not None else None,
            "end": ts.findtext("endTime") if ts is not None else None,
            "playback_uri": item.findtext(".//playbackURI"),
        })
    return [s for s in out if s["playback_uri"]]


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #
class HikvisionNVR:
    def __init__(self, host, user, password, port=80, scheme="http", timeout=60, verify=True):
        from requests.auth import HTTPDigestAuth
        self.base = f"{scheme}://{host}:{port}"
        self.auth = HTTPDigestAuth(user, password)
        self.timeout = timeout
        self.verify = verify

    def _post(self, path, xml, **kw):
        import requests
        return requests.post(self.base + path, data=xml,
                             headers={"Content-Type": "application/xml"},
                             auth=self.auth, timeout=self.timeout, verify=self.verify, **kw)

    def search(self, track, start, end):
        r = self._post("/ISAPI/ContentMgmt/search", build_search_xml(track, start, end))
        r.raise_for_status()
        return parse_search(r.text)

    def download(self, playback_uri, out_path):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with self._post("/ISAPI/ContentMgmt/download", build_download_xml(playback_uri), stream=True) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    if chunk:
                        f.write(chunk)
        return out_path

    def fetch_range(self, track, start, end, out_dir):
        """Search + download every segment overlapping [start, end]. Returns
        [{path, start, end}] — feed each `start` to the backfill pipeline's clip_start."""
        results = []
        for i, seg in enumerate(self.search(track, start, end)):
            path = os.path.join(out_dir, f"nvr_{track}_{i:03d}.mp4")
            self.download(seg["playback_uri"], path)
            results.append({"path": path, "start": seg["start"], "end": seg["end"]})
        return results


def main():
    ap = argparse.ArgumentParser(description="Fetch Hikvision NVR recordings for a time range.")
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=80)
    ap.add_argument("--user", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--channel", type=int, required=True, help="NVR channel number (1-based)")
    ap.add_argument("--substream", action="store_true", help="use the sub stream (track *02)")
    ap.add_argument("--start", required=True, help="ISO datetime, e.g. 2026-07-10T08:00:00")
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="/workspace/data/backfill")
    ap.add_argument("--https", action="store_true")
    ap.add_argument("--insecure", action="store_true", help="skip TLS verification")
    a = ap.parse_args()

    nvr = HikvisionNVR(a.host, a.user, a.password, a.port,
                       "https" if a.https else "http", verify=not a.insecure)
    tr = track_id(a.channel, a.substream)
    start = datetime.datetime.fromisoformat(a.start)
    end = datetime.datetime.fromisoformat(a.end)
    segs = nvr.fetch_range(tr, start, end, a.out)
    if not segs:
        print("no recordings found for that channel/range", flush=True)
        return
    for s in segs:
        print(f"{s['path']}\t{s['start']}\t{s['end']}", flush=True)


if __name__ == "__main__":
    main()
