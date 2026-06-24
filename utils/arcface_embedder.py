"""
Standalone ArcFace embedder shared by offline enrollment and live inference.

Recognition only works if gallery embeddings and live embeddings come from a
*pixel-identically* preprocessed embedder. To guarantee that, both paths import
this one module. Two backends are provided:

  - ArcFaceONNX : onnxruntime, CPU/GPU, used for offline enrollment and for
                  validating the pipeline without a GPU.
  - ArcFaceTRT  : TensorRT 10.x, used in the live DeepStream pipeline.

Both share the exact preprocessing dictated by config/config_arcface.txt:
    model-color-format=0   -> RGB channel order
    net-scale-factor=0.0078125  (= 1/128)
    offsets=127.5;127.5;127.5
    net-width=112, net-height=112, NCHW float32
i.e.  out = 0.0078125 * (pixel_rgb - 127.5)

Input chips are expected as HxWx3 uint8 in BGR (OpenCV native); conversion to
RGB happens here so neither caller can get the channel order wrong.
"""
import numpy as np
import cv2

NET_SIZE = 112
NET_SCALE = 0.0078125
OFFSET = 127.5
EMBED_DIM = 512


def preprocess(chips_bgr):
    """
    Convert a list/array of HxWx3 BGR uint8 chips into an (N, 3, 112, 112)
    float32 NCHW blob using the ArcFace preprocessing contract.
    """
    if isinstance(chips_bgr, np.ndarray) and chips_bgr.ndim == 3:
        chips_bgr = [chips_bgr]
    blob = np.empty((len(chips_bgr), 3, NET_SIZE, NET_SIZE), dtype=np.float32)
    for i, c in enumerate(chips_bgr):
        if c.shape[0] != NET_SIZE or c.shape[1] != NET_SIZE:
            c = cv2.resize(c, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = NET_SCALE * (rgb - OFFSET)
        blob[i] = rgb.transpose(2, 0, 1)
    return blob


def l2norm(vecs):
    """Row-wise L2 normalization of an (N, D) or (D,) array."""
    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim == 1:
        n = np.linalg.norm(vecs)
        return vecs / n if n > 0 else vecs
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class ArcFaceONNX:
    """ArcFace via onnxruntime. Suitable for offline enrollment and CPU tests."""

    def __init__(self, onnx_path, providers=None):
        import onnxruntime as ort
        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # Drop providers that are not actually available rather than erroring.
        avail = set(ort.get_available_providers())
        providers = [p for p in providers if p in avail] or ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def embed_raw(self, chips_bgr):
        """Return raw (N, 512) embeddings (not normalized)."""
        blob = preprocess(chips_bgr)
        outs = []
        # ArcFace ONNX export here fixes the output batch at 1, so iterate.
        for i in range(blob.shape[0]):
            out = self.sess.run([self.out_name], {self.in_name: blob[i:i + 1]})[0]
            outs.append(out.reshape(-1))
        return np.asarray(outs, dtype=np.float32)

    def embed(self, chips_bgr):
        """Return L2-normalized (N, 512) embeddings."""
        return l2norm(self.embed_raw(chips_bgr))


class ArcFaceTRT:
    """
    ArcFace via TensorRT 10.x. Used in the live pipeline. Requires a GPU and
    the serialized engine built for batch=1 (config/config_arcface.txt).

    Note: this path cannot be exercised without a GPU; validate on-device that
    its output cosine-matches ArcFaceONNX (> 0.99) before trusting a gallery
    enrolled with one backend against the other.
    """

    def __init__(self, engine_path, max_batch=16):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
        self._trt = trt
        self._cudart = cudart
        self.max_batch = max_batch

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()

        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.in_name = n
            else:
                self.out_name = n

        in_nbytes = max_batch * 3 * NET_SIZE * NET_SIZE * 4
        out_nbytes = max_batch * EMBED_DIM * 4
        self.d_in = self._cuda_malloc(in_nbytes)
        self.d_out = self._cuda_malloc(out_nbytes)
        err, self.stream = cudart.cudaStreamCreate()
        self._check(err)

    def _cuda_malloc(self, nbytes):
        err, ptr = self._cudart.cudaMalloc(nbytes)
        self._check(err)
        return ptr

    def _check(self, err):
        cudart = self._cudart
        if isinstance(err, tuple):
            err = err[0]
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA error: {err}")

    def embed_raw(self, chips_bgr):
        cudart = self._cudart
        blob = preprocess(chips_bgr)
        n = blob.shape[0]
        if n > self.max_batch:
            # Chunk to respect the preallocated buffers.
            return np.concatenate([self.embed_raw(blob_chunk)
                                   for blob_chunk in np.array_split(chips_bgr, n // self.max_batch + 1)], axis=0)
        blob = np.ascontiguousarray(blob)
        self.ctx.set_input_shape(self.in_name, (n, 3, NET_SIZE, NET_SIZE))
        self.ctx.set_tensor_address(self.in_name, int(self.d_in))
        self.ctx.set_tensor_address(self.out_name, int(self.d_out))

        self._check(cudart.cudaMemcpyAsync(
            self.d_in, blob.ctypes.data, blob.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
        self.ctx.execute_async_v3(self.stream)
        out = np.empty((n, EMBED_DIM), dtype=np.float32)
        self._check(cudart.cudaMemcpyAsync(
            out.ctypes.data, self.d_out, out.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
        self._check(cudart.cudaStreamSynchronize(self.stream))
        return out

    def embed(self, chips_bgr):
        return l2norm(self.embed_raw(chips_bgr))


def make_embedder(backend, model_path, **kw):
    """Factory: backend in {'onnx', 'trt'}."""
    backend = (backend or "onnx").lower()
    if backend == "trt":
        return ArcFaceTRT(model_path, **kw)
    return ArcFaceONNX(model_path, **kw)
