// NeuralForge-X — CUDA C++ similarity kernels.
//
// Compiled at runtime by CuPy's NVRTC (>= 12.8 for Blackwell sm_120) and launched
// on the GPU. Metric codes: 0 = cosine, 1 = dot, 2 = l2 (Euclidean distance).
//
// Layout matches the CPU core: row-major float32, vector i at offset i*dim.

extern "C" {

// Per-row L2 norm: norms[row] = sqrt(sum_j X[row,j]^2). One thread per row.
__global__ void row_norms(const float* __restrict__ X, int rows, int dim,
                          float* __restrict__ norms) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* x = X + (size_t)row * dim;
    float s = 0.0f;
    for (int j = 0; j < dim; ++j) s += x[j] * x[j];
    norms[row] = sqrtf(s);
}

// Full (q x n) similarity matrix, one thread per output element.
// `qnorm`/`cnorm` are used only for the cosine metric (precomputed by row_norms).
__global__ void batch_sim(const float* __restrict__ Q, const float* __restrict__ C,
                          int q, int n, int dim, int metric,
                          const float* __restrict__ qnorm, const float* __restrict__ cnorm,
                          float* __restrict__ out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // corpus index
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // query index
    if (i >= q || j >= n) return;

    const float* a = Q + (size_t)i * dim;
    const float* b = C + (size_t)j * dim;
    size_t o = (size_t)i * n + j;

    if (metric == 2) {  // L2 distance
        float acc = 0.0f;
        for (int t = 0; t < dim; ++t) { float d = a[t] - b[t]; acc += d * d; }
        out[o] = sqrtf(acc);
    } else {
        float acc = 0.0f;
        for (int t = 0; t < dim; ++t) acc += a[t] * b[t];
        if (metric == 0) {  // cosine
            float den = qnorm[i] * cnorm[j];
            out[o] = den > 0.0f ? acc / den : 0.0f;
        } else {  // dot
            out[o] = acc;
        }
    }
}

// Scores of a single query against every corpus row (for top-k). One thread/row.
__global__ void score_query(const float* __restrict__ query, const float* __restrict__ C,
                            int n, int dim, int metric, float qnorm,
                            const float* __restrict__ cnorm, float* __restrict__ out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    const float* b = C + (size_t)row * dim;

    if (metric == 2) {
        float acc = 0.0f;
        for (int t = 0; t < dim; ++t) { float d = query[t] - b[t]; acc += d * d; }
        out[row] = sqrtf(acc);
    } else {
        float acc = 0.0f;
        for (int t = 0; t < dim; ++t) acc += query[t] * b[t];
        if (metric == 0) {
            float den = qnorm * cnorm[row];
            out[row] = den > 0.0f ? acc / den : 0.0f;
        } else {
            out[row] = acc;
        }
    }
}

// Pairwise reduction over one vector pair using a single block + shared memory.
// Emits out = {dot(a,b), ||a||^2, ||b||^2}; the host derives cosine/dot/l2.
// Launch with one block; dynamic shared memory = 3 * blockDim.x * sizeof(float).
__global__ void pair_reduce(const float* __restrict__ a, const float* __restrict__ b,
                            int dim, float* __restrict__ out) {
    extern __shared__ float sh[];
    float* s_dot = sh;
    float* s_na = sh + blockDim.x;
    float* s_nb = sh + 2 * blockDim.x;

    float d = 0.0f, na = 0.0f, nb = 0.0f;
    for (int t = threadIdx.x; t < dim; t += blockDim.x) {
        float x = a[t], y = b[t];
        d += x * y; na += x * x; nb += y * y;
    }
    s_dot[threadIdx.x] = d; s_na[threadIdx.x] = na; s_nb[threadIdx.x] = nb;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_dot[threadIdx.x] += s_dot[threadIdx.x + stride];
            s_na[threadIdx.x] += s_na[threadIdx.x + stride];
            s_nb[threadIdx.x] += s_nb[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { out[0] = s_dot[0]; out[1] = s_na[0]; out[2] = s_nb[0]; }
}

}  // extern "C"
