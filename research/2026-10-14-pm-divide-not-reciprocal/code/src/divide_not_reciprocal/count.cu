// Walk every float32 bit pattern and compare four ways of dividing by 6 against the IEEE quotient.
//
// What: for each of the 2^32 patterns u, x = bits(u), q = div.rn.f32(x, 6) (the IEEE 754 quotient: the exact
// result rounded once to nearest even, written as inline PTX so that no compiler flag can touch it), and
// p = one variant of "x divided by 6". The distance between p and q is counted in units of the last place
// (ulp: one step between neighbouring floats), per input binade (the top 9 bits of u: sign and exponent).
// Why: within exponent fields 4..254 (both signs), the exact residue rule predicts 2,796,202
// differences per binade: approximately one third, each one step away from zero. Smaller magnitudes
// and fast-math modes are separate cohorts. The whole-space walk also counts those regions.
//
// Built twice by exhaustive.py: once with nvcc's defaults (-prec-div=true, IEEE division) and once with
// --use_fast_math (-prec-div=false -ftz=true). Each variant is a __noinline__ function so its PTX body can
// be read on its own.
#include <cuda_runtime.h>
#include <stdint.h>

// The IEEE quotient. Inline PTX: nvcc's -ftz and -prec-div flags rewrite the PTX it emits, not PTX we write.
__device__ __forceinline__ float ieee_div6(float x) {
    float r;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(6.0f));
    return r;
}

// Variants. 0: the reciprocal multiply (the constant 1.0f / 6.0f is folded at compile time to fl(1/6)).
// 1: plain C division (IEEE by default, approximate under --use_fast_math). 2: __fdividef (div.approx.f32).
// 3: the __fdiv_rn intrinsic (the control: IEEE in both builds, unless -ftz changes subnormals).
template <int V> __device__ __noinline__ float variant(float x);
template <> __device__ __noinline__ float variant<0>(float x) { return x * (1.0f / 6.0f); }
template <> __device__ __noinline__ float variant<1>(float x) { return x / 6.0f; }
template <> __device__ __noinline__ float variant<2>(float x) { return __fdividef(x, 6.0f); }
template <> __device__ __noinline__ float variant<3>(float x) { return __fdiv_rn(x, 6.0f); }

// Floats of one sign ordered as integers: neighbouring floats differ by 1, +0 and -0 both map to 0.
__device__ __forceinline__ long long ordered(float f) {
    unsigned int b = __float_as_uint(f);
    long long m = (long long)(b & 0x7fffffffu);
    return (b & 0x80000000u) ? -m : m;
}

enum { SAME = 0, AWAY1 = 1, TOWARD1 = 2, FAR = 3, NAN_BOTH = 4, NAN_ONE = 5, NCLASS = 6 };
enum { NBIN = 512 };   // sign and exponent of the input: u >> 23

__device__ __forceinline__ void flush(unsigned int* hist, unsigned int* max_ulp, unsigned int bin,
                                      unsigned int* c, unsigned int& mx) {
#pragma unroll
    for (int k = 0; k < NCLASS; ++k) {
        if (c[k]) { atomicAdd(&hist[bin * NCLASS + k], c[k]); c[k] = 0; }
    }
    if (mx) { atomicMax(&max_ulp[bin], mx); mx = 0; }
}

// The probe gives a separate SASS inspection point. Whether the compiler retains a call or inlines
// a variant depends on the toolchain and must be checked in the emitted code.
template <int V>
__global__ void probe_kernel(const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    y[i] = variant<V>(x[i]);
}

// Each thread walks `per_thread` consecutive patterns, so a thread's patterns share a bin until a binade
// boundary; it accumulates in registers and adds to the global histogram once per bin it touches.
// max_ulp has one slot per bin, so the largest distance can be read per region of the input space.
template <int V>
__global__ void count_kernel(unsigned long long lo, unsigned long long hi, unsigned int per_thread,
                             unsigned int* hist, unsigned int* max_ulp) {
    unsigned long long start = lo + (unsigned long long)(blockIdx.x * blockDim.x + threadIdx.x) * per_thread;
    if (start >= hi) return;
    unsigned long long end = start + per_thread;
    if (end > hi) end = hi;
    unsigned int cur = (unsigned int)(start >> 23);
    unsigned int c[NCLASS] = {0, 0, 0, 0, 0, 0};
    unsigned int mx = 0;
    for (unsigned long long u = start; u < end; ++u) {
        unsigned int bin = (unsigned int)(u >> 23);
        if (bin != cur) { flush(hist, max_ulp, cur, c, mx); cur = bin; }
        float x = __uint_as_float((unsigned int)u);
        float q = ieee_div6(x);
        float p = variant<V>(x);
        bool qn = isnan(q), pn = isnan(p);
        if (qn || pn) { c[(qn && pn) ? NAN_BOTH : NAN_ONE]++; continue; }
        long long d = ordered(p) - ordered(q);
        if (d == 0) { c[SAME]++; continue; }
        long long ad = d < 0 ? -d : d;
        bool away = (__float_as_uint(p) & 0x7fffffffu) > (__float_as_uint(q) & 0x7fffffffu);
        // Bit magnitudes handle signed zero and remain valid in an FTZ build.
        if (ad == 1) c[away ? AWAY1 : TOWARD1]++; else c[FAR]++;
        unsigned int adu = ad > 0xffffffffll ? 0xffffffffu : (unsigned int)ad;
        if (adu > mx) mx = adu;
    }
    flush(hist, max_ulp, cur, c, mx);
}

// Instantiate the probes so their SASS is in the library.
template __global__ void probe_kernel<0>(const float*, float*);
template __global__ void probe_kernel<1>(const float*, float*);
template __global__ void probe_kernel<2>(const float*, float*);
template __global__ void probe_kernel<3>(const float*, float*);

// hist_out: NBIN * NCLASS counters; max_ulp_out: NBIN largest distances, one per bin; ms_out: kernel time.
extern "C" int run_count(int variant, unsigned long long lo, unsigned long long hi,
                         unsigned int* hist_out, unsigned int* max_ulp_out, float* ms_out) {
    if (variant < 0 || variant > 3 || lo >= hi || hi > (1ull << 32)) return (int)cudaErrorInvalidValue;
    const unsigned int per_thread = 4096, block = 256;
    unsigned long long n = hi - lo;
    unsigned long long threads = (n + per_thread - 1) / per_thread;
    unsigned int grid = (unsigned int)((threads + block - 1) / block);
    unsigned int* hist = nullptr;
    unsigned int* mx = nullptr;
    cudaEvent_t t0 = nullptr, t1 = nullptr;
    cudaError_t e = cudaSuccess, cleanup_error = cudaSuccess;
#define CHECK_CUDA(call) do { e = (call); if (e != cudaSuccess) goto cleanup; } while (0)
    CHECK_CUDA(cudaMalloc(&hist, NBIN * NCLASS * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&mx, NBIN * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(hist, 0, NBIN * NCLASS * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(mx, 0, NBIN * sizeof(unsigned int)));
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0));
    switch (variant) {
        case 0: count_kernel<0><<<grid, block>>>(lo, hi, per_thread, hist, mx); break;
        case 1: count_kernel<1><<<grid, block>>>(lo, hi, per_thread, hist, mx); break;
        case 2: count_kernel<2><<<grid, block>>>(lo, hi, per_thread, hist, mx); break;
        case 3: count_kernel<3><<<grid, block>>>(lo, hi, per_thread, hist, mx); break;
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(ms_out, t0, t1));
    CHECK_CUDA(cudaMemcpy(hist_out, hist, NBIN * NCLASS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(max_ulp_out, mx, NBIN * sizeof(unsigned int), cudaMemcpyDeviceToHost));
cleanup:
    if (hist) { cleanup_error = cudaFree(hist); if (e == cudaSuccess) e = cleanup_error; }
    if (mx) { cleanup_error = cudaFree(mx); if (e == cudaSuccess) e = cleanup_error; }
    if (t0) { cleanup_error = cudaEventDestroy(t0); if (e == cudaSuccess) e = cleanup_error; }
    if (t1) { cleanup_error = cudaEventDestroy(t1); if (e == cudaSuccess) e = cleanup_error; }
#undef CHECK_CUDA
    return (int)e;
}
