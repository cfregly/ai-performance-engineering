#include <cassert>
#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptiling(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / 32; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // size of the warp subtile
  constexpr uint WMITER = (WM * WN) / (32 * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 64/2=32
  constexpr uint WSUBN = WN / WNITER; // 32/2=16

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % 32;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

  // allocate space for the current blocktile in SMEM
  __shared__ bf16 As[BM * BK];
  __shared__ bf16 Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // we cache into registers on the warptile level
  bf16 regM[WMITER * TM] = {0.0};
  bf16 regN[WNITER * TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Load from A
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      bf16 tmp[4];
      const float2 x = *reinterpret_cast<const float2*>(&A[(innerRowA + offset) * K + innerColA * 4]);
      memcpy(&tmp[0], &x, sizeof(bf16) * 4);
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp[0];
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp[1];
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp[2];
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp[3];
    }
    
    // Load from B
    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float2 *>(
          &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<const float2 *>(
              &B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
    __syncthreads();
    
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // populate registers for whole warptile
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint i = 0; i < TM; ++i) {
          regM[wSubRowIdx * TM + i] =
              As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                 threadRowInWarp * TM + i];
        }
      }
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        for (uint i = 0; i < TN; ++i) {
          regN[wSubColIdx * TN + i] =
              Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                 threadColInWarp * TN + i];
        }
      }
  
      // execute warptile matmul
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
          // calculate per-thread results
          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
              threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                            (wSubColIdx * TN) + resIdxN] +=
                  __bfloat162float(regM[wSubRowIdx * TM + resIdxM]) *
                  __bfloat162float(regN[wSubColIdx * TN + resIdxN]);
            }
          }
        }
      }
    }

    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
  }

  // write out the results
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // move C pointer to current warp subtile
      bf16 *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          bf16 tmp[4];
          float2 x;
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp[0] = threadResults[i + 0];
          tmp[1] = threadResults[i + 1];
          tmp[2] = threadResults[i + 2];
          tmp[3] = threadResults[i + 3];
          memcpy(&x, &tmp[0], sizeof(bf16) * 4);
          // write 
          *reinterpret_cast<float2*>(&C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN]) = x;
        }
      }
    }
  }
}

void runKernel1(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
  const uint K10_NUM_THREADS = 128;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 16;
  const uint K10_WN = 64;
  const uint K10_WM = 64;
  const uint K10_WNITER = 4;
  const uint K10_TN = 4;
  const uint K10_TM = 8;
  dim3 blockDim(K10_NUM_THREADS);

  constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
  static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K10_WM * K10_WN) % (32 * K10_TM * K10_TN * K10_WNITER) == 0);
  constexpr uint K10_WMITER =
  (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
  // warpsubtile in warptile
  static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

  static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
  "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
  "issues during GMEM->SMEM tiling (loading only parts of the "
  "final row of Bs during each iteraion)");
  static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
  "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
  "issues during GMEM->SMEM tiling (loading only parts of the "
  "final row of As during each iteration)");
  static_assert(K10_BN % (16 * K10_TN) == 0,
  "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K10_BM % (16 * K10_TM) == 0,
  "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
  "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
  "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
  sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS>
  <<<gridDim, blockDim>>>(M, N, K, A, B, C);
}