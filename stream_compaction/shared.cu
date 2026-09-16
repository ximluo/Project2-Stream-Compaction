#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "common.h"
#include "shared.h"

#define blockSize 128
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

namespace StreamCompaction {
    namespace Shared {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaiveScanBlock(int n, int *odata, const int *idata, int *blockSums) {
            extern __shared__ int temp[];
            int tid = threadIdx.x;
            int i = blockIdx.x * blockDim.x + tid;
            int x = i < n ? idata[i] : 0;
            int pout = 0;
            int pin = 1;
            temp[tid] = tid > 0 && i - 1 < n ? idata[i - 1] : 0;
            __syncthreads();
            for (int offset = 1; offset < blockDim.x; offset <<= 1) {
                pout = 1 - pout;
                pin = 1 - pout;
                if (tid >= offset) {
                    temp[pout * blockDim.x + tid] = temp[pin * blockDim.x + tid - offset] + temp[pin * blockDim.x + tid];
                } else {
                    temp[pout * blockDim.x + tid] = temp[pin * blockDim.x + tid];
                }
                __syncthreads();
            }
            if (i < n) {
                odata[i] = temp[pout * blockDim.x + tid];
            }
            if (tid == blockDim.x - 1) {
                blockSums[blockIdx.x] = temp[pout * blockDim.x + tid] + x;
            }
        }

        __global__ void kernScanBlock(int n, int *odata, const int *idata, int *blockSums) {
            extern __shared__ int temp[];
            int tid = threadIdx.x;
            int elems = 2 * blockDim.x;
            int base = blockIdx.x * elems;
            int ai = tid;
            int bi = tid + blockDim.x;
            int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
            int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
            temp[ai + bankOffsetA] = base + ai < n ? idata[base + ai] : 0;
            temp[bi + bankOffsetB] = base + bi < n ? idata[base + bi] : 0;
            int offset = 1;
            for (int d = elems >> 1; d > 0; d >>= 1) {
                __syncthreads();
                if (tid < d) {
                    int a = offset * (2 * tid + 1) - 1;
                    int b = offset * (2 * tid + 2) - 1;
                    a += CONFLICT_FREE_OFFSET(a);
                    b += CONFLICT_FREE_OFFSET(b);
                    temp[b] += temp[a];
                }
                offset <<= 1;
            }
            __syncthreads();
            if (tid == 0) {
                int last = elems - 1 + CONFLICT_FREE_OFFSET(elems - 1);
                blockSums[blockIdx.x] = temp[last];
                temp[last] = 0;
            }
            for (int d = 1; d < elems; d <<= 1) {
                offset >>= 1;
                __syncthreads();
                if (tid < d) {
                    int a = offset * (2 * tid + 1) - 1;
                    int b = offset * (2 * tid + 2) - 1;
                    a += CONFLICT_FREE_OFFSET(a);
                    b += CONFLICT_FREE_OFFSET(b);
                    int t = temp[a];
                    temp[a] = temp[b];
                    temp[b] += t;
                }
            }
            __syncthreads();
            if (base + ai < n) {
                odata[base + ai] = temp[ai + bankOffsetA];
            }
            if (base + bi < n) {
                odata[base + bi] = temp[bi + bankOffsetB];
            }
        }

        __global__ void kernAddIncrements(int n, int elems, int *data, const int *increments) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) {
                return;
            }
            data[i] += increments[i / elems];
        }

        void scanDevice(int n, int *dev_odata, const int *dev_idata, std::vector<int*> &sums, int level, bool naive) {
            int elems = naive ? blockSize : 2 * blockSize;
            int numBlocks = (n + elems - 1) / elems;
            if (naive) {
                kernNaiveScanBlock<<<numBlocks, blockSize, 2 * blockSize * sizeof(int)>>>(n, dev_odata, dev_idata, sums[level]);
            } else {
                int sharedBytes = (elems + CONFLICT_FREE_OFFSET(elems - 1)) * sizeof(int);
                kernScanBlock<<<numBlocks, blockSize, sharedBytes>>>(n, dev_odata, dev_idata, sums[level]);
            }
            if (numBlocks > 1) {
                scanDevice(numBlocks, sums[level], sums[level], sums, level + 1, naive);
                kernAddIncrements<<<(n + blockSize - 1) / blockSize, blockSize>>>(n, elems, dev_odata, sums[level]);
            }
        }

        void scanImpl(int n, int *odata, const int *idata, bool naive) {
            int elems = naive ? blockSize : 2 * blockSize;
            int *dev_in;
            int *dev_out;
            cudaMalloc(&dev_in, n * sizeof(int));
            cudaMalloc(&dev_out, n * sizeof(int));
            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            std::vector<int*> sums;
            int m = n;
            do {
                int numBlocks = (m + elems - 1) / elems;
                int *dev_sums;
                cudaMalloc(&dev_sums, numBlocks * sizeof(int));
                sums.push_back(dev_sums);
                m = numBlocks;
            } while (m > 1);
            timer().startGpuTimer();
            scanDevice(n, dev_out, dev_in, sums, 0, naive);
            checkCUDAError("shared scan failed");
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_in);
            cudaFree(dev_out);
            for (int *p : sums) {
                cudaFree(p);
            }
        }

        void scan(int n, int *odata, const int *idata) {
            scanImpl(n, odata, idata, false);
        }

        void scanNaive(int n, int *odata, const int *idata) {
            scanImpl(n, odata, idata, true);
        }
    }
}
