#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "radix.h"

#define blockSize 256

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernMapBit(int n, int bit, int *bools, int *notBools, const int *idata) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) {
                return;
            }
            int b = (idata[i] >> bit) & 1;
            bools[i] = b;
            notBools[i] = 1 - b;
        }

        __global__ void kernRadixScatter(int n, int *odata, const int *idata, const int *bools, const int *falseIndices) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) {
                return;
            }
            int totalFalses = falseIndices[n - 1] + 1 - bools[n - 1];
            int f = falseIndices[i];
            int dst = bools[i] ? i - f + totalFalses : f;
            odata[dst] = idata[i];
        }

        void sort(int n, int *odata, const int *idata) {
            int N = 1 << ilog2ceil(n);
            int maxValue = 0;
            for (int i = 0; i < n; i++) {
                maxValue = std::max(maxValue, idata[i]);
            }
            int numBits = ilog2ceil(maxValue + 1);
            int *dev_a;
            int *dev_b;
            int *dev_bools;
            int *dev_false;
            cudaMalloc(&dev_a, n * sizeof(int));
            cudaMalloc(&dev_b, n * sizeof(int));
            cudaMalloc(&dev_bools, n * sizeof(int));
            cudaMalloc(&dev_false, N * sizeof(int));
            cudaMemcpy(dev_a, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            dim3 blocks((n + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            for (int bit = 0; bit < numBits; bit++) {
                cudaMemset(dev_false + n, 0, (N - n) * sizeof(int));
                kernMapBit<<<blocks, blockSize>>>(n, bit, dev_bools, dev_false, dev_a);
                Efficient::scanInPlace(N, dev_false);
                kernRadixScatter<<<blocks, blockSize>>>(n, dev_b, dev_a, dev_bools, dev_false);
                std::swap(dev_a, dev_b);
            }
            checkCUDAError("radix sort failed");
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_a, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_a);
            cudaFree(dev_b);
            cudaFree(dev_bools);
            cudaFree(dev_false);
        }
    }
}
