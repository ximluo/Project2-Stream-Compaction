#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 1024

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernNaiveScan(int n, int offset, int *odata, const int *idata) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= n) {
                return;
            }
            if (k >= offset) {
                odata[k] = idata[k - offset] + idata[k];
            } else {
                odata[k] = idata[k];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *dev_a;
            int *dev_b;
            cudaMalloc(&dev_a, n * sizeof(int));
            cudaMalloc(&dev_b, n * sizeof(int));
            cudaMemcpy(dev_a, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            dim3 blocks((n + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            for (int offset = 1; offset < n; offset <<= 1) {
                kernNaiveScan<<<blocks, blockSize>>>(n, offset, dev_b, dev_a);
                std::swap(dev_a, dev_b);
            }
            checkCUDAError("naive scan failed");
            timer().endGpuTimer();
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_a, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_a);
            cudaFree(dev_b);
        }
    }
}
