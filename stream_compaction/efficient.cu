#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int *data) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) {
                return;
            }
            int k = i << (d + 1);
            data[k + (1 << (d + 1)) - 1] += data[k + (1 << d) - 1];
        }

        __global__ void kernDownSweep(int n, int d, int *data) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) {
                return;
            }
            int k = i << (d + 1);
            int left = k + (1 << d) - 1;
            int right = k + (1 << (d + 1)) - 1;
            int t = data[left];
            data[left] = data[right];
            data[right] += t;
        }

        void scanInPlace(int N, int *dev_data) {
            int logN = ilog2(N);
            for (int d = 0; d < logN; d++) {
                int threads = N >> (d + 1);
                kernUpSweep<<<(threads + blockSize - 1) / blockSize, blockSize>>>(threads, d, dev_data);
            }
            cudaMemset(dev_data + N - 1, 0, sizeof(int));
            for (int d = logN - 1; d >= 0; d--) {
                int threads = N >> (d + 1);
                kernDownSweep<<<(threads + blockSize - 1) / blockSize, blockSize>>>(threads, d, dev_data);
            }
            checkCUDAError("efficient scan failed");
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int N = 1 << ilog2ceil(n);
            int *dev_data;
            cudaMalloc(&dev_data, N * sizeof(int));
            cudaMemset(dev_data, 0, N * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            scanInPlace(N, dev_data);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int N = 1 << ilog2ceil(n);
            int *dev_idata;
            int *dev_odata;
            int *dev_bools;
            int *dev_indices;
            cudaMalloc(&dev_idata, n * sizeof(int));
            cudaMalloc(&dev_odata, n * sizeof(int));
            cudaMalloc(&dev_bools, n * sizeof(int));
            cudaMalloc(&dev_indices, N * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(dev_indices, 0, N * sizeof(int));
            dim3 blocks((n + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            Common::kernMapToBoolean<<<blocks, blockSize>>>(n, dev_bools, dev_idata);
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            scanInPlace(N, dev_indices);
            Common::kernScatter<<<blocks, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            checkCUDAError("efficient compact failed");
            timer().endGpuTimer();
            int lastBool;
            int lastIndex;
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastBool + lastIndex;
            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            return count;
        }
    }
}
