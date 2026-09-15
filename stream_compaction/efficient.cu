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

        // up-sweep: each node adds its left child into its right child
        __global__ void kernUpSweep(int n, int stride, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            // only the nodes at this level of the tree do work
            if (index % (stride * 2) == 0) {
                data[index + stride * 2 - 1] += data[index + stride - 1];
            }
        }

        // down-sweep: left child gets the parent, right child gets parent + old left
        __global__ void kernDownSweep(int n, int stride, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index % (stride * 2) == 0) {
                int left = data[index + stride - 1];
                data[index + stride - 1] = data[index + stride * 2 - 1];
                data[index + stride * 2 - 1] += left;
            }
        }

        // device-side scan with no timer and no host copies, so compact can reuse it
        // dev_data must already be padded to paddedN and zero-filled past n
        void scanDevice(int paddedN, int* dev_data) {
            dim3 fullBlocks((paddedN + blockSize - 1) / blockSize);

            for (int stride = 1; stride < paddedN; stride *= 2) {
                kernUpSweep << <fullBlocks, blockSize >> > (paddedN, stride, dev_data);
            }

            cudaMemset(dev_data + paddedN - 1, 0, sizeof(int));

            for (int stride = paddedN / 2; stride >= 1; stride /= 2) {
                kernDownSweep << <fullBlocks, blockSize >> > (paddedN, stride, dev_data);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int depth = ilog2ceil(n);
            int paddedN = 1 << depth;   // the tree needs a power-of-two length

            int* dev_data;
            cudaMalloc((void**)&dev_data, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed");

            // zero out the padding so it does not affect the sums
            cudaMemset(dev_data, 0, paddedN * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

        

            timer().startGpuTimer();

            scanDevice(paddedN, dev_data);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy back failed");

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

            int depth = ilog2ceil(n);
            int paddedN = 1 << depth;

            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            int* dev_indices;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc failed");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 fullBlocks((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            Common::kernMapToBoolean << <fullBlocks, blockSize >> > (n, dev_bools, dev_idata);

            // the scan buffer is padded, so zero it first then copy the bools in
            cudaMemset(dev_indices, 0, paddedN * sizeof(int));
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            scanDevice(paddedN, dev_indices);

            Common::kernScatter << <fullBlocks, blockSize >> > (n, dev_odata,
                dev_idata, dev_bools, dev_indices);
            
            timer().endGpuTimer();
            // exclusive scan drops the last element, so read it back and add it
            int lastIndex = 0;
            int lastBool = 0;
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + lastBool;

            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy back failed");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            return count;
        }
    }
}
