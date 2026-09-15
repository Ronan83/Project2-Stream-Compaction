CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 2**

* Xuan Zhu
  * [GitHub](https://github.com/Ronan83)
* Tested on: Windows 11, AMD Ryzen AI 7 350, 32GB DDR5, NVIDIA GeForce RTX 5070 Laptop GPU (Personal computer)
* Built with CUDA 13.3, Visual Studio 2022, Release configuration, run without the debugger attached.

## Overview

This project implements exclusive prefix sum (scan) and stream compaction on both the CPU and the GPU, and uses the scan as a building block for a GPU radix sort. The point of the exercise is to compare implementations that differ in asymptotic work, memory traffic, and kernel launch count, and to see which of those actually determines runtime on real hardware.

### Features

* Serial CPU exclusive scan
* CPU stream compaction without scan (single write pointer)
* CPU stream compaction via map, scan, and scatter
* Naive GPU scan, `O(N log N)` work, ping-pong buffers
* Work-efficient GPU scan, `O(N)` work, up-sweep and down-sweep over a balanced tree
* Work-efficient GPU stream compaction
* Thrust `exclusive_scan` as a library baseline
* Thread-compacted launch configuration for the work-efficient scan (extra credit, see Part 5)
* GPU radix sort for non-negative integers, built on the work-efficient scan (extra credit)
* Power-of-two and non-power-of-two input support throughout

Everything lives in the `stream_compaction` subproject so it can be lifted into the path tracer later.

## Implementation Notes

**CPU scan** is a single accumulating loop, as the instructions ask. Because `compactWithScan` needs a scan internally and the performance timer cannot be started twice, the loop is factored into a timer-free `scanImpl` that both entry points call.

**Naive GPU scan** launches one kernel per doubling offset, so `ilog2ceil(n)` launches total. Threads cannot be synchronized across blocks, which is why each level is its own launch rather than a loop inside one kernel. Two device buffers are swapped each level so no thread reads a location another thread is writing. The algorithm produces an inclusive scan, so the copy back to the host shifts right by one and writes a zero into element 0.

**Work-efficient GPU scan** rounds the working array up to the next power of two and zeroes the padding so it does not contribute to any partial sum. Up-sweep reduces into the right child of each node; the root is then cleared, which is what turns the result into an exclusive scan; down-sweep pushes prefixes back down. The pass is done in place, since no thread ever reads a location another thread writes in the same level.

**Stream compaction** maps to a boolean array, scans it to get destination indices, and scatters. The element count is `indices[n-1] + bools[n-1]`, because an exclusive scan does not account for the last element; on the GPU those two values are read back with two small `cudaMemcpy` calls. The scan buffer is allocated at the padded size and is kept separate from the boolean array, since the scan runs in place and would otherwise destroy the flags before the count can be computed.

**Thrust scan** wraps `thrust::exclusive_scan`. The `device_vector` construction and the copy back sit outside the timer so the measurement covers only the scan itself.

## Performance Analysis

All numbers below come from Release builds run without the debugger. GPU timings use CUDA events and exclude `cudaMalloc` and the initial and final `cudaMemcpy`; CPU timings use `std::chrono`.

A word on measurement noise, because it affects how the rest of this section should be read. Repeating the same binary on the same input produced run-to-run spreads of up to 30% for both the CPU and the GPU scans. For example, the work-efficient scan at `N = 2^20` with a block size of 256 measured 0.475 ms on one run and 0.635 ms on the next. Differences smaller than roughly 0.15 ms at this input size are therefore not meaningful, and I have tried not to draw conclusions from them.

### Block Size

Block size was swept across five values at `N = 2^20`, with the same value applied to the naive scan, the work-efficient scan, and radix sort. Where two numbers appear, they are two separate runs of the same binary.

| Block size | Naive scan (ms) | Work-efficient scan (ms) |
| ---------- | --------------- | ------------------------ |
| 32         | 0.779           | 0.441                    |
| 64         | 0.460           | 0.548                    |
| 128        | 0.384 / 0.344   | 0.496 / 0.504            |
| 256        | 0.408 / 0.387   | 0.475 / 0.635            |
| 512        | 0.357 / 0.329   | 0.475 / 0.565            |

The naive scan shows one effect that is clearly larger than the noise: a block size of 32 is about twice as slow as anything else. At 32 threads a block is a single warp, so an SM has far fewer resident warps available to switch to while a memory request is outstanding. Since every thread in this kernel does one global load, one add, and one global store, there is nothing to hide the latency behind, and throughput drops. From 128 upward the naive scan is flat within noise.

The work-efficient scan shows no trend at all across the whole sweep. Every value falls between 0.44 and 0.64 ms, which is narrower than the spread I measured by re-running a single configuration. This is itself informative: occupancy is not what limits this kernel, so changing the block size does not move it.

**Block size 128 was used for all remaining measurements.** It is tied for fastest on the naive scan and indistinguishable from the rest on the work-efficient scan.

### Scan vs. Array Size

All four implementations, block size 128, power-of-two inputs.

| N        | CPU (ms) | Naive (ms) | Work-efficient (ms) | Thrust (ms) |
| -------- | -------- | ---------- | ------------------- | ----------- |
| 2^12     | 0.0021   | 0.124      | 0.277               | 0.113       |
| 2^16     | 0.0176   | 0.195      | 0.306               | 0.076       |
| 2^20     | 0.347    | 0.384      | 0.496               | 0.360       |
| 2^22     | 1.096    | 1.084      | 0.611               | 0.518       |
| 2^24     | 4.541    | 10.639     | 4.413               | 1.027       |

Three things stand out.

**The CPU wins until roughly 2^22.** At 2^12 it is more than fifty times faster than any GPU version. The GPU curves are nearly flat from 2^12 to 2^16 even though the input grows sixteen-fold, which says the GPU is not doing meaningful work in that range at all: the time is kernel launch and synchronization overhead. A scan at 2^12 is roughly 4000 additions, and the GPU cannot amortize twelve to twenty-four kernel launches over that.

**The naive scan collapses at 2^24**, going from roughly parity with the CPU at 2^22 to 2.3x slower at 2^24. Its work is `O(N log N)`, and more importantly every one of its 24 levels reads and writes the entire array in global memory. That is about 24 x 2 x 64 MB of traffic for a kernel whose only arithmetic is a single integer add per element. It is bandwidth-bound, and the bandwidth requirement grows as `N log N` while the work-efficient version's grows as `N`.

**The work-efficient scan is slower than naive below 2^22 and faster above it.** Doing less total work does not help at small sizes because it costs more launches, not fewer: the up-sweep and down-sweep together need `2 * log2(N)` kernels versus the naive version's `log2(N)`. At 2^12 that is 24 launches against 12, and when launch overhead dominates, the algorithm with more launches loses despite doing less work. The asymptotic advantage only shows up once there is enough data for the memory traffic to matter more than the launch count.

**Thrust is fastest at every size above 2^12 and is 4.3x faster than my best implementation at 2^24.** It does the scan in a small fixed number of kernels rather than one per tree level, which means it must be staging partial results in shared memory within each block and then combining block sums, instead of round-tripping the whole array through global memory `2 log N` times. That is exactly the structural difference between my implementation and a production one, and it is the single largest factor in the comparison.

### Stream Compaction vs. Array Size

| N        | CPU without scan (ms) | CPU with scan (ms) | Work-efficient GPU (ms) |
| -------- | --------------------- | ------------------ | ----------------------- |
| 2^12     | 0.008                 | 0.012              | 0.296                   |
| 2^16     | 0.111                 | 0.179              | 0.350                   |
| 2^20     | 1.789                 | 3.010              | 0.511                   |
| 2^22     | 7.058                 | 15.198             | 1.035                   |
| 2^24     | 26.111                | 55.232             | 5.934                   |

Compaction crosses over much earlier than scan, around 2^18 to 2^20, and the GPU margin at 2^24 is 4.4x over the better CPU version. The reason is that compaction is a worse fit for a CPU than scan is. `compactWithoutScan` branches on every element and writes to a moving pointer, so it mispredicts constantly on random data; `compactWithScan` is worse still, roughly 2x slower than the direct version, because it makes three passes over the array and materializes two temporary arrays instead of one. The GPU version does the same three phases, but the map and scatter are perfectly parallel and the branch becomes a predicated store.

This is also why the map-scan-scatter formulation is worth learning even though it looks like a pointless detour on a CPU. The serial version is slower precisely because the formulation is designed for a machine that cannot maintain a single shared write pointer.

### Where the Time Goes

Summarizing the bottleneck for each implementation:

* **CPU scan**: bound by serial dependency, one add per element with no parallelism available. It scales linearly and cleanly, which is why it stays competitive far longer than one might expect.
* **Naive GPU scan**: global memory bandwidth. Every level touches the whole array, and the arithmetic intensity is one add per two loads and one store.
* **Work-efficient GPU scan**: at small sizes, kernel launch overhead (`2 log N` launches). At large sizes, global memory bandwidth plus poor coalescing. Within a level, active threads touch addresses `stride * 2` apart, so at large strides a warp's 32 accesses fall in 32 different cache lines and generate 32 separate transactions instead of a handful of coalesced ones.
* **Thrust**: whatever it is bound by, it is not the round trips my version makes. Its advantage grows with `N`, which is consistent with it moving far less data through global memory.

## Extra Credit: Thread-Compacted Work-Efficient Scan (Part 5)

The obvious implementation launches `paddedN` threads at every level and has each thread test `index % (stride * 2) == 0` to decide whether it participates. At the deepest levels this launches millions of threads so that a handful can work. The fix is to launch only the threads that are needed and map each one directly onto its node:

```
int numThreads = paddedN / (stride * 2);
...
int index = (tid + 1) * stride * 2 - 1;
data[index] += data[index - stride];
```

This removes the modulo entirely and leaves no idle threads. Both versions were built and measured; the only difference between the two builds is these three functions.

| Configuration | 2^20 POT | 2^20 NPOT | 2^24 POT | 2^24 NPOT |
| ------------- | -------- | --------- | -------- | --------- |
| Modulo guard, full launch | 0.4963 | 0.3911 | 4.4128 | 4.3049 |
| Compacted threads         | 0.4295 | 0.4765 | 4.4103 | 4.3500 |

**The optimization produced no measurable speedup on this GPU.** At 2^24 the two versions differ by 0.06%, far inside the run-to-run variation documented above, and in two of the four cells the "optimized" version is nominally slower. I am reporting this as a negative result rather than picking a favorable pair of runs.

The explanation follows from the bottleneck analysis. The optimization targets occupancy, and occupancy is not what limits this kernel. Two specific reasons:

First, the memory traffic is unchanged. Active threads touch exactly the same addresses in the same scattered pattern in both versions, so the number of memory transactions and their coalescing behavior are identical. Only the count of threads that return immediately without touching memory changes.

Second, the levels where the optimization helps most are the levels that matter least. Memory traffic per level halves as `stride` doubles, so the first level of the up-sweep alone accounts for half the total traffic, and at that level the naive version is already 50% efficient (8M active threads out of 16M launched). The deep levels where the naive version launches millions of threads for a few dozen workers contribute almost nothing to the total time, because they move almost no data.

What this points to is that reducing the amount of data moved through global memory, rather than the number of threads launched, is where the remaining performance is. That is what the Thrust comparison shows and what a shared-memory implementation would address.

## Extra Credit: Radix Sort

A GPU radix sort for non-negative integers, implemented with the same map-scan-scatter machinery. For each bit, `e[i]` is set to 1 where the bit is 0, `e` is scanned into `f` with the work-efficient scan, and `totalFalses = f[n-1] + e[n-1]` gives the number of elements with a 0 bit. Each element then goes to `f[i]` if its bit is 0, or `i - f[i] + totalFalses` if it is 1, which is a stable partition. Repeating over all bits from least to most significant sorts the array.

The number of passes is chosen from the data: the host scans the input for its maximum and runs only `ilog2ceil(maxVal + 1)` passes rather than a fixed 32. For the test data, which is uniform over `[0, 50)`, this is 6 passes instead of 32.

Called as:

```
StreamCompaction::Radix::sort(n, odata, idata);
```

Example, with `SIZE` reduced so the whole array prints:

```
    [  42  48  45  24  34  46  12  30  13  29   8  25  15 ...  48   0 ]
==== radix sort, power-of-two ====
   elapsed time: 1.45933ms    (CUDA Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    passed
```

Correctness is checked against `std::sort` on a copy of the input, for both power-of-two and non-power-of-two lengths.

Timings:

| N    | Radix sort (ms) |
| ---- | --------------- |
| 2^12 | 1.459           |
| 2^16 | 2.689           |
| 2^20 | 2.837           |
| 2^22 | 5.452           |
| 2^24 | 36.703          |

This is not fast, and it is worth being clear about why. Each pass runs a complete work-efficient scan, which is `2 log N` kernel launches on its own, plus a map and a scatter. Worse, each pass reads `totalFalses` back to the host with two `cudaMemcpy` calls, and each of those is a synchronization point that drains the pipeline. Six passes therefore cost six full pipeline stalls plus roughly 6 x (2 log N + 2) launches. Keeping `totalFalses` on the device and computing the scatter offsets in a kernel would remove the stalls; that is the obvious next change.

The value of the exercise is structural rather than performance: it shows that scan is a primitive other parallel algorithms are built out of, which is the same reason it is worth optimizing for the path tracer.

The implementation handles non-negative integers only. Negative values would sort incorrectly because the sign bit in two's complement inverts the ordering; flipping the sign bit before sorting and again afterward would fix this.

## Build Notes

`stream_compaction/CMakeLists.txt` was modified only to add `radix.h` and `radix.cu` to `SOURCE_FILES`. No other `CMakeLists.txt` changes were made.

The build does require one extra compiler flag on Windows. CUDA 13.3's bundled CCCL headers fail to compile under MSVC's traditional preprocessor with `fatal error C1189`, so the standard-conforming preprocessor has to be enabled. Configure with:

```
cmake .. -DCMAKE_CUDA_FLAGS="-Xcompiler /Zc:preprocessor -Xcompiler /utf-8"
```

The `/utf-8` half is cosmetic; it suppresses the `warning C4819` flood that CUDA's headers produce under a non-UTF-8 system code page.

`scanDevice` was added to `stream_compaction/efficient.h` so radix sort can reuse the device-side scan without going through the host or starting a second timer.

## Test Output

`N = 2^20`, `NPOT = 2^20 - 3`, block size 128. Radix sort tests are additions to the provided test program; everything else is the stock test suite.

```
****************
** SCAN TESTS **
****************
    [  16  25  40  43  32   3   8  14  38   8  26  31  39 ...  36   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.2718ms    (std::chrono Measured)
    [   0  16  41  81 124 156 159 167 181 219 227 253 284 ... 25665224 25665260 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.2561ms    (std::chrono Measured)
    [   0  16  41  81 124 156 159 167 181 219 227 253 284 ... 25665122 25665142 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.344064ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.321376ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.503936ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.390848ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.393696ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.31792ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   1   0   3   2   3   2   2   0   0   0   3   1 ...   2   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 1.7668ms    (std::chrono Measured)
    [   1   3   2   3   2   2   3   1   3   1   2   2   2 ...   1   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 1.745ms    (std::chrono Measured)
    [   1   3   2   3   2   2   3   1   3   1   2   2   2 ...   2   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 3.3368ms    (std::chrono Measured)
    [   1   3   2   3   2   2   3   1   3   1   2   2   2 ...   1   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.543616ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.508064ms    (CUDA Measured)
    passed

*********************
** RADIX SORT TESTS**
*********************
    [  16  25  40  43  32   3   8  14  38   8  26  31  39 ...  36   0 ]
==== radix sort, power-of-two ====
   elapsed time: 3.47376ms    (CUDA Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    passed
==== radix sort, non-power-of-two ====
   elapsed time: 2.53066ms    (CUDA Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    passed
```
