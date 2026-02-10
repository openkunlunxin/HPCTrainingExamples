#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

#include <hip/hip_runtime.h>

#define TILE_SIZE 32

// __global__
// void transpose_kernel_tiled(const double* __restrict input,
//                             double* __restrict output,
//                             const int height,
//                             const int width);

#define GIDX(y, x, sizex) y * sizex + x
#define PAD 1

/* Use a **shared‑memory tile** (`TILE_SIZE × (TILE_SIZE+PAD)`) to stage the data.
 *    Pad the shared‑memory tile to avoid bank conflicts.
 * Load the tile from the **row‑major source** (contiguous reads).
 * `__syncthreads()`.
 * Write the transposed tile back to the **row‑major destination** (`output[col][row]`),
 *    which is now a **contiguous write** pattern.
 */

__global__ void transpose_kernel_tiled(
   const double* __restrict input, double* __restrict output,
   const int srcHeight, const int srcWidth)
{
    // thread coordinates in the source matrix
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // source global coordinates this thread will read
    const int srcX = blockIdx.x * TILE_SIZE + tx;
    const int srcY = blockIdx.y * TILE_SIZE + ty;

    // allocate a shared (LDS) memory tile with padding to avoid bank conflicts
    __shared__ double tile[TILE_SIZE][TILE_SIZE + PAD];

    // Read from global memory into tile with coalesced reads
    if (srcY < srcHeight && srcX < srcWidth) {
        tile[ty][tx] = input[GIDX(srcY, srcX, srcWidth)];
    } else {
        tile[ty][tx] = 0.0;                // guard value – never used for writes
    }

    // Synchronize to make sure all of the tile is updated before using it
    __syncthreads();

    // destination global coordinates this thread will write
    const int dstY = blockIdx.x * TILE_SIZE + ty; // swapped axes
    const int dstX = blockIdx.y * TILE_SIZE + tx;

    // Write back to global memory with coalesced writes
    if (dstY < srcWidth && dstX < srcHeight) {
        output[GIDX(dstY, dstX, srcHeight)] = tile[tx][ty];
    }
}


// Macro for checking GPU API return values
#define hipCheck(call)                                                                          \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)

int main(int argc, char *argv[])
{
    std::cout << "AMD GPU Tiled Matrix Transpose Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;

    int iterations = 5;

    // Test different matrix sizes
    std::vector<std::pair<int, int>> test_sizes = {
        //{256, 256},
        //{512, 512},
        //{1024, 1024},
        //{2048, 2048},
        //{4096, 4096},
        {8192, 8192}
    };

    for (const auto& size : test_sizes) {
        int height = size.first;
        int width = size.second;

        // Allocate host memory
        double* h_input = new double[height * width];
        double* h_output = new double[width * height];

        // Generate test data
        for (int i = 0; i < height * width; ++i) {
            h_input[i] = static_cast<double>(i % 1000);
        }

        // Allocate device memory
        double *d_input, *d_output;
        size_t input_size = height * width * sizeof(double);
        size_t output_size = width * height * sizeof(double);

        hipCheck( hipMalloc(&d_input, input_size) );
        hipCheck( hipMalloc(&d_output, output_size) );

        // Copy input data to device
        hipCheck( hipMemcpy(d_input, h_input, input_size, hipMemcpyHostToDevice) );

        std::cout << "\nTesting Matrix dimensions: " << height << " x " << width << std::endl;
        std::cout << "Input size: " << input_size / (1024.0 * 1024.0) << " MiB" << std::endl;
        std::cout << "Output size: " << output_size / (1024.0 * 1024.0) << " MiB" << std::endl;
        std::cout << "=========================================" << std::endl;

        dim3 block_size(TILE_SIZE, TILE_SIZE);
        dim3 grid_size((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

        // Warm up
        transpose_kernel_tiled<<<grid_size, block_size>>>(d_input, d_output, height, width);
        hipCheck( hipDeviceSynchronize() );

        // Time the kernel execution
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            transpose_kernel_tiled<<<grid_size, block_size>>>(d_input, d_output, height, width);
        }

        hipCheck( hipDeviceSynchronize() );
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float time_tiled = duration.count() / static_cast<float>(iterations);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Tiled Transpose, Read and Write Contiguous - Average Time: " << time_tiled << " μs" << std::endl;

        std::cout << "=========================================" << std::endl;

        // Copy result back to verify correctness (only for first version)
        hipCheck( hipMemcpy(h_output, d_output, output_size, hipMemcpyDeviceToHost) );

        // Verify correctness
        bool is_correct = true;

        for (int i = 0; i < height && is_correct; ++i) {
            for (int j = 0; j < width; ++j) {
                if (h_input[i * width + j] != h_output[j * height + i]) {
                    is_correct = false;
                    break;
                }
            }
        }

        std::cout << "Verification: " << (is_correct ? "PASSED" : "FAILED") << std::endl;

        // Cleanup
        hipCheck( hipFree(d_input) );
        hipCheck( hipFree(d_output) );

        delete[] h_input;
        delete[] h_output;
    }
}
