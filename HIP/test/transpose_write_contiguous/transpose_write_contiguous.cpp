#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

#include <hip/hip_runtime.h>

#define TILE_SIZE 32

// __global__
// void transpose_kernel_write_contiguous(const double* __restrict input,
//                                        double* __restrict output,
//                                        const int height,
//                                        const int width);

/* Basic version with write contiguous memory
 Assume a 3 x 4 matrix (height = 3, width = 4) stored row‑major:
 After transposition we want a 4 x 3 matrix, also stored row‑major:
 height = 3, width = 4   height = 4, width = 3
 output (row‑major)  input(row_major)
 | 0  1  2  3 |       |  0  4  8 |
 | 4  5  6  7 |       |  1  5  9 |
 | 8  9 10 11 |       |  2  6 10 |
                      |  3  7 11 |
reading -- 0 4 8 1 5 9 2 6 10 3 7 11
writing -- 0 1 2 3 4 5 6 7 8 9 10 11
*/

#define GIDX(y, x, sizex) y * sizex + x

__global__ void transpose_kernel_write_contiguous(
  const double* __restrict__ input, double* __restrict__ output,
  int srcHeight, int srcWidth) {
    // Calculate destination global thread indices
    const int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    const int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    const int dstWidth = srcHeight;
    const int dstHeight = srcWidth;

    // Boundary check
    if (dstY < dstHeight && dstX < dstWidth) {
        // Transpose: output[y][x] = input[x][y]
        const int input_gid = GIDX(dstX,dstY,srcWidth); // flipped axis
        const int output_gid = GIDX(dstY,dstX,dstWidth);

        output[output_gid] = input[input_gid];
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
    std::cout << "AMD GPU Write Contiguous Matrix Transpose Benchmark" << std::endl;
    std::cout << "===================================================" << std::endl;

    int iterations = 5;

    // Test different matrix sizes
    std::vector<std::pair<int, int>> test_sizes = {
        {256, 256},
        {512, 512},
        {1024, 1024},
        {2048, 2048},
        {4096, 4096},
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
        transpose_kernel_write_contiguous<<<grid_size, block_size>>>(d_input, d_output, height, width);
        hipCheck( hipDeviceSynchronize() );

        // Time the kernel execution
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            transpose_kernel_write_contiguous<<<grid_size, block_size>>>(d_input, d_output, height, width);
        }

        hipCheck( hipDeviceSynchronize() );
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float time_basic_write_contiguous = duration.count() / static_cast<float>(iterations);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Basic Transpose, Write Contiguous - Average Time: " << time_basic_write_contiguous << " μs" << std::endl;

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
