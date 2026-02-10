#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

#include <hip/hip_runtime.h>

#define TILE_SIZE 32

// __global__
// void transpose_kernel_read_contiguous(const double* __restrict input,
//                                       double* __restrict output,
//                                       const int height,
//                                       const int width);

/* Basic version with read contiguous memory
 Assume a 4 × 3 matrix (height = 4, width = 3) stored row‑major:
 After transposition we want a 3 × 4 matrix, also stored row‑major:
 height = 4, width = 3, height = 3  width = 4
 input (row‑major)  output(row_major)
 |  0   1   2 |       |  0  3  6  9 |
 |  3   4   5 |       |  1  4  7 10 |
 |  6   7   8 |       |  2  5  8 11 |
 |  9  10  11 |       

reading -- 0 1 2 3 4 5 6 7 8 9 10 11
writing -- 0 3 6 9 1 4 7 10 2 5 8 11
*/

#define GIDX(y, x, sizex) y * sizex + x

__global__ void transpose_kernel_read_contiguous(
  const double* __restrict__ input, double* __restrict__ output,
  int srcHeight, int srcWidth) {
    // Calculate source global thread indices
    const int srcX = blockIdx.x * blockDim.x + threadIdx.x;
    const int srcY = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (srcY < srcHeight && srcX < srcWidth) {
        // Transpose: output[x][y] = input[y][x]
        const int input_gid = GIDX(srcY,srcX,srcWidth);
        const int output_gid = GIDX(srcX,srcY,srcHeight); // flipped axis
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
    std::cout << "AMD GPU Read Contiguous Matrix Transpose Benchmark" << std::endl;
    std::cout << "==================================================" << std::endl;

    int iterations = 10;

    // Test different matrix sizes
    std::vector<std::pair<int, int>> test_sizes = {
       // {256, 256},
       // {512, 512},
       // {1024, 1024},
       // {2048, 2048},
       // {4096, 4096},
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
        transpose_kernel_read_contiguous<<<grid_size, block_size>>>(d_input, d_output, height, width);
        hipCheck( hipDeviceSynchronize() );

        // Time the kernel execution
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            transpose_kernel_read_contiguous<<<grid_size, block_size>>>(d_input, d_output, height, width);
        }

        hipCheck( hipDeviceSynchronize() );
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float time_basic_read_contiguous = duration.count() / static_cast<float>(iterations);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Basic Transpose, Read Contiguous - Average Time: " << time_basic_read_contiguous << " μs" << std::endl;

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
