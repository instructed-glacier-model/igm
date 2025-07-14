#define EIGEN_USE_GPU

#include <cuda_runtime.h>  // CUDA API
#include <cuda.h>          // CUDA driver

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_device_functions.h"


// Required for Eigen::GpuDevice
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using namespace tensorflow;

__global__ void Interpolate2DKernel(float* output, const float* grid_values,
                                    const float* particle_coords,
                                    int batch, int height, int width, int num_particles) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;

    if (pid >= num_particles) return;

    float y_pos = particle_coords[pid * 2 + 0];
    float x_pos = particle_coords[pid * 2 + 1];

    int x1 = int(x_pos);
    int y1 = int(y_pos);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    float dx = x2 - x1;
    float dy = y2 - y1;

    float x_left_weight = (x_pos - x1) / dx;
    float x_right_weight = (x2 - x_pos) / dx;
    float y_bottom_weight = (y_pos - y1) / dy;
    float y_top_weight = (y2 - y_pos) / dy;

    for (int d = 0; d < batch; ++d) {
        float Q11 = grid_values[d * height * width + y1 * width + x1];
        float Q12 = grid_values[d * height * width + y2 * width + x1];
        float Q21 = grid_values[d * height * width + y1 * width + x2];
        float Q22 = grid_values[d * height * width + y2 * width + x2];

        float R1 = x_left_weight * Q21 + x_right_weight * Q11;
        float R2 = x_left_weight * Q22 + x_right_weight * Q12;
        float P = y_bottom_weight * R2 + y_top_weight * R1;

        output[d * num_particles + pid] = P;
    }
}

void LaunchInterpolate2DKernel(float* output, const float* grid_values,
                               const float* particle_coords,
                               int batch, int height, int width, int num_particles,
                               const Eigen::GpuDevice& d) {
    int block_size = 256;
    int num_blocks = (num_particles + block_size - 1) / block_size;

    Interpolate2DKernel<<<num_blocks, block_size, 0, d.stream()>>>(
        output, grid_values, particle_coords,
        batch, height, width, num_particles
    );
}
