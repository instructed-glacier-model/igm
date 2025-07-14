#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Interpolate2D")
    .Input("grid: float")        // [depth, height, width]
    .Input("particles: float")   // [num_particles, 2]
    .Output("interpolated: float") // [depth, num_particles]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle grid, particles;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &grid));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &particles));

        shape_inference::DimensionHandle depth = c->Dim(grid, 0);
        shape_inference::DimensionHandle num_particles = c->Dim(particles, 0);
        c->set_output(0, c->MakeShape({depth, num_particles}));
        return OkStatus();
    });

#if GOOGLE_CUDA
void LaunchInterpolate2DKernel(float* output, const float* grid_values,
                               const float* particle_coords,
                               int depth, int height, int width, int num_particles,
                               const Eigen::GpuDevice& d);

class Interpolate2DOp : public OpKernel {
public:
    explicit Interpolate2DOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& grid = context->input(0);        // [depth, height, width]
        const Tensor& particles = context->input(1);   // [num_particles, 2]

        OP_REQUIRES(context, grid.dims() == 3, errors::InvalidArgument("grid must be 3D"));
        OP_REQUIRES(context, particles.dims() == 2 && particles.dim_size(1) == 2,
                    errors::InvalidArgument("particles must be [N, 2]"));

        const int depth = grid.dim_size(0);
        const int height = grid.dim_size(1);
        const int width = grid.dim_size(2);
        const int num_particles = particles.dim_size(0);

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, {depth, num_particles}, &output));

        const Eigen::GpuDevice& d = context->eigen_device<Eigen::GpuDevice>();

        LaunchInterpolate2DKernel(
            output->flat<float>().data(),
            grid.flat<float>().data(),
            particles.flat<float>().data(),
            depth, height, width, num_particles, d
        );
    }
};

REGISTER_KERNEL_BUILDER(Name("Interpolate2D").Device(DEVICE_GPU), Interpolate2DOp);
#endif
