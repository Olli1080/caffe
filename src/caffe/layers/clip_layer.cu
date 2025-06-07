#include "caffe/layers/clip_layer.hpp"

#include <vector>

#include "caffe/util/device_alternate.hpp"

//#include "caffe/layers/clip_layer.hpp"
//#include "caffe/util/math_functions.hpp"

namespace caffe {

__global__ void ClipForward(const int n, const float* in, float* out,
    float p_min, float p_max) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = fmaxf(p_min, fminf(in[index], p_max));
  }
}

__global__ void ClipForward(const int n, const double* in, double* out,
    double p_min, double p_max) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = fmax(p_min, fmin(in[index], p_max));
  }
}

template <typename Dtype>
__global__ void ClipBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype p_min, Dtype p_max) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * (
            in_data[index] >= p_min && in_data[index] <= p_max);
  }
}

template <typename Dtype>
void ClipLayer<Dtype>::forward_kernel(int count, const Dtype* bottom_data, Dtype* top_data, Dtype p_min, Dtype p_max)
{
    ClipForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, p_min, p_max);
}

template<typename Dtype>
void ClipLayer<Dtype>::backward_kernel(int count, const Dtype* top_diff, const Dtype* bottom_data, Dtype* bottom_diff, Dtype p_min, Dtype p_max)
{
	ClipBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		count, top_diff, bottom_data, bottom_diff, p_min, p_max);
}

template void ClipLayer<float>::forward_kernel(int, const float*, float*, float, float);
template void ClipLayer<double>::forward_kernel(int, const double*, double*, double, double);

template void ClipLayer<float>::backward_kernel(int, const float*, const float*, float*, float, float);
template void ClipLayer<double>::backward_kernel(int, const double*, const double*, double*, double, double);
INSTANTIATE_LAYER_GPU_FUNCS(ClipLayer);


}  // namespace caffe
