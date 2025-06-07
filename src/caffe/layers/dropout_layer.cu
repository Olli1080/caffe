#include "caffe/layers/dropout_layer.hpp"

#include <vector>

#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DropoutForward(const int n, const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (mask[index] > threshold) * scale;
  }
}

template <typename Dtype>
__global__ void DropoutBackward(const int n, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::forward_kernel(int count, const Dtype* bottom_data, unsigned int* mask, Dtype* top_data)
{
    DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, mask, uint_thres_, scale_, top_data);
}

template <typename Dtype>
void DropoutLayer<Dtype>::backward_kernel(int count, const Dtype* top_diff, const unsigned int* mask, Dtype* bottom_diff)
{
	DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, mask, uint_thres_, scale_, bottom_diff);
}
template void DropoutLayer<float>::forward_kernel(int, const float*, unsigned int*, float*);
template void DropoutLayer<double>::forward_kernel(int, const double*, unsigned int*, double*);

template void DropoutLayer<float>::backward_kernel(int, const float*, const unsigned int*, float*);
template void DropoutLayer<double>::backward_kernel(int, const double*, const unsigned int*, double*);
INSTANTIATE_LAYER_GPU_FUNCS(DropoutLayer);

}  // namespace caffe
