#include "caffe/layers/swish_layer.hpp"

#include <cmath>
#include <vector>

#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SwishBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, const Dtype* sigmoid_output_data, Dtype* out_diff,
    const Dtype beta) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype swish_x = out_data[index];
    out_diff[index] = in_diff[index] * (beta * swish_x
        + sigmoid_output_data[index] * (1 - beta * swish_x));
  }
}

template <typename Dtype>
void SwishLayer<Dtype>::backward_kernel(const int n, const Dtype* in_diff,
    const Dtype* out_data, const Dtype* sigmoid_output_data, Dtype* out_diff,
    const Dtype beta)
{
    SwishBackward<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
        n, in_diff, out_data, sigmoid_output_data, out_diff, beta);
}

template void SwishLayer<float>::backward_kernel(const int, const float*, const float*, const float*, float*, const float);
template void SwishLayer<double>::backward_kernel(const int, const double*, const double*, const double*, double*, const double);
}  // namespace caffe
