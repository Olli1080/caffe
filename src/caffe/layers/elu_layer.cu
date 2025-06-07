#include "caffe/layers/elu_layer.hpp"

#include <algorithm>
#include <vector>

namespace caffe {

template <typename Dtype>
__global__ void ELUForward(const int n, const Dtype* in, Dtype* out,
    Dtype alpha) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] :
        alpha * (exp(in[index]) - 1);
  }
}

template <typename Dtype>
__global__ void ELUBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, const Dtype* in_data,
    Dtype* out_diff, Dtype alpha) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_data[index] > 0 ? in_diff[index] :
        in_diff[index] * (out_data[index] + alpha);
  }
}

template <typename Dtype>
void ELULayer<Dtype>::forward_kernel(int count, const Dtype* in, Dtype* out,
    Dtype alpha)
{
	ELUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, in, out, alpha);
}

template <typename Dtype>
void ELULayer<Dtype>::backward_kernel(int count, const Dtype* in_diff,
    const Dtype* out_data, const Dtype* in_data,
    Dtype* out_diff, Dtype alpha)
{
    ELUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, in_diff, out_data, in_data, out_diff, alpha);
}
template void ELULayer<float>::forward_kernel(int, const float*, float*, float);
template void ELULayer<double>::forward_kernel(int, const double*, double*, double);

template void ELULayer<float>::backward_kernel(int, const float*, const float*, const float*, float*, float);
template void ELULayer<double>::backward_kernel(int, const double*, const double*, const double*, double*, double);

//INSTANTIATE_LAYER_GPU_FUNCS(ELULayer);


}  // namespace caffe
