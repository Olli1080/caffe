#include "caffe/layers/relu_layer.hpp"

#include <algorithm>
#include <vector>

namespace caffe {

template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope);
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::forward_kernel(int count, const Dtype* in, Dtype* out,
    Dtype negative_slope)
{
    ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, in, out, negative_slope);
}

template <typename Dtype>
void ReLULayer<Dtype>::backward_kernel(int count, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope)
{
    ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, in_diff, in_data, out_diff, negative_slope);
}

template void ReLULayer<float>::forward_kernel(int, const float*, float*, float);
template void ReLULayer<double>::forward_kernel(int, const double*, double*, double);

template void ReLULayer<float>::backward_kernel(int, const float*, const float*, float*, float);
template void ReLULayer<double>::backward_kernel(int, const double*, const double*, double*, double);
//#ifdef CPU_ONLY
//STUB_GPU(ReLULayer);
//#else
//INSTANTIATE_LAYER_GPU_FUNCS_EXTERN(ReLULayer);
//#endif

//INSTANTIATE_LAYER_GPU_FUNCS(ReLULayer);


}  // namespace caffe
