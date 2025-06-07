#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"

#include <vector>

#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void SigmoidCrossEntropyLossForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* target, Dtype* loss,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
    if (has_ignore_label_ && target_value == ignore_label_) {
      loss[i] = 0;
      counts[i] = 0;
    } else {
      loss[i] = input_data[i] * (target[i] - (input_data[i] >= 0)) -
          log(1 + exp(input_data[i] - 2 * input_data[i] *
          (input_data[i] >= 0)));
      counts[i] = 1;
    }
  }
}

template <typename Dtype>
__global__ void SigmoidCrossEntropyLossIgnoreDiffGPU(const int count,
    const int ignore_label, const Dtype* target, Dtype* diff) {
  CUDA_KERNEL_LOOP(i, count) {
    const int target_value = static_cast<int>(target[i]);
    if (target_value == ignore_label) {
      diff[i] = 0;
    }
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::forward_kernel(const int nthreads,
    const Dtype* input_data, const Dtype* target, Dtype* loss,
    Dtype* counts)
{
    SigmoidCrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, input_data, target, loss,
      has_ignore_label_, ignore_label_, counts);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::backward_kernel(const int count, const Dtype* target, Dtype* diff)
{
    SigmoidCrossEntropyLossIgnoreDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, ignore_label_, target, diff);
}

template void SigmoidCrossEntropyLossLayer<float>::forward_kernel(const int, const float*, const float*, float*, float*);
template void SigmoidCrossEntropyLossLayer<double>::forward_kernel(const int, const double*, const double*, double*, double*);

template void SigmoidCrossEntropyLossLayer<float>::backward_kernel(const int, const float*, float*);
template void SigmoidCrossEntropyLossLayer<double>::backward_kernel(const int, const double*, double*);
//INSTANTIATE_LAYER_GPU_FUNCS(SigmoidCrossEntropyLossLayer);

}  // namespace caffe
