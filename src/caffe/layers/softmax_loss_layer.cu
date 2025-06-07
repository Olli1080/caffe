#include "caffe/layers/softmax_loss_layer.hpp"

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                      Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::forward_kernel(const int nthreads,
    const Dtype* prob_data, const Dtype* label, Dtype* loss,
    const int dim,
    Dtype* counts)
{
    SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::backward_kernel(const int nthreads, const Dtype* top,
    const Dtype* label, Dtype* bottom_diff, const int dim, Dtype* counts)
{
    SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top, label, bottom_diff,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
}

template void SoftmaxWithLossLayer<float>::forward_kernel(const int, const float*, const float*, float*, const int, float*);
template void SoftmaxWithLossLayer<double>::forward_kernel(const int, const double*, const double*, double*, const int, double*);

template void SoftmaxWithLossLayer<float>::backward_kernel(const int, const float*, const float*, float*, const int, float*);
template void SoftmaxWithLossLayer<double>::backward_kernel(const int, const double*, const double*, double*, const int, double*);

}  // namespace caffe
