#include "caffe/layers/elu_layer.hpp"

#include <algorithm>
#include <vector>

#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
void ELULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype alpha = this->layer_param_->elu_param().alpha();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + alpha * (exp(std::min(bottom_data[i], Dtype(0))) - Dtype(1));
  }
}

template <typename Dtype>
void ELULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype alpha = this->layer_param_->elu_param().alpha();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + (alpha + top_data[i]) * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ELULayer);
#else
template <typename Dtype>
void ELULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[0]->count();
    Dtype alpha = this->layer_param_->elu_param().alpha();
    // NOLINT_NEXT_LINE(whitespace/operators)
    forward_kernel(count, bottom_data, top_data, alpha);
    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void ELULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* top_diff = top[0]->gpu_diff();
        const Dtype* top_data = top[0]->gpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        const int count = bottom[0]->count();
        Dtype alpha = this->layer_param_->elu_param().alpha();
        // NOLINT_NEXT_LINE(whitespace/operators)
        backward_kernel(count, top_diff, top_data, bottom_data, bottom_diff, alpha);
        CUDA_POST_KERNEL_CHECK;
    }
}
extern template void ELULayer<float>::forward_kernel(int, const float*, float*, float);
extern template void ELULayer<double>::forward_kernel(int, const double*, double*, double);

extern template void ELULayer<float>::backward_kernel(int, const float*, const float*, const float*, float*, float);
extern template void ELULayer<double>::backward_kernel(int, const double*, const double*, const double*, double*, double);

#endif

INSTANTIATE_CLASS(ELULayer);
REGISTER_LAYER_CLASS(ELU);

}  // namespace caffe
