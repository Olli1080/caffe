#include "caffe/layers/clip_layer.hpp"

#include <algorithm>
#include <vector>

#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
void ClipLayer<Dtype>::Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
    const std::vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();

  Dtype min = this->layer_param_->clip_param().min();
  Dtype max = this->layer_param_->clip_param().max();

  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(min, std::min(bottom_data[i], max));
  }
}

template <typename Dtype>
void ClipLayer<Dtype>::Backward_cpu(const std::vector<Blob<Dtype>*>& top,
    const std::vector<bool>& propagate_down,
    const std::vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();

    Dtype min = this->layer_param_->clip_param().min();
    Dtype max = this->layer_param_->clip_param().max();

    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * (
              bottom_data[i] >= min && bottom_data[i] <= max);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ClipLayer);
#else
template <typename Dtype>
void ClipLayer<Dtype>::Forward_gpu(const std::vector<Blob<Dtype>*>& bottom,
    const std::vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[0]->count();
    Dtype p_min = this->layer_param_->clip_param().min();
    Dtype p_max = this->layer_param_->clip_param().max();
    // NOLINT_NEXT_LINE(whitespace/operators)
    forward_kernel(count, bottom_data, top_data, p_min, p_max);
    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void ClipLayer<Dtype>::Backward_gpu(const std::vector<Blob<Dtype>*>& top,
    const std::vector<bool>& propagate_down,
    const std::vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* top_diff = top[0]->gpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        const int count = bottom[0]->count();
        Dtype p_min = this->layer_param_->clip_param().min();
        Dtype p_max = this->layer_param_->clip_param().max();
        // NOLINT_NEXT_LINE(whitespace/operators)
        backward_kernel(count, top_diff, bottom_data, bottom_diff, p_min, p_max);
        CUDA_POST_KERNEL_CHECK;
    }
}
extern template void ClipLayer<float>::forward_kernel(int, const float*, float*, float, float);
extern template void ClipLayer<double>::forward_kernel(int, const double*, double*, double, double);

extern template void ClipLayer<float>::backward_kernel(int, const float*, const float*, float*, float, float);
extern template void ClipLayer<double>::backward_kernel(int, const double*, const double*, double*, double, double);
//INSTANTIATE_LAYER_GPU_FUNCS_EXTERN(ClipLayer);
#endif

INSTANTIATE_CLASS(ClipLayer);
REGISTER_LAYER_CLASS(Clip);

}  // namespace caffe
