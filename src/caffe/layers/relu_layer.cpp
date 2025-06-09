#include "caffe/layers/relu_layer.hpp"

#include <algorithm>
#include <vector>

#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
    const std::vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_->relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const std::vector<Blob<Dtype>*>& top,
    const std::vector<bool>& propagate_down,
    const std::vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_->relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#else
template <typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(const std::vector<Blob<Dtype>*>& bottom,
    const std::vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_->relu_param().negative_slope();
    // NOLINT_NEXT_LINE(whitespace/operators)
    forward_kernel(count, bottom_data, top_data, negative_slope);
    CUDA_POST_KERNEL_CHECK;
    // << " count: " << count << " bottom_data: "
    //     << (unsigned long)bottom_data
    //     << " top_data: " << (unsigned long)top_data
    //     << " blocks: " << CAFFE_GET_BLOCKS(count)
    //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_gpu(const std::vector<Blob<Dtype>*>& top,
    const std::vector<bool>& propagate_down,
    const std::vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* top_diff = top[0]->gpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        const int count = bottom[0]->count();
        Dtype negative_slope = this->layer_param_->relu_param().negative_slope();
        // NOLINT_NEXT_LINE(whitespace/operators)
        backward_kernel(count, top_diff, bottom_data, bottom_diff, negative_slope);
        CUDA_POST_KERNEL_CHECK;
    }
}

extern template void ReLULayer<float>::forward_kernel(int, const float*, float*, float);
extern template void ReLULayer<double>::forward_kernel(int, const double*, double*, double);

extern template void ReLULayer<float>::backward_kernel(int, const float*, const float*, float*, float);
extern template void ReLULayer<double>::backward_kernel(int, const double*, const double*, double*, double);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
