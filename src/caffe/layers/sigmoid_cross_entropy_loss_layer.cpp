#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"

#include <algorithm>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const std::vector<Blob<Dtype>*>& bottom, const std::vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  has_ignore_label_ =
    this->layer_param_->loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_->loss_param().ignore_label();
  }
  if (this->layer_param_->loss_param().has_normalization()) {
    normalization_ = this->layer_param_->loss_param().normalization();
  } else if (this->layer_param_->loss_param().has_normalize()) {
    normalization_ = this->layer_param_->loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = LossParameter_NormalizationMode_BATCH_SIZE;
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const std::vector<Blob<Dtype>*>& bottom, const std::vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->shape(0);  // batch size
  inner_num_ = bottom[0]->count(1);  // instance size: |output| == |target|
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

// TODO(shelhamer) loss normalization should be pulled up into LossLayer,
// instead of duplicated here and in SoftMaxWithLossLayer
template <typename Dtype>
Dtype SigmoidCrossEntropyLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const std::vector<Blob<Dtype>*>& bottom, const std::vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  int valid_count = 0;
  Dtype loss = 0;
  for (int i = 0; i < bottom[0]->count(); ++i) {
    const int target_value = static_cast<int>(target[i]);
    if (has_ignore_label_ && target_value == ignore_label_) {
      continue;
    }
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    ++valid_count;
  }
  normalizer_ = get_normalizer(normalization_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer_;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const std::vector<Blob<Dtype>*>& top, const std::vector<bool>& propagate_down,
    const std::vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    // Zero out gradient of ignored targets.
    if (has_ignore_label_) {
      for (int i = 0; i < count; ++i) {
        const int target_value = static_cast<int>(target[i]);
        if (target_value == ignore_label_) {
          bottom_diff[i] = 0;
        }
      }
    }
    // Scale down gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidCrossEntropyLossLayer);
#else
template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // The forward pass computes the sigmoid outputs.
    sigmoid_bottom_vec_[0] = bottom[0];
    sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
    // Compute the loss (negative log likelihood)
    const int count = bottom[0]->count();
    // Stable version of loss computation from input data
    const Dtype* input_data = bottom[0]->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    // Since this memory is not used for anything, we use it here to avoid having
    // to allocate new GPU memory to accumulate intermediate results.
    Dtype* loss_data = bottom[0]->mutable_gpu_diff();
    Dtype* count_data = bottom[1]->mutable_gpu_diff();
    Dtype valid_count;
    // NOLINT_NEXT_LINE(whitespace/operators)
    forward_kernel(count, input_data, target, loss_data, count_data);
    // Only launch another CUDA kernel if we actually need the valid count.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
        caffe_gpu_asum(count, count_data, &valid_count);
    }
    else {
        valid_count = count;
    }
    Dtype loss;
    caffe_gpu_asum(count, loss_data, &loss);
    normalizer_ = get_normalizer(normalization_, valid_count);
    top[0]->mutable_cpu_data()[0] = loss / normalizer_;

    // Clear scratch memory to prevent interfering with backward (see #6202).
    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
        LOG(FATAL) << this->type()
            << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
        // First, compute the diff
        const int count = bottom[0]->count();
        const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
        const Dtype* target = bottom[1]->gpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        caffe_copy(count, sigmoid_output_data, bottom_diff);
        caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
        // Zero out gradient of ignored targets.
        if (has_ignore_label_) {
            // NOLINT_NEXT_LINE(whitespace/operators)
            backward_kernel(count, target, bottom_diff);
        }
        // Scale down gradient
        Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
        caffe_gpu_scal(count, loss_weight, bottom_diff);
    }
}

extern template void SigmoidCrossEntropyLossLayer<float>::forward_kernel(const int, const float*, const float*, float*, float*);
extern template void SigmoidCrossEntropyLossLayer<double>::forward_kernel(const int, const double*, const double*, double*, double*);

extern template void SigmoidCrossEntropyLossLayer<float>::backward_kernel(const int, const float*, float*);
extern template void SigmoidCrossEntropyLossLayer<double>::backward_kernel(const int, const double*, double*);
#endif

INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);

}  // namespace caffe
