#include "caffe/layers/softmax_loss_layer.hpp"

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const std::vector<Blob<Dtype>*>& bottom, const std::vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(*this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_->loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_->loss_param().ignore_label();
  }
  if (!this->layer_param_->loss_param().has_normalization() &&
      this->layer_param_->loss_param().has_normalize()) {
    normalization_ = this->layer_param_->loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_->loss_param().normalization();
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const std::vector<Blob<Dtype>*>& bottom, const std::vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_->softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::get_normalizer(
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
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const std::vector<Blob<Dtype>*>& bottom, const std::vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
      ++count;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const std::vector<Blob<Dtype>*>& top,
    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          ++count;
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#else
template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const std::vector<Blob<Dtype>*>& bottom, const std::vector<Blob<Dtype>*>& top) {
    softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is not used for anything, we use it here to avoid having
    // to allocate new GPU memory to accumulate intermediate results.
    Dtype* loss_data = bottom[0]->mutable_gpu_diff();
    // Similarly, this memory is never used elsewhere, and thus we can use it
    // to avoid having to allocate additional GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    forward_kernel(nthreads, prob_data, label, loss_data, dim, counts);
    Dtype loss;
    caffe_gpu_asum(nthreads, loss_data, &loss);
    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
        caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
        valid_count);
    if (top.size() == 2) {
        top[1]->ShareData(prob_);
    }

    // Clear scratch memory to prevent interfering with backward (see #6202).
    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const std::vector<Blob<Dtype>*>& top,
    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
        LOG(FATAL) << this->type()
            << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        const Dtype* prob_data = prob_.gpu_data();
        const Dtype* top_data = top[0]->gpu_data();
        caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
        const Dtype* label = bottom[1]->gpu_data();
        const int dim = prob_.count() / outer_num_;
        const int nthreads = outer_num_ * inner_num_;
        // Since this memory is never used for anything else,
        // we use to to avoid allocating new GPU memory.
        Dtype* counts = prob_.mutable_gpu_diff();
        // NOLINT_NEXT_LINE(whitespace/operators)
        backward_kernel(nthreads, top_data, label, bottom_diff, dim, counts);

        Dtype valid_count = -1;
        // Only launch another CUDA kernel if we actually need the count of valid
        // outputs.
        if (normalization_ == LossParameter_NormalizationMode_VALID &&
            has_ignore_label_) {
            caffe_gpu_asum(nthreads, counts, &valid_count);
        }
        const Dtype loss_weight = top[0]->cpu_diff()[0] /
            get_normalizer(normalization_, valid_count);
        caffe_gpu_scal(prob_.count(), loss_weight, bottom_diff);
    }
}
extern template void SoftmaxWithLossLayer<float>::forward_kernel(const int, const float*, const float*, float*, const int, float*);
extern template void SoftmaxWithLossLayer<double>::forward_kernel(const int, const double*, const double*, double*, const int, double*);

extern template void SoftmaxWithLossLayer<float>::backward_kernel(const int, const float*, const float*, float*, const int, float*);
extern template void SoftmaxWithLossLayer<double>::backward_kernel(const int, const double*, const double*, double*, const int, double*);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe
