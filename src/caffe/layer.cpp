#include "caffe/layer.hpp"

#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
Layer<Dtype>::Layer(const LayerParameter& param)
    : layer_param_(std::make_unique<LayerParameter>(param)) {
    // Set phase and copy blobs (if there are any).
    phase_ = param.phase();
    if (layer_param_->blobs_size() > 0) {
        blobs_.resize(layer_param_->blobs_size());
        for (int i = 0; i < layer_param_->blobs_size(); ++i) {
            blobs_[i].reset(new Blob<Dtype>());
            blobs_[i]->FromProto(layer_param_->blobs(i));
        }
    }
}

template <typename Dtype>
Layer<Dtype>::~Layer()
{
}

// Forward and backward wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template <typename Dtype>
Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top) {
  Dtype loss = 0;
  Reshape(bottom, top);
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Forward_cpu(bottom, top);
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->cpu_data();
      const Dtype* loss_weights = top[top_id]->cpu_diff();
      loss += caffe_cpu_dot(count, data, loss_weights);
    }
    break;
  case Caffe::GPU:
    Forward_gpu(bottom, top);
#ifndef CPU_ONLY
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->gpu_data();
      const Dtype* loss_weights = top[top_id]->gpu_diff();
      Dtype blob_loss = 0;
      caffe_gpu_dot(count, data, loss_weights, &blob_loss);
      loss += blob_loss;
    }
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
  return loss;
}

template <typename Dtype>
void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Backward_cpu(top, propagate_down, bottom);
    break;
  case Caffe::GPU:
    Backward_gpu(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
const LayerParameter& Layer<Dtype>::layer_param() const
{
    return *layer_param_;
}

// Serialize LayerParameter to protocol buffer
template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
  param->Clear();
  param->CopyFrom(*layer_param_);
  param->clear_blobs();
  for (int i = 0; i < blobs_.size(); ++i) {
    blobs_[i]->ToProto(param->add_blobs(), write_diff);
  }
}

template <typename Dtype>
void Layer<Dtype>::SetLossWeights(const vector<Blob<Dtype>*>& top)
{
    const int num_loss_weights = layer_param_->loss_weight_size();
    if (num_loss_weights) {
        CHECK_EQ(top.size(), num_loss_weights) << "loss_weight must be "
            "unspecified or specified once per top blob.";
        for (int top_id = 0; top_id < top.size(); ++top_id) {
            const Dtype loss_weight = layer_param_->loss_weight(top_id);
            if (loss_weight == Dtype(0)) { continue; }
            this->set_loss(top_id, loss_weight);
            const int count = top[top_id]->count();
            Dtype* loss_multiplier = top[top_id]->mutable_cpu_diff();
            caffe_set(count, loss_weight, loss_multiplier);
        }
    }
}

INSTANTIATE_CLASS(Layer);

}  // namespace caffe
