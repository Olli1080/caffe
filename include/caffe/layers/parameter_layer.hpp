#ifndef CAFFE_PARAMETER_LAYER_HPP_
#define CAFFE_PARAMETER_LAYER_HPP_

#include <vector>

#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
class ParameterLayer : public Layer<Dtype> {
 public:
  explicit ParameterLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                  const vector<Blob<Dtype>*>& top) override
  {
    if (this->blobs_.size() > 0) {
      LOG(INFO) << "Skipping parameter initialization";
    } else {
      this->blobs_.resize(1);
      this->blobs_[0].reset(new Blob<Dtype>());
      this->blobs_[0]->Reshape(this->layer_param_.parameter_param().shape());
    }
    top[0]->Reshape(this->layer_param_.parameter_param().shape());
  }

  void Reshape(const vector<Blob<Dtype>*>& bottom,
               const vector<Blob<Dtype>*>& top) override { }

  [[nodiscard]] const char* type() const override { return "Parameter"; }
  [[nodiscard]] int ExactNumBottomBlobs() const override { return 0; }
  [[nodiscard]] int ExactNumTopBlobs() const override { return 1; }

 protected:
  void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                   const vector<Blob<Dtype>*>& top) override
  {
    top[0]->ShareData(*(this->blobs_[0]));
    top[0]->ShareDiff(*(this->blobs_[0]));
  }

  void Backward_cpu(const vector<Blob<Dtype>*>& top,
                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) override
  { }
};

}  // namespace caffe

#endif
