#include "caffe/layers/parameter_layer.hpp"

#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
void ParameterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    if (this->blobs_.size() > 0) {
        LOG(INFO) << "Skipping parameter initialization";
    }
    else {
        this->blobs_.resize(1);
        this->blobs_[0].reset(new Blob<Dtype>());
        this->blobs_[0]->Reshape(this->layer_param_->parameter_param().shape());
    }
    top[0]->Reshape(this->layer_param_->parameter_param().shape());
}

template <typename Dtype>
void ParameterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    top[0]->ShareData(*(this->blobs_[0]));
    top[0]->ShareDiff(*(this->blobs_[0]));
}

INSTANTIATE_CLASS(ParameterLayer);
REGISTER_LAYER_CLASS(Parameter);

}  // namespace caffe
