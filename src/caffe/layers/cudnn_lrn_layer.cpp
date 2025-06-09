#ifdef USE_CUDNN
#include "caffe/layers/cudnn_lrn_layer.hpp"

#include <vector>

#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
void CuDNNLRNLayer<Dtype>::LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
    const std::vector<Blob<Dtype>*>& top) {
  LRNLayer<Dtype>::LayerSetUp(bottom, top);

  CUDNN_CHECK(cudnnCreate(&handle_));
  CUDNN_CHECK(cudnnCreateLRNDescriptor(&norm_desc_));
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);

  // create a LRN handle
  handles_setup_ = true;

  size_ = this->layer_param().lrn_param().local_size();
  alpha_ = this->layer_param().lrn_param().alpha();
  beta_ = this->layer_param().lrn_param().beta();
  k_ = this->layer_param().lrn_param().k();
}

template <typename Dtype>
void CuDNNLRNLayer<Dtype>::Reshape(const std::vector<Blob<Dtype>*>& bottom,
    const std::vector<Blob<Dtype>*>& top) {
  LRNLayer<Dtype>::Reshape(bottom, top);
  cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  cudnn::setTensor4dDesc<Dtype>(&top_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  CUDNN_CHECK(cudnnSetLRNDescriptor(norm_desc_, size_, alpha_, beta_, k_));
}

template <typename Dtype>
CuDNNLRNLayer<Dtype>::~CuDNNLRNLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);

  // destroy LRN handle
  cudnnDestroy(handle_);
}
#ifdef CPU_ONLY
STUB_GPU(CuDNNLRNLayer);
#else
template <typename Dtype>
void CuDNNLRNLayer<Dtype>::Forward_gpu(const std::vector<Blob<Dtype>*>& bottom,
    const std::vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    CUDNN_CHECK(cudnnLRNCrossChannelForward(
        handle_, norm_desc_, CUDNN_LRN_CROSS_CHANNEL_DIM1,
        cudnn::dataType<Dtype>::one,
        bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        top_desc_, top_data));
}

template <typename Dtype>
void CuDNNLRNLayer<Dtype>::Backward_gpu(const std::vector<Blob<Dtype>*>& top,
    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    CUDNN_CHECK(cudnnLRNCrossChannelBackward(
        handle_, norm_desc_, CUDNN_LRN_CROSS_CHANNEL_DIM1,
        cudnn::dataType<Dtype>::one,
        top_desc_, top_data,
        top_desc_, top_diff,
        bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        bottom_desc_, bottom_diff));
}
#endif

INSTANTIATE_CLASS(CuDNNLRNLayer);

}   // namespace caffe
#endif
