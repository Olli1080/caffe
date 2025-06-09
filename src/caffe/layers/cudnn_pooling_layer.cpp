#ifdef USE_CUDNN
#include "caffe/layers/cudnn_pooling_layer.hpp"

#include <vector>

#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
    const std::vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::LayerSetUp(bottom, top);
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  cudnn::createPoolingDesc(&pooling_desc_,
      this->layer_param_->pooling_param().pool(), &mode_,
      this->kernel_h_, this->kernel_w_, this->pad_h_, this->pad_w_,
      this->stride_h_, this->stride_w_);
  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Reshape(const std::vector<Blob<Dtype>*>& bottom,
    const std::vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::Reshape(bottom, top);
  cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  cudnn::setTensor4dDesc<Dtype>(&top_desc_, bottom[0]->num(),
      this->channels_, this->pooled_height_, this->pooled_width_);
}

template <typename Dtype>
CuDNNPoolingLayer<Dtype>::~CuDNNPoolingLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyPoolingDescriptor(pooling_desc_);
  cudnnDestroy(handle_);
}
#ifdef CPU_ONLY
STUB_GPU(CuDNNPoolingLayer);
#else
template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Forward_gpu(const std::vector<Blob<Dtype>*>& bottom,
    const std::vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    CUDNN_CHECK(cudnnPoolingForward(handle_, pooling_desc_,
        cudnn::dataType<Dtype>::one,
        bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        top_desc_, top_data));
}

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Backward_gpu(const std::vector<Blob<Dtype>*>& top,
    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
        return;
    }
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    CUDNN_CHECK(cudnnPoolingBackward(handle_, pooling_desc_,
        cudnn::dataType<Dtype>::one,
        top_desc_, top_data, top_desc_, top_diff,
        bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        bottom_desc_, bottom_diff));
}
#endif

INSTANTIATE_CLASS(CuDNNPoolingLayer);

}   // namespace caffe
#endif
