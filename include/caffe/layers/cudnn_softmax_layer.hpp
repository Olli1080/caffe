#ifndef CAFFE_CUDNN_SOFTMAX_LAYER_HPP_
#define CAFFE_CUDNN_SOFTMAX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"


#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

#ifdef USE_CUDNN
/**
 * @brief cuDNN implementation of SoftmaxLayer.
 *        Fallback to SoftmaxLayer for CPU mode.
 */
template <typename Dtype>
class CAFFE_EXPORT CuDNNSoftmaxLayer : public SoftmaxLayer<Dtype> {
 public:
  explicit CuDNNSoftmaxLayer(const LayerParameter& param)
      : SoftmaxLayer<Dtype>(param), handles_setup_(false) {}

  void LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
                  const std::vector<Blob<Dtype>*>& top) override;
  void Reshape(const std::vector<Blob<Dtype>*>& bottom,
               const std::vector<Blob<Dtype>*>& top) override;
  ~CuDNNSoftmaxLayer() override;

 protected:
  void Forward_gpu(const std::vector<Blob<Dtype>*>& bottom,
                   const std::vector<Blob<Dtype>*>& top) override;
  void Backward_gpu(const std::vector<Blob<Dtype>*>& top,
                    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;

  bool handles_setup_;
  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t bottom_desc_;
  cudnnTensorDescriptor_t top_desc_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_SOFTMAX_LAYER_HPP_
