#ifndef CAFFE_SPP_LAYER_HPP_
#define CAFFE_SPP_LAYER_HPP_

#include <vector>

#include "concat_layer.hpp"
#include "flatten_layer.hpp"
#include "pooling_layer.hpp"
#include "split_layer.hpp"

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"


namespace caffe {
	class SPPParameter;

	/**
 * @brief Does spatial pyramid pooling on the input image
 *        by taking the max, average, etc. within regions
 *        so that the result vector of different sized
 *        images are of the same size.
 */
template <typename Dtype>
class SPPLayer : public Layer<Dtype> {
 public:
  explicit SPPLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  void LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
                  const std::vector<Blob<Dtype>*>& top) override;
  void Reshape(const std::vector<Blob<Dtype>*>& bottom,
               const std::vector<Blob<Dtype>*>& top) override;

  [[nodiscard]] const char* type() const override { return "SPP"; }
  [[nodiscard]] int ExactNumBottomBlobs() const override { return 1; }
  [[nodiscard]] int ExactNumTopBlobs() const override { return 1; }

 protected:
  void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
                   const std::vector<Blob<Dtype>*>& top) override;
  void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
                    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
  // calculates the kernel and stride dimensions for the pooling layer,
  // returns a correctly configured LayerParameter for a PoolingLayer
  virtual LayerParameter GetPoolingParam(int pyramid_level,
                                         int bottom_h, int bottom_w, SPPParameter spp_param);

  int pyramid_height_;
  int bottom_h_, bottom_w_;
  int num_;
  int channels_;
  int kernel_h_, kernel_w_;
  int pad_h_, pad_w_;
  bool reshaped_first_time_;

  /// the internal Split layer that feeds the pooling layers
  std::shared_ptr<SplitLayer<Dtype> > split_layer_;
  /// top vector holder used in call to the underlying SplitLayer::Forward
  std::vector<Blob<Dtype>*> split_top_vec_;
  /// bottom vector holder used in call to the underlying PoolingLayer::Forward
  std::vector<std::vector<Blob<Dtype>*>*> pooling_bottom_vecs_;
  /// the internal Pooling layers of different kernel sizes
  std::vector<std::shared_ptr<PoolingLayer<Dtype> > > pooling_layers_;
  /// top vector holders used in call to the underlying PoolingLayer::Forward
  std::vector<std::vector<Blob<Dtype>*>*> pooling_top_vecs_;
  /// pooling_outputs stores the outputs of the PoolingLayers
  std::vector<Blob<Dtype>*> pooling_outputs_;
  /// the internal Flatten layers that the Pooling layers feed into
  std::vector<FlattenLayer<Dtype>*> flatten_layers_;
  /// top vector holders used in call to the underlying FlattenLayer::Forward
  std::vector<std::vector<Blob<Dtype>*>*> flatten_top_vecs_;
  /// flatten_outputs stores the outputs of the FlattenLayers
  std::vector<Blob<Dtype>*> flatten_outputs_;
  /// bottom vector holder used in call to the underlying ConcatLayer::Forward
  std::vector<Blob<Dtype>*> concat_bottom_vec_;
  /// the internal Concat layers that the Flatten layers feed into
  std::shared_ptr<ConcatLayer<Dtype> > concat_layer_;
};

}  // namespace caffe

#endif  // CAFFE_SPP_LAYER_HPP_
