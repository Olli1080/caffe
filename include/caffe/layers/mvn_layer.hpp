#ifndef CAFFE_MVN_LAYER_HPP_
#define CAFFE_MVN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"


namespace caffe {

/**
 * @brief Normalizes the input to have 0-mean and/or unit (1) variance.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class MVNLayer : public Layer<Dtype> {
 public:
  explicit MVNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  void Reshape(const std::vector<Blob<Dtype>*>& bottom,
               const std::vector<Blob<Dtype>*>& top) override;

  [[nodiscard]] const char* type() const override { return "MVN"; }
  [[nodiscard]] int ExactNumBottomBlobs() const override { return 1; }
  [[nodiscard]] int ExactNumTopBlobs() const override { return 1; }

 protected:
  void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
                   const std::vector<Blob<Dtype>*>& top) override;
  void Forward_gpu(const std::vector<Blob<Dtype>*>& bottom,
                   const std::vector<Blob<Dtype>*>& top) override;
  void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
                    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
  void Backward_gpu(const std::vector<Blob<Dtype>*>& top,
                    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;

  Blob<Dtype> mean_, variance_, temp_;

  /// sum_multiplier is used to carry out sum using BLAS
  Blob<Dtype> sum_multiplier_;
  Dtype eps_;
};

}  // namespace caffe

#endif  // CAFFE_MVN_LAYER_HPP_
