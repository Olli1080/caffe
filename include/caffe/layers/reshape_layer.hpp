#ifndef CAFFE_XXX_LAYER_HPP_
#define CAFFE_XXX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"


namespace caffe {

/*
 * @brief Reshapes the input Blob into an arbitrary-sized output Blob.
 *
 * Note: similarly to FlattenLayer, this layer does not change the input values
 * (see FlattenLayer, Blob::ShareData and Blob::ShareDiff).
 */
template <typename Dtype>
class ReshapeLayer : public Layer<Dtype> {
 public:
  explicit ReshapeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  void LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
                  const std::vector<Blob<Dtype>*>& top) override;
  void Reshape(const std::vector<Blob<Dtype>*>& bottom,
               const std::vector<Blob<Dtype>*>& top) override;

  [[nodiscard]] const char* type() const override { return "Reshape"; }
  [[nodiscard]] int ExactNumBottomBlobs() const override { return 1; }
  [[nodiscard]] int ExactNumTopBlobs() const override { return 1; }

 protected:
  void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
                   const std::vector<Blob<Dtype>*>& top) override {}

  void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
                    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override {}

  void Forward_gpu(const std::vector<Blob<Dtype>*>& bottom,
                   const std::vector<Blob<Dtype>*>& top) override {}

  void Backward_gpu(const std::vector<Blob<Dtype>*>& top,
                    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override {}

  /// @brief vector of axes indices whose dimensions we'll copy from the bottom
  std::vector<int> copy_axes_;
  /// @brief the index of the axis whose dimension we infer, or -1 if none
  int inferred_axis_;
  /// @brief the product of the "constant" output dimensions
  int constant_count_;
};

}  // namespace caffe

#endif  // CAFFE_XXX_LAYER_HPP_
