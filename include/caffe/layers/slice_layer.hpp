#ifndef CAFFE_SLICE_LAYER_HPP_
#define CAFFE_SLICE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"


namespace caffe {

/**
 * @brief Takes a Blob and slices it along either the num or channel dimension,
 *        outputting multiple sliced Blob results.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SliceLayer : public Layer<Dtype> {
 public:
  explicit SliceLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  void LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
                  const std::vector<Blob<Dtype>*>& top) override;
  void Reshape(const std::vector<Blob<Dtype>*>& bottom,
               const std::vector<Blob<Dtype>*>& top) override;

  [[nodiscard]] const char* type() const override { return "Slice"; }
  [[nodiscard]] int ExactNumBottomBlobs() const override { return 1; }
  [[nodiscard]] int MinTopBlobs() const override { return 1; }

 protected:
  void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
                   const std::vector<Blob<Dtype>*>& top) override;
  void Forward_gpu(const std::vector<Blob<Dtype>*>& bottom,
                   const std::vector<Blob<Dtype>*>& top) override;
  void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
                    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
  void Backward_gpu(const std::vector<Blob<Dtype>*>& top,
                    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;

  int count_;
  int num_slices_;
  int slice_size_;
  int slice_axis_;
  std::vector<int> slice_point_;
};

}  // namespace caffe

#endif  // CAFFE_SLICE_LAYER_HPP_
