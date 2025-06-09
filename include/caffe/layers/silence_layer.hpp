#ifndef CAFFE_SILENCE_LAYER_HPP_
#define CAFFE_SILENCE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"


namespace caffe {

/**
 * @brief Ignores bottom blobs while producing no top blobs. (This is useful
 *        to suppress outputs during testing.)
 */
template <typename Dtype>
class SilenceLayer : public Layer<Dtype> {
 public:
  explicit SilenceLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  void Reshape(const std::vector<Blob<Dtype>*>& bottom,
               const std::vector<Blob<Dtype>*>& top) override {}

  [[nodiscard]] const char* type() const override { return "Silence"; }
  [[nodiscard]] int MinBottomBlobs() const override { return 1; }
  [[nodiscard]] int ExactNumTopBlobs() const override { return 0; }

 protected:
  void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
                   const std::vector<Blob<Dtype>*>& top) override {}
  // We can't define Forward_gpu here, since STUB_GPU will provide
  // its own definition for CPU_ONLY mode.
  void Forward_gpu(const std::vector<Blob<Dtype>*>& bottom,
                   const std::vector<Blob<Dtype>*>& top) override;
  void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
                    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
  void Backward_gpu(const std::vector<Blob<Dtype>*>& top,
                    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
};

}  // namespace caffe

#endif  // CAFFE_SILENCE_LAYER_HPP_
