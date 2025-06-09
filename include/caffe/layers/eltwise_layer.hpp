#ifndef CAFFE_ELTWISE_LAYER_HPP_
#define CAFFE_ELTWISE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"


namespace caffe {
	enum EltwiseParameter_EltwiseOp : int;

	/**
 * @brief Compute elementwise operations, such as product and sum,
 *        along multiple input Blobs.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class EltwiseLayer : public Layer<Dtype> {
 public:
  explicit EltwiseLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  void LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
                  const std::vector<Blob<Dtype>*>& top) override;
  void Reshape(const std::vector<Blob<Dtype>*>& bottom,
               const std::vector<Blob<Dtype>*>& top) override;

  [[nodiscard]] const char* type() const override { return "Eltwise"; }
  [[nodiscard]] int MinBottomBlobs() const override { return 2; }
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

  EltwiseParameter_EltwiseOp op_;
  std::vector<Dtype> coeffs_;
  Blob<int> max_idx_;

  bool stable_prod_grad_;

private:

#ifndef CPU_ONLY
    void forward_kernel(int count, const Dtype* bottom_data_a, const Dtype* bottom_data_b, int blob_idx, Dtype* top_data, int* mask);
    void backward_kernel(int count, const Dtype* top_diff, const int blob_idx, const int* mask, Dtype* bottom_diff);
#endif
};

}  // namespace caffe

#endif  // CAFFE_ELTWISE_LAYER_HPP_
