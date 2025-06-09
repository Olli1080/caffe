#ifndef CAFFE_POOLING_LAYER_HPP_
#define CAFFE_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"


namespace caffe {
	enum PoolingParameter_RoundMode : int;

	/**
 * @brief Pools the input image by taking the max, average, etc. within regions.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class CAFFE_EXPORT PoolingLayer : public Layer<Dtype> {
 public:
  explicit PoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  void LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
                  const std::vector<Blob<Dtype>*>& top) override;
  void Reshape(const std::vector<Blob<Dtype>*>& bottom,
               const std::vector<Blob<Dtype>*>& top) override;

  [[nodiscard]] const char* type() const override { return "Pooling"; }
  [[nodiscard]] int ExactNumBottomBlobs() const override { return 1; }
  [[nodiscard]] int MinTopBlobs() const override { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  [[nodiscard]] int MaxTopBlobs() const override;

 protected:
  void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
                   const std::vector<Blob<Dtype>*>& top) override;
  void Forward_gpu(const std::vector<Blob<Dtype>*>& bottom,
                   const std::vector<Blob<Dtype>*>& top) override;
  void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
                    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
  void Backward_gpu(const std::vector<Blob<Dtype>*>& top,
                    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int pad_h_, pad_w_;
  int channels_;
  int height_, width_;
  int pooled_height_, pooled_width_;
  bool global_pooling_;
  PoolingParameter_RoundMode round_mode_;
  Blob<Dtype> rand_idx_;
  Blob<int> max_idx_;

private:

#ifndef CPU_ONLY
    void MaxPoolForwardKernel(const int nthreads,
        const Dtype* const bottom_data, const int num,
        Dtype* const top_data, int* mask, Dtype* top_mask);

    void AvePoolForwardKernel(const int nthreads,
        const Dtype* const bottom_data, const int num,
        Dtype* const top_data);

    void StoPoolForwardTrainKernel(const int nthreads,
        const Dtype* const bottom_data,
        const int num, Dtype* const rand_idx, Dtype* const top_data);

    void StoPoolForwardTestKernel(const int nthreads,
        const Dtype* const bottom_data,
        const int num, Dtype* const top_data);



    void MaxPoolBackwardKernel(const int nthreads, const Dtype* const top_diff,
        const int* const mask, const Dtype* const top_mask, const int num, Dtype* const bottom_diff);

    void AvePoolBackwardKernel(const int nthreads, const Dtype* const top_diff,
        const int num, Dtype* const bottom_diff);

    void StoPoolBackwardKernel(const int nthreads,
        const Dtype* const rand_idx, const Dtype* const top_diff,
        const int num, Dtype* const bottom_diff);
#endif
};

}  // namespace caffe

#endif  // CAFFE_POOLING_LAYER_HPP_
