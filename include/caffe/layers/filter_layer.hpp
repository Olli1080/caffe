#ifndef CAFFE_FILTER_LAYER_HPP_
#define CAFFE_FILTER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"


namespace caffe {

/**
 * @brief Takes two+ Blobs, interprets last Blob as a selector and
 *  filter remaining Blobs accordingly with selector data (0 means that
 * the corresponding item has to be filtered, non-zero means that corresponding
 * item needs to stay).
 */
template <typename Dtype>
class FilterLayer : public Layer<Dtype> {
 public:
  explicit FilterLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  void LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
                  const std::vector<Blob<Dtype>*>& top) override;
  void Reshape(const std::vector<Blob<Dtype>*>& bottom,
               const std::vector<Blob<Dtype>*>& top) override;

  [[nodiscard]] const char* type() const override { return "Filter"; }
  [[nodiscard]] int MinBottomBlobs() const override { return 2; }
  [[nodiscard]] int MinTopBlobs() const override { return 1; }

 protected:
  /**
   * @param bottom input Blob vector (length 2+)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs to be filtered @f$ x_1 @f$
   *   -# ...
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs to be filtered @f$ x_K @f$
   *   -# @f$ (N \times 1 \times 1 \times 1) @f$
   *      the selector blob
   * @param top output Blob vector (length 1+)
   *   -# @f$ (S \times C \times H \times W) @f$ ()
   *        the filtered output @f$ x_1 @f$
   *        where S is the number of items
   *        that haven't been filtered
   *      @f$ (S \times C \times H \times W) @f$
   *        the filtered output @f$ x_K @f$
   *        where S is the number of items
   *        that haven't been filtered
   */
  void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
                   const std::vector<Blob<Dtype>*>& top) override;
  void Forward_gpu(const std::vector<Blob<Dtype>*>& bottom,
                   const std::vector<Blob<Dtype>*>& top) override;

  /**
   * @brief Computes the error gradient w.r.t. the forwarded inputs.
   *
   * @param top output Blob vector (length 1+), providing the error gradient with
   *        respect to the outputs
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 2+), into which the top error
   *        gradient is copied
   */
  void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
                    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
  void Backward_gpu(const std::vector<Blob<Dtype>*>& top,
                    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;

  bool first_reshape_;
  std::vector<int> indices_to_forward_;
};

}  // namespace caffe

#endif  // CAFFE_FILTER_LAYER_HPP_
