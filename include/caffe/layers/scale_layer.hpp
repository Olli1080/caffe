#ifndef CAFFE_SCALE_LAYER_HPP_
#define CAFFE_SCALE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/bias_layer.hpp"

namespace caffe {

/**
 * @brief Computes the elementwise product of two input Blobs, with the shape of
 *        the latter Blob "broadcast" to match the shape of the former.
 *        Equivalent to tiling the latter Blob, then computing the elementwise
 *        product. Note: for efficiency and convenience, this layer can
 *        additionally perform a "broadcast" sum too when `bias_term: true`
 *        is set.
 *
 * The latter, scale input may be omitted, in which case it's learned as
 * parameter of the layer (as is the bias, if it is included).
 */
template <typename Dtype>
class ScaleLayer: public Layer<Dtype> {
 public:
  explicit ScaleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                  const vector<Blob<Dtype>*>& top) override;
  void Reshape(const vector<Blob<Dtype>*>& bottom,
               const vector<Blob<Dtype>*>& top) override;

  [[nodiscard]] const char* type() const override { return "Scale"; }
  // Scale
  [[nodiscard]] int MinBottomBlobs() const override { return 1; }
  [[nodiscard]] int MaxBottomBlobs() const override { return 2; }
  [[nodiscard]] int ExactNumTopBlobs() const override { return 1; }

 protected:
  /**
   * In the below shape specifications, @f$ i @f$ denotes the value of the
   * `axis` field given by `this->layer_param_.scale_param().axis()`, after
   * canonicalization (i.e., conversion from negative to positive index,
   * if applicable).
   *
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (d_0 \times ... \times
   *           d_i \times ... \times d_j \times ... \times d_n) @f$
   *      the first factor @f$ x @f$
   *   -# @f$ (d_i \times ... \times d_j) @f$
   *      the second factor @f$ y @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (d_0 \times ... \times
   *           d_i \times ... \times d_j \times ... \times d_n) @f$
   *      the product @f$ z = x y @f$ computed after "broadcasting" y.
   *      Equivalent to tiling @f$ y @f$ to have the same shape as @f$ x @f$,
   *      then computing the elementwise product.
   */
  void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                   const vector<Blob<Dtype>*>& top) override;
  void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                   const vector<Blob<Dtype>*>& top) override;
  void Backward_cpu(const vector<Blob<Dtype>*>& top,
                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) override;
  void Backward_gpu(const vector<Blob<Dtype>*>& top,
                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) override;

  shared_ptr<Layer<Dtype> > bias_layer_;
  vector<Blob<Dtype>*> bias_bottom_vec_;
  vector<bool> bias_propagate_down_;
  int bias_param_id_;

  Blob<Dtype> sum_multiplier_;
  Blob<Dtype> sum_result_;
  Blob<Dtype> temp_;
  int axis_;
  int outer_dim_, scale_dim_, inner_dim_;
};


}  // namespace caffe

#endif  // CAFFE_SCALE_LAYER_HPP_
