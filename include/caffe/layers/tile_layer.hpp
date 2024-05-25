#ifndef CAFFE_TILE_LAYER_HPP_
#define CAFFE_TILE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Copy a Blob along specified dimensions.
 */
template <typename Dtype>
class TileLayer : public Layer<Dtype> {
 public:
  explicit TileLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  void Reshape(const vector<Blob<Dtype>*>& bottom,
               const vector<Blob<Dtype>*>& top) override;

  [[nodiscard]] const char* type() const override { return "Tile"; }
  [[nodiscard]] int ExactNumBottomBlobs() const override { return 1; }
  [[nodiscard]] int ExactNumTopBlobs() const override { return 1; }

 protected:
  void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                   const vector<Blob<Dtype>*>& top) override;
  void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                   const vector<Blob<Dtype>*>& top) override;

  void Backward_cpu(const vector<Blob<Dtype>*>& top,
                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) override;
  void Backward_gpu(const vector<Blob<Dtype>*>& top,
                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) override;

  unsigned int axis_, tiles_, outer_dim_, inner_dim_;
};

}  // namespace caffe

#endif  // CAFFE_TILE_LAYER_HPP_
