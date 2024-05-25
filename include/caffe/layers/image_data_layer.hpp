#ifndef CAFFE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}

  ~ImageDataLayer() override;
  void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                      const vector<Blob<Dtype>*>& top) override;

  [[nodiscard]] const char* type() const override { return "ImageData"; }
  [[nodiscard]] int ExactNumBottomBlobs() const override { return 0; }
  [[nodiscard]] int ExactNumTopBlobs() const override { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  void load_batch(Batch<Dtype>* batch) override;

  vector<std::pair<std::string, int> > lines_;
  int lines_id_;
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
