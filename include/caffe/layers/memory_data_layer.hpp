#ifndef CAFFE_MEMORY_DATA_LAYER_HPP_
#define CAFFE_MEMORY_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from memory.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class CAFFE_EXPORT MemoryDataLayer : public BaseDataLayer<Dtype> {
 public:
  explicit MemoryDataLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param), has_new_data_(false) {}

  void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                      const vector<Blob<Dtype>*>& top) override;

  [[nodiscard]] const char* type() const override { return "MemoryData"; }
  [[nodiscard]] int ExactNumBottomBlobs() const override { return 0; }
  [[nodiscard]] int ExactNumTopBlobs() const override { return 2; }

  virtual void AddDatumVector(const vector<Datum>& datum_vector);
#ifdef USE_OPENCV
  virtual void AddMatVector(const vector<cv::Mat>& mat_vector,
      const vector<int>& labels);
#endif  // USE_OPENCV

  // Reset should accept const pointers, but can't, because the memory
  //  will be given to Blob, which is mutable
  void Reset(Dtype* data, Dtype* label, int n);
  void set_batch_size(int new_size);

  [[nodiscard]] int batch_size() const { return batch_size_; }
  [[nodiscard]] int channels() const { return channels_; }
  [[nodiscard]] int height() const { return height_; }
  [[nodiscard]] int width() const { return width_; }

 protected:
  void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                   const vector<Blob<Dtype>*>& top) override;

  int batch_size_, channels_, height_, width_, size_;
  Dtype* data_;
  Dtype* labels_;
  int n_;
  size_t pos_;
  Blob<Dtype> added_data_;
  Blob<Dtype> added_label_;
  bool has_new_data_;
};

}  // namespace caffe

#endif  // CAFFE_MEMORY_DATA_LAYER_HPP_
