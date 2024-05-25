#ifndef CAFFE_HDF5_OUTPUT_LAYER_HPP_
#define CAFFE_HDF5_OUTPUT_LAYER_HPP_

#include "hdf5.h"

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

#define HDF5_DATA_DATASET_NAME "data"
#define HDF5_DATA_LABEL_NAME "label"

/**
 * @brief Write blobs to disk as HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class HDF5OutputLayer : public Layer<Dtype> {
 public:
  explicit HDF5OutputLayer(const LayerParameter& param)
      : Layer<Dtype>(param), file_opened_(false) {}

  ~HDF5OutputLayer() override;
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                  const vector<Blob<Dtype>*>& top) override;
  // Data layers have no bottoms, so reshaping is trivial.
  void Reshape(const vector<Blob<Dtype>*>& bottom,
               const vector<Blob<Dtype>*>& top) override {}

  [[nodiscard]] const char* type() const override { return "HDF5Output"; }
  // TODO: no limit on the number of blobs
  [[nodiscard]] int ExactNumBottomBlobs() const override { return 2; }
  [[nodiscard]] int ExactNumTopBlobs() const override { return 0; }

  [[nodiscard]] std::string file_name() const { return file_name_; }

 protected:
  void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                   const vector<Blob<Dtype>*>& top) override;
  void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                   const vector<Blob<Dtype>*>& top) override;
  void Backward_cpu(const vector<Blob<Dtype>*>& top,
                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) override;
  void Backward_gpu(const vector<Blob<Dtype>*>& top,
                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) override;
  virtual void SaveBlobs();

  bool file_opened_;
  std::string file_name_;
  hid_t file_id_;
  Blob<Dtype> data_blob_;
  Blob<Dtype> label_blob_;
};

}  // namespace caffe

#endif  // CAFFE_HDF5_OUTPUT_LAYER_HPP_
