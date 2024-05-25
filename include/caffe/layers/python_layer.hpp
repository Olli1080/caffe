#ifndef CAFFE_PYTHON_LAYER_HPP_
#define CAFFE_PYTHON_LAYER_HPP_

#include <boost/python.hpp>
#include <vector>

#include "caffe/layer.hpp"

namespace bp = boost::python;

namespace caffe {

template <typename Dtype>
class PythonLayer : public Layer<Dtype> {
 public:
  PythonLayer(PyObject* self, const LayerParameter& param)
      : Layer<Dtype>(param), self_(bp::handle<>(bp::borrowed(self))) { }

  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                  const vector<Blob<Dtype>*>& top) override
  {
    // Disallow PythonLayer in MultiGPU training stage, due to GIL issues
    // Details: https://github.com/BVLC/caffe/issues/2936
    if (this->phase_ == TRAIN && Caffe::solver_count() > 1
        && !Caffe::multiprocess()) {
      LOG(FATAL) << "PythonLayer does not support CLI Multi-GPU, use train.py";
    }
    self_.attr("param_str") = bp::str(
        this->layer_param_.python_param().param_str());
    self_.attr("phase") = static_cast<int>(this->phase_);
    self_.attr("setup")(bottom, top);
  }

  void Reshape(const vector<Blob<Dtype>*>& bottom,
               const vector<Blob<Dtype>*>& top) override
  {
    self_.attr("reshape")(bottom, top);
  }

  const char* type() const override { return "Python"; }

 protected:
  void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                   const vector<Blob<Dtype>*>& top) override
  {
    self_.attr("forward")(bottom, top);
  }

  void Backward_cpu(const vector<Blob<Dtype>*>& top,
                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) override
  {
    self_.attr("backward")(top, propagate_down, bottom);
  }

 private:
  bp::object self_;
};

}  // namespace caffe

#endif
