#include "caffe/layers/neuron_layer.hpp"

#include <vector>

#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
void NeuronLayer<Dtype>::Reshape(const std::vector<Blob<Dtype>*>& bottom,
      const std::vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

INSTANTIATE_CLASS(NeuronLayer);

}  // namespace caffe
