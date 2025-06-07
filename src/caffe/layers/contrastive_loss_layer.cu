#include "caffe/layers/contrastive_loss_layer.hpp"

//#include <algorithm>
//#include <vector>

namespace caffe {

template <typename Dtype>
__global__ void CLLBackward(const int count, const int channels,
    const Dtype margin, const bool legacy_version, const Dtype alpha,
    const Dtype* y, const Dtype* diff, const Dtype* dist_sq,
    Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    int n = i / channels;  // the num index, to access y and dist_sq
    if (static_cast<int>(y[n])) {  // similar pairs
      bottom_diff[i] = alpha * diff[i];
    } else {  // dissimilar pairs
      Dtype mdist(0.0);
      Dtype beta(0.0);
      if (legacy_version) {
        mdist = (margin - dist_sq[n]);
        beta = -alpha;
      } else {
        Dtype dist = sqrt(dist_sq[n]);
        mdist = (margin - dist);
        beta = -alpha * mdist / (dist + Dtype(1e-4)) * diff[i];
      }
      if (mdist > 0.0) {
        bottom_diff[i] = beta;
      } else {
        bottom_diff[i] = 0;
      }
    }
  }
}

template<typename Dtype>
void ContrastiveLossLayer<Dtype>::backward_kernel(int count, int channels, Dtype margin, bool legacy_version, Dtype alpha, const vector<Blob<Dtype>*>& bottom, int i)
{
	CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, channels, margin, legacy_version, alpha,
          bottom[2]->gpu_data(),  // pair similarity 0 or 1
          diff_.gpu_data(),  // the cached eltwise difference between a and b
          dist_sq_.gpu_data(),  // the cached square distance between a and b
          bottom[i]->mutable_gpu_diff());
}
template void ContrastiveLossLayer<float>::backward_kernel(int, int, float, bool, float, const vector<Blob<float>*>&, int);
template void ContrastiveLossLayer<double>::backward_kernel(int, int, double, bool, double, const vector<Blob<double>*>&, int);
//INSTANTIATE_LAYER_GPU_FUNCS(ContrastiveLossLayer);

}  // namespace caffe
