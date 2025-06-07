#include "caffe/layers/eltwise_layer.hpp"

#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxForward(const int nthreads, const Dtype* bottom_data_a,
    const Dtype* bottom_data_b, const int blob_idx, Dtype* top_data,
    int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    if (bottom_data_a[index] > bottom_data_b[index]) {
      // only update for very first bottom_data blob (blob_idx == 0)
      if (blob_idx == 0) {
        maxval = bottom_data_a[index];
        top_data[index] = maxval;
        maxidx = blob_idx;
        mask[index] = maxidx;
      }
    } else {
      maxval = bottom_data_b[index];
      top_data[index] = maxval;
      maxidx = blob_idx + 1;
      mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void MaxBackward(const int nthreads, const Dtype* top_diff,
    const int blob_idx, const int* mask, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype gradient = 0;
    if (mask[index] == blob_idx) {
      gradient += top_diff[index];
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::forward_kernel(int count, const Dtype* bottom_data_a, const Dtype* bottom_data_b, int blob_idx, Dtype* top_data, int* mask)
{
    MaxForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data_a, bottom_data_b, blob_idx, top_data, mask);
}

template <typename Dtype>
void EltwiseLayer<Dtype>::backward_kernel(int count, const Dtype* top_diff, const int blob_idx, const int* mask, Dtype* bottom_diff)
{
	MaxBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, blob_idx, mask, bottom_diff);
}
template void EltwiseLayer<float>::forward_kernel(int, const float*, const float*, int, float*, int*);
template void EltwiseLayer<double>::forward_kernel(int, const double*, const double*, int, double*, int*);

template void EltwiseLayer<float>::backward_kernel(int, const float*, const int, const int*, float*);
template void EltwiseLayer<double>::backward_kernel(int, const double*, const int, const int*, double*);
//INSTANTIATE_LAYER_GPU_FUNCS(EltwiseLayer);

}  // namespace caffe
