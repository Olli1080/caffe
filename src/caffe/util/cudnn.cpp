#ifdef USE_CUDNN
#include "caffe/util/cudnn.hpp"

#include "caffe/proto/caffe.pb.h"

namespace caffe {
namespace cudnn {

void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc, PoolingParameter_PoolMethod const& poolmethod,
	cudnnPoolingMode_t* mode, int h, int w, int pad_h, int pad_w, int stride_h, int stride_w)
{
  switch (poolmethod) {
  case PoolingParameter_PoolMethod_MAX:
    *mode = CUDNN_POOLING_MAX;
    break;
  case PoolingParameter_PoolMethod_AVE:
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc, *mode,
        CUDNN_PROPAGATE_NAN, h, w, pad_h, pad_w, stride_h, stride_w));
#else
  CUDNN_CHECK(cudnnSetPooling2dDescriptor_v4(*pool_desc, *mode,
        CUDNN_PROPAGATE_NAN, h, w, pad_h, pad_w, stride_h, stride_w));
#endif
}

float dataType<float>::oneval = 1.0;
float dataType<float>::zeroval = 0.0;
const void* dataType<float>::one =
    static_cast<void *>(&dataType<float>::oneval);
const void* dataType<float>::zero =
    static_cast<void *>(&dataType<float>::zeroval);

double dataType<double>::oneval = 1.0;
double dataType<double>::zeroval = 0.0;
const void* dataType<double>::one =
    static_cast<void *>(&dataType<double>::oneval);
const void* dataType<double>::zero =
    static_cast<void *>(&dataType<double>::zeroval);

}  // namespace cudnn
}  // namespace caffe
#endif
