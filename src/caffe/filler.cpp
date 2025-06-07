#include "caffe/filler.hpp"

#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
Filler<Dtype>::Filler(const FillerParameter& param)
    : filler_param_(std::make_unique<FillerParameter>(param)) {}

template <typename Dtype>
void ConstantFiller<Dtype>::Fill(Blob<Dtype>* blob)
{
    Dtype* data = blob->mutable_cpu_data();
    const int count = blob->count();
    const Dtype value = this->filler_param_->value();
    CHECK(count);
    for (int i = 0; i < count; ++i) {
        data[i] = value;
    }
    CHECK_EQ(this->filler_param_->sparse(), -1)
        << "Sparsity not supported by this Filler.";
}

template <typename Dtype>
void UniformFiller<Dtype>::Fill(Blob<Dtype>* blob)
{
    CHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), Dtype(this->filler_param_->min()),
        Dtype(this->filler_param_->max()), blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_->sparse(), -1)
        << "Sparsity not supported by this Filler.";
}

template <typename Dtype>
void GaussianFiller<Dtype>::Fill(Blob<Dtype>* blob)
{
    Dtype* data = blob->mutable_cpu_data();
    CHECK(blob->count());
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_->mean()),
        Dtype(this->filler_param_->std()), blob->mutable_cpu_data());
    int sparse = this->filler_param_->sparse();
    CHECK_GE(sparse, -1);
    if (sparse >= 0) {
        // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
        // These have num == channels == 1; width is number of inputs; height is
        // number of outputs.  The 'sparse' variable specifies the mean number
        // of non-zero input weights for a given output.
        CHECK_GE(blob->num_axes(), 1);
        const int num_outputs = blob->shape(0);
        Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
        rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(int)));
        auto mask = static_cast<int*>(rand_vec_->mutable_cpu_data());
        caffe_rng_bernoulli(blob->count(), non_zero_probability, mask);
        for (int i = 0; i < blob->count(); ++i) {
            data[i] *= mask[i];
        }
    }
}

template <typename Dtype>
void PositiveUnitballFiller<Dtype>::Fill(Blob<Dtype>* blob)
{
    Dtype* data = blob->mutable_cpu_data();
    DCHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), 0, 1, blob->mutable_cpu_data());
    // We expect the filler to not be called very frequently, so we will
    // just use a simple implementation
    int dim = blob->count() / blob->shape(0);
    CHECK(dim);
    for (int i = 0; i < blob->shape(0); ++i) {
        Dtype sum = 0;
        for (int j = 0; j < dim; ++j) {
            sum += data[i * dim + j];
        }
        for (int j = 0; j < dim; ++j) {
            data[i * dim + j] /= sum;
        }
    }
    CHECK_EQ(this->filler_param_->sparse(), -1)
        << "Sparsity not supported by this Filler.";
}

template <typename Dtype>
void XavierFiller<Dtype>::Fill(Blob<Dtype>* blob)
{
    CHECK(blob->count());
    int fan_in = blob->count() / blob->shape(0);
    // Compatibility with ND blobs
    int fan_out = blob->num_axes() > 1 ?
        blob->count() / blob->shape(1) :
        blob->count();
    Dtype n = static_cast<Dtype>(fan_in);  // default to fan_in
    if (this->filler_param_->variance_norm() ==
        FillerParameter_VarianceNorm_AVERAGE) {
        n = (fan_in + fan_out) / Dtype(2);
    }
    else if (this->filler_param_->variance_norm() ==
        FillerParameter_VarianceNorm_FAN_OUT) {
        n = static_cast<Dtype>(fan_out);
    }
    Dtype scale = sqrt(Dtype(3) / n);
    caffe_rng_uniform<Dtype>(blob->count(), -scale, scale,
        blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_->sparse(), -1)
        << "Sparsity not supported by this Filler.";
}

template <typename Dtype>
void MSRAFiller<Dtype>::Fill(Blob<Dtype>* blob)
{
    CHECK(blob->count());
    int fan_in = blob->count() / blob->shape(0);
    // Compatibility with ND blobs
    int fan_out = blob->num_axes() > 1 ?
        blob->count() / blob->shape(1) :
        blob->count();
    Dtype n = static_cast<Dtype>(fan_in);  // default to fan_in
    if (this->filler_param_->variance_norm() ==
        FillerParameter_VarianceNorm_AVERAGE) {
        n = (fan_in + fan_out) / Dtype(2);
    }
    else if (this->filler_param_->variance_norm() ==
        FillerParameter_VarianceNorm_FAN_OUT) {
        n = static_cast<Dtype>(fan_out);
    }
    Dtype std = sqrt(Dtype(2) / n);
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(0), std,
        blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_->sparse(), -1)
        << "Sparsity not supported by this Filler.";
}

template <typename Dtype>
void BilinearFiller<Dtype>::Fill(Blob<Dtype>* blob)
{
    CHECK_EQ(blob->num_axes(), 4) << "Blob must be 4 dim.";
    CHECK_EQ(blob->width(), blob->height()) << "Filter must be square";
    Dtype* data = blob->mutable_cpu_data();
    int f = static_cast<int>(ceil(blob->width() / 2.));
    Dtype c = static_cast<Dtype>((blob->width() - 1) / (2. * f));
    for (int i = 0; i < blob->count(); ++i) {
        Dtype x = static_cast<Dtype>(i % blob->width());
        Dtype y = static_cast<Dtype>((i / blob->width()) % blob->height());
        data[i] = (1 - fabs(x / f - c)) * (1 - fabs(y / f - c));
    }
    CHECK_EQ(this->filler_param_->sparse(), -1)
        << "Sparsity not supported by this Filler.";
}

template <typename Dtype>
Filler<Dtype>* GetFiller(const FillerParameter& param) {
  const std::string& type = param.type();
  if (type == "constant") {
    return new ConstantFiller<Dtype>(param);
  } else if (type == "gaussian") {
    return new GaussianFiller<Dtype>(param);
  } else if (type == "positive_unitball") {
    return new PositiveUnitballFiller<Dtype>(param);
  } else if (type == "uniform") {
    return new UniformFiller<Dtype>(param);
  } else if (type == "xavier") {
    return new XavierFiller<Dtype>(param);
  } else if (type == "msra") {
    return new MSRAFiller<Dtype>(param);
  } else if (type == "bilinear") {
    return new BilinearFiller<Dtype>(param);
  } else {
    CHECK(false) << "Unknown filler name: " << param.type();
  }
  return nullptr;
}

template Filler<float>* GetFiller<float>(const FillerParameter&);
template Filler<double>* GetFiller<double>(const FillerParameter&);
}
