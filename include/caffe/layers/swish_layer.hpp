#ifndef CAFFE_SWISH_LAYER_HPP_
#define CAFFE_SWISH_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"


#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

/**
 * @brief Swish non-linearity @f$ y = x \sigma (\beta x) @f$.
 *        A novel activation function that tends to work better than ReLU [1].
 *
 * [1] Prajit Ramachandran, Barret Zoph, Quoc V. Le. "Searching for
 *     Activation Functions". arXiv preprint arXiv:1710.05941v2 (2017).
 */
template <typename Dtype>
class SwishLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides SwishParameter swish_param,
   *     with SwishLayer options:
   *   - beta (\b optional, default 1).
   *     the value @f$ \beta @f$ in the @f$ y = x \sigma (\beta x) @f$.
   */
  explicit SwishLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param),
        sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
        sigmoid_input_(new Blob<Dtype>()),
        sigmoid_output_(new Blob<Dtype>()) {}

  void LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
                  const std::vector<Blob<Dtype>*>& top) override;
  void Reshape(const std::vector<Blob<Dtype>*>& bottom,
               const std::vector<Blob<Dtype>*>& top) override;

  inline const char* type() const override { return "Swish"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the computed outputs @f$
   *        y = x \sigma (\beta x)
   *      @f$.
   */
  void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
                   const std::vector<Blob<Dtype>*>& top) override;
  void Forward_gpu(const std::vector<Blob<Dtype>*>& bottom,
                   const std::vector<Blob<Dtype>*>& top) override;

  /**
   * @brief Computes the error gradient w.r.t. the sigmoid inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (N \times C \times H \times W) @f$
   *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
   *      with respect to computed outputs @f$ y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial x}
   *            = \frac{\partial E}{\partial y}(\beta y +
   *              \sigma (\beta x)(1 - \beta y))
   *      @f$ if propagate_down[0]
   */
  void Backward_cpu(const std::vector<Blob<Dtype>*>& top,
                    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;
  void Backward_gpu(const std::vector<Blob<Dtype>*>& top,
                    const std::vector<bool>& propagate_down, const std::vector<Blob<Dtype>*>& bottom) override;

  /// The internal SigmoidLayer
  std::shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  /// sigmoid_input_ stores the input of the SigmoidLayer.
  std::shared_ptr<Blob<Dtype> > sigmoid_input_;
  /// sigmoid_output_ stores the output of the SigmoidLayer.
  std::shared_ptr<Blob<Dtype> > sigmoid_output_;
  /// bottom vector holder to call the underlying SigmoidLayer::Forward
  std::vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  /// top vector holder to call the underlying SigmoidLayer::Forward
  std::vector<Blob<Dtype>*> sigmoid_top_vec_;

private:

#ifndef CPU_ONLY
    void backward_kernel(const int n, const Dtype* in_diff,
        const Dtype* out_data, const Dtype* sigmoid_output_data, Dtype* out_diff,
        const Dtype beta);
#endif
};

}  // namespace caffe

#endif  // CAFFE_SWISH_LAYER_HPP_
