#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include "caffe/common.hpp"

namespace caffe {

/**
 * Virtual class encapsulate std::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
 */
class InternalThread {
 public:
  InternalThread() = default;
  virtual ~InternalThread();

  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  void StartInternalThread();

  /** Will not return until the internal thread has exited. */
  void StopInternalThread();

  [[nodiscard]] bool is_started() const;

 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  virtual void InternalThreadEntry() {}

  /* Should be tested when running loops to exit when requested. */
  [[nodiscard]] bool must_stop() const;

 private:
  void entry(int device, Caffe::Brew mode, unsigned int rand_seed,
      int solver_count, int solver_rank, bool multiprocess);

  shared_ptr<std::thread> thread_;
  bool stop_requested = false;
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
