#ifndef CAFFE_UTIL_BENCHMARK_H_
#define CAFFE_UTIL_BENCHMARK_H_

#include <chrono>

#include "caffe/util/device_alternate.hpp"

namespace caffe {

class Timer {
 public:
  Timer();
  virtual ~Timer();
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();
  virtual float Seconds();

  [[nodiscard]] bool initialized() const { return initialized_; }
  [[nodiscard]] bool running() const { return running_; }
  [[nodiscard]] bool has_run_at_least_once() const { return has_run_at_least_once_; }

 protected:
  void Init();

  bool initialized_;
  bool running_;
  bool has_run_at_least_once_;
#ifndef CPU_ONLY
  cudaEvent_t start_gpu_;
  cudaEvent_t stop_gpu_;
#endif
  std::chrono::high_resolution_clock::time_point start_cpu_;
  std::chrono::high_resolution_clock::time_point stop_cpu_;

  std::chrono::duration<float> elapsed_time_;
};

class CPUTimer : public Timer {
 public:
  explicit CPUTimer();
  ~CPUTimer() override = default;
  void Start() override;
  void Stop() override;
  float MilliSeconds() override;
  float MicroSeconds() override;
};

}  // namespace caffe

#endif   // CAFFE_UTIL_BENCHMARK_H_
