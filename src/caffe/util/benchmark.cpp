#include "caffe/common.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

Timer::Timer()
    : initialized_(false),
      running_(false),
      has_run_at_least_once_(false) {
  Init();
}

Timer::~Timer() {
  if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
    CUDA_CHECK(cudaEventDestroy(start_gpu_));
    CUDA_CHECK(cudaEventDestroy(stop_gpu_));
#else
    NO_GPU;
#endif
  }
}

void Timer::Start() {
  if (!running()) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      CUDA_CHECK(cudaEventRecord(start_gpu_, nullptr));
#else
      NO_GPU;
#endif
    } else {
      start_cpu_ = std::chrono::high_resolution_clock::now();
    }
    running_ = true;
    has_run_at_least_once_ = true;
  }
}

void Timer::Stop() {
  if (running()) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      CUDA_CHECK(cudaEventRecord(stop_gpu_, nullptr));
#else
      NO_GPU;
#endif
    } else {
      stop_cpu_ = std::chrono::high_resolution_clock::now();
    }
    running_ = false;
  }
}


float Timer::MicroSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
  	float elapsed_milliseconds;
    CUDA_CHECK(cudaEventSynchronize(stop_gpu_));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_milliseconds, start_gpu_,
                                    stop_gpu_));
    // Cuda only measure milliseconds with resolution of 0.5 microseconds
    elapsed_time_ = std::chrono::duration<float, std::ratio<1, 5000000>>(elapsed_milliseconds);
#else
      NO_GPU;
#endif
  } else {
    elapsed_time_ = stop_cpu_ - start_cpu_;
  }
  return std::chrono::duration<float, std::micro>(elapsed_time_).count();
}

float Timer::MilliSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
    float elapsed_milliseconds;
    CUDA_CHECK(cudaEventSynchronize(stop_gpu_));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_milliseconds, start_gpu_,
                                    stop_gpu_));
    elapsed_time_ = std::chrono::duration<float, std::ratio<1, 5000000>>(elapsed_milliseconds);
#else
      NO_GPU;
#endif
  } else {
  	elapsed_time_ = stop_cpu_ - start_cpu_;
  }
  return std::chrono::duration<float, std::milli>(elapsed_time_).count();
}

float Timer::Seconds() {
  return elapsed_time_.count();
}

void Timer::Init() {
  if (!initialized()) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      CUDA_CHECK(cudaEventCreate(&start_gpu_));
      CUDA_CHECK(cudaEventCreate(&stop_gpu_));
#else
      NO_GPU;
#endif
    }
    initialized_ = true;
  }
}

CPUTimer::CPUTimer() {
  this->initialized_ = true;
  this->running_ = false;
  this->has_run_at_least_once_ = false;
}

void CPUTimer::Start() {
  if (!running()) {
    this->start_cpu_ = std::chrono::high_resolution_clock::now();
    this->running_ = true;
    this->has_run_at_least_once_ = true;
  }
}

void CPUTimer::Stop() {
  if (running()) {
    this->stop_cpu_ = std::chrono::high_resolution_clock::now();
    this->running_ = false;
  }
}

float CPUTimer::MilliSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  elapsed_time_ = stop_cpu_ - start_cpu_;
  return std::chrono::duration<float, std::milli>(elapsed_time_).count();
}

float CPUTimer::MicroSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  elapsed_time_ = stop_cpu_ - start_cpu_;
  return std::chrono::duration<float, std::micro>(elapsed_time_).count();
}

}  // namespace caffe
