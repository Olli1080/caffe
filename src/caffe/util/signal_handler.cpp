#include <glog/logging.h>

#include <csignal>

#include "caffe/util/signal_handler.h"

#ifndef SIGHUP
#define SIGHUP SIGBREAK
#endif

namespace {
  volatile sig_atomic_t got_sigint = false;
  volatile sig_atomic_t got_sighup = false;
  bool already_hooked_up = false;

  void handle_signal(int signal) {
    switch (signal) {
    case SIGHUP:
      got_sighup = true;
      break;
    case SIGINT:
      got_sigint = true;
      break;
    }
  }

  void HookupHandler() {
    if (already_hooked_up) {
      LOG(FATAL) << "Tried to hookup signal handlers more than once.";
    }
    already_hooked_up = true;

    if (std::signal(SIGHUP, handle_signal) == SIG_ERR) {
      LOG(FATAL) << "Cannot install SIGHUP handler.";
    }
    if (std::signal(SIGINT, handle_signal) == SIG_ERR) {
      LOG(FATAL) << "Cannot install SIGINT handler.";
    }
  }

  // Set the signal handlers to the default.
  void UnhookHandler() {
    if (already_hooked_up) {
      if (std::signal(SIGHUP, handle_signal) == SIG_ERR) {
        LOG(FATAL) << "Cannot uninstall SIGHUP handler.";
      }
      if (std::signal(SIGINT, handle_signal) == SIG_ERR) {
        LOG(FATAL) << "Cannot uninstall SIGINT handler.";
      }

      already_hooked_up = false;
    }
  }

  // Return true iff a SIGINT has been received since the last time this
  // function was called.
  bool GotSIGINT() {
    bool result = got_sigint;
    got_sigint = false;
    return result;
  }

  // Return true iff a SIGHUP has been received since the last time this
  // function was called.
  bool GotSIGHUP() {
    bool result = got_sighup;
    got_sighup = false;
    return result;
  }
}  // namespace

namespace caffe {

SignalHandler::SignalHandler(SolverAction::Enum SIGINT_action,
                             SolverAction::Enum SIGHUP_action):
  SIGINT_action_(SIGINT_action),
  SIGHUP_action_(SIGHUP_action) {
  HookupHandler();
}

SignalHandler::~SignalHandler() {
  UnhookHandler();
}

SolverAction::Enum SignalHandler::CheckForSignals() const {
  if (GotSIGHUP()) {
    return SIGHUP_action_;
  }
  if (GotSIGINT()) {
    return SIGINT_action_;
  }
  return SolverAction::NONE;
}

// Return the function that the solver can use to find out if a snapshot or
// early exit is being requested.
ActionCallback SignalHandler::GetActionFunction() const
{
  return [this] { return CheckForSignals(); };
}

}  // namespace caffe
