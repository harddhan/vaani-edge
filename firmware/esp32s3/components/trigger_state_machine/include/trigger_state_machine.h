// Trigger state machine: turns per-window keyword probabilities into a
// LISTENING / KEYWORD_DETECTED / STREAMING / SESSION_COMPLETE /
// ERROR_RECOVERY state sequence, mirroring server/session.py's
// TriggerStateMachine (the Python reference implementation used for
// offline evaluation). Keep this logic in sync with that file - see
// docs/state_machine.md for the shared spec.
#ifndef TRIGGER_STATE_MACHINE_H_
#define TRIGGER_STATE_MACHINE_H_

#include <cstdint>

enum class AppState {
    kListening,
    kKeywordDetected,
    kStreaming,
    kSessionComplete,
    kErrorRecovery,
};

class TriggerStateMachine {
   public:
    TriggerStateMachine(float threshold, int consecutive_positive_windows, int cooldown_windows);

    // Feed one window's keyword-class probability. Returns the new
    // AppState after processing this window. Callers drive their own
    // transition out of kKeywordDetected into kStreaming (once pre-roll
    // + streaming setup begins) and out of kStreaming into
    // kSessionComplete (once silence/timeout/END_SESSION is reached) via
    // ForceState(); this class only owns the *detection* half of the
    // state machine (LISTENING <-> KEYWORD_DETECTED).
    AppState Update(float probability);

    // Allows the caller (main loop) to explicitly advance/reset state,
    // e.g. AppState::kStreaming once streaming setup completes, or back
    // to AppState::kListening after SESSION_COMPLETE / ERROR_RECOVERY.
    void ForceState(AppState state);

    AppState state() const { return state_; }

   private:
    float threshold_;
    int consecutive_positive_windows_;
    int cooldown_windows_;

    int consecutive_count_ = 0;
    int cooldown_remaining_ = 0;
    AppState state_ = AppState::kListening;
};

#endif  // TRIGGER_STATE_MACHINE_H_
