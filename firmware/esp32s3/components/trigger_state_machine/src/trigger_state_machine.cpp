#include "trigger_state_machine.h"

TriggerStateMachine::TriggerStateMachine(float threshold, int consecutive_positive_windows,
                                           int cooldown_windows)
    : threshold_(threshold),
      consecutive_positive_windows_(consecutive_positive_windows > 0 ? consecutive_positive_windows : 1),
      cooldown_windows_(cooldown_windows > 0 ? cooldown_windows : 0) {}

AppState TriggerStateMachine::Update(float probability) {
    if (state_ != AppState::kListening) {
        // Only LISTENING actively evaluates new windows for a trigger;
        // once KEYWORD_DETECTED/STREAMING/SESSION_COMPLETE/ERROR_RECOVERY
        // is entered, the main loop (app_main.cpp) drives further
        // transitions explicitly via ForceState().
        return state_;
    }

    if (cooldown_remaining_ > 0) {
        --cooldown_remaining_;
        consecutive_count_ = 0;
        return state_;
    }

    if (probability >= threshold_) {
        ++consecutive_count_;
    } else {
        consecutive_count_ = 0;
    }

    if (consecutive_count_ >= consecutive_positive_windows_) {
        consecutive_count_ = 0;
        cooldown_remaining_ = cooldown_windows_;
        state_ = AppState::kKeywordDetected;
    }

    return state_;
}

void TriggerStateMachine::ForceState(AppState state) {
    state_ = state;
    if (state == AppState::kListening) {
        consecutive_count_ = 0;
        // cooldown_remaining_ is intentionally NOT reset here: cooldown
        // continues to suppress re-triggers immediately after a session
        // completes, per the "trigger cooldown" requirement.
    }
}
