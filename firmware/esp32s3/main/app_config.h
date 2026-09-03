#ifndef APP_CONFIG_H_
#define APP_CONFIG_H_

#include <cstdint>

namespace app_config {

constexpr uint32_t kSampleRateHz = 16000;
constexpr uint8_t kChannels = 1;

constexpr uint32_t kFrameDurationMs = 30;
constexpr uint32_t kFrameSizeSamples = 480;

constexpr uint32_t kPreRollMs = 800;
constexpr uint32_t kPreRollSamples = 12800;

constexpr uint32_t kRingBufferSamples = 32000;
constexpr uint32_t kMaxStreamDurationMs = 8000;

constexpr int16_t kSilenceRmsThreshold = 200;
constexpr uint32_t kSilenceHangMs = 500;

constexpr int kNumMelBins = 40;
constexpr int kNumMfcc = 13;
constexpr int kNumFrames = 50;
constexpr int kFftSize = 256;

constexpr int kWindowLengthSamples = 512;
constexpr int kHopLengthSamples = 320;

constexpr float kPreEmphasis = 0.98f;

constexpr float kDetectionThreshold = 0.80f;
constexpr int kConsecutivePositiveWindows = 3;
constexpr uint32_t kCooldownMs = 1500;

constexpr int kNumClasses = 4;

constexpr int kTensorArenaBytes = 60 * 1024;

constexpr const char* kServerHost = "192.168.1.100";
constexpr uint16_t kServerPort = 8765;

constexpr uint32_t kConnectTimeoutMs = 5000;
constexpr uint32_t kSessionTimeoutMs = 15000;

constexpr uint32_t kTcpSendBufferBytes = 4096;
constexpr uint32_t kTcpRecvBufferBytes = 2048;

constexpr const char* kWifiSsid = "REPLACE_ME_SSID";
constexpr const char* kWifiPassword = "REPLACE_ME_PASSWORD";

}

#endif