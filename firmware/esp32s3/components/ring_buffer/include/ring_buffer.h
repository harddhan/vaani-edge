#ifndef RING_BUFFER_H_
#define RING_BUFFER_H_

#include <cstddef>
#include <cstdint>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

class RingBuffer {
   public:
    RingBuffer(int16_t* storage, size_t capacity_samples);

    bool Push(const int16_t* samples, size_t count);

    size_t GetLast(int16_t* out, size_t count);

    size_t Read(int16_t* out, size_t count);

    size_t capacity() const { return capacity_; }

    size_t filled() const;

   private:
    int16_t* storage_;
    size_t capacity_;
    size_t write_pos_ = 0;
    size_t filled_ = 0;
    size_t read_pos_ = 0;

    SemaphoreHandle_t mutex_;
};

#endif