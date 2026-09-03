#include "ring_buffer.h"

#include <algorithm>
#include <cstring>

namespace {
constexpr TickType_t kMutexTimeoutTicks = pdMS_TO_TICKS(5);
}

RingBuffer::RingBuffer(
    int16_t* storage,
    size_t capacity_samples
)
    : storage_(storage),
      capacity_(capacity_samples) {
    mutex_ = xSemaphoreCreateMutex();
}

bool RingBuffer::Push(
    const int16_t* samples,
    size_t count
) {
    if (samples == nullptr ||
        count == 0 ||
        capacity_ == 0) {
        return false;
    }

    if (xSemaphoreTake(
            mutex_,
            kMutexTimeoutTicks
        ) != pdTRUE) {
        return false;
    }

    if (count > capacity_) {
        samples += count - capacity_;
        count = capacity_;
    }

    size_t end = write_pos_ + count;

    if (end <= capacity_) {
        std::memcpy(
            storage_ + write_pos_,
            samples,
            count * sizeof(int16_t)
        );
    } else {
        size_t first_part =
            capacity_ - write_pos_;

        std::memcpy(
            storage_ + write_pos_,
            samples,
            first_part * sizeof(int16_t)
        );

        std::memcpy(
            storage_,
            samples + first_part,
            (count - first_part) * sizeof(int16_t)
        );
    }

    write_pos_ = end % capacity_;

    if (filled_ + count > capacity_) {
        size_t overflow =
            filled_ + count - capacity_;

        read_pos_ =
            (read_pos_ + overflow) % capacity_;

        filled_ = capacity_;
    } else {
        filled_ += count;
    }

    xSemaphoreGive(mutex_);

    return true;
}

size_t RingBuffer::GetLast(
    int16_t* out,
    size_t count
) {
    if (out == nullptr ||
        count == 0 ||
        capacity_ == 0) {
        return 0;
    }

    if (xSemaphoreTake(
            mutex_,
            kMutexTimeoutTicks
        ) != pdTRUE) {
        return 0;
    }

    count = std::min(count, filled_);

    if (count == 0) {
        xSemaphoreGive(mutex_);
        return 0;
    }

    size_t start =
        (write_pos_ + capacity_ - count) %
        capacity_;

    if (start + count <= capacity_) {
        std::memcpy(
            out,
            storage_ + start,
            count * sizeof(int16_t)
        );
    } else {
        size_t first_part =
            capacity_ - start;

        std::memcpy(
            out,
            storage_ + start,
            first_part * sizeof(int16_t)
        );

        std::memcpy(
            out + first_part,
            storage_,
            (count - first_part) * sizeof(int16_t)
        );
    }

    xSemaphoreGive(mutex_);

    return count;
}

size_t RingBuffer::Read(
    int16_t* out,
    size_t count
) {
    if (out == nullptr ||
        count == 0 ||
        capacity_ == 0) {
        return 0;
    }

    if (xSemaphoreTake(
            mutex_,
            kMutexTimeoutTicks
        ) != pdTRUE) {
        return 0;
    }

    count = std::min(count, filled_);

    if (count == 0) {
        xSemaphoreGive(mutex_);
        return 0;
    }

    size_t end = read_pos_ + count;

    if (end <= capacity_) {
        std::memcpy(
            out,
            storage_ + read_pos_,
            count * sizeof(int16_t)
        );
    } else {
        size_t first_part =
            capacity_ - read_pos_;

        std::memcpy(
            out,
            storage_ + read_pos_,
            first_part * sizeof(int16_t)
        );

        std::memcpy(
            out + first_part,
            storage_,
            (count - first_part) * sizeof(int16_t)
        );
    }

    read_pos_ =
        (read_pos_ + count) % capacity_;

    filled_ -= count;

    xSemaphoreGive(mutex_);

    return count;
}

size_t RingBuffer::filled() const {
    return filled_;
}