#include "device.hxx"


Device::Device(int32_t _device_number) : device_number(_device_number) {
}

bool Device::is_cpu() const {
    return device_number < 0;
}

bool Device::is_gpu() const {
    return device_number >= 0;
}

int32_t Device::get_gpu() const {
    return device_number;
}

ostream& operator<<(ostream& stream, const Device& device) {
    if (device.is_cpu()) {
        stream << "CPU";
    } else {
        stream << "GPU: " << device.device_number;
    }

    return stream;
}


