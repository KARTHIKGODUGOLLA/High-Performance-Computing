#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include "device.hxx"
#include "tensor.cuh"


Tensor::Tensor(const Device &_device, uint32_t _batch_size, uint32_t _channels, uint32_t _y_size, uint32_t _x_size) : device(_device), batch_size(_batch_size), channels(_channels), y_size(_y_size), x_size(_x_size) {

    //get the width of each dimension in the 1D float array
    channel_width = y_size * x_size;
    batch_width = channels * channel_width;
    size = batch_size * batch_width;

    if (device.is_cpu()) {
        values = new float[size];
        errors = new float[size];
    } else {
        //allocate the memory on the GPU
        cudaSetDevice(device.get_gpu());
        cudaMallocManaged(&values, size * sizeof(float));
        cudaMallocManaged(&errors, size * sizeof(float));
    }
}


Tensor::~Tensor() {
    if (device.is_cpu()) {
        delete values;
        delete errors;
    } else {
        cudaFree(values);
        cudaFree(errors);
    }
}

float Tensor::get(int32_t batch, int32_t channel, int32_t y, int32_t x) {
    return values[(batch * batch_width) + (channel * channel_width) + (y * x_size) + x];

}

void Tensor::set(int32_t batch, int32_t channel, int32_t y, int32_t x, float value) {
    values[(batch * batch_width) + (channel * channel_width) + (y * x_size) + x] = value;
}

void Tensor::accumulate(int32_t batch, int32_t channel, int32_t y, int32_t x, float value) {
    values[(batch * batch_width) + (channel * channel_width) + (y * x_size) + x] += value;
}



void Tensor::reset() {
    if (device.is_cpu()) {
        memset(values, 0, size * sizeof(float));
        memset(errors, 0, size * sizeof(float));
    } else {
        cudaMemset(values, 0, size * sizeof(float));
        cudaMemset(errors, 0, size * sizeof(float));
    }
}

void Tensor::print_values() {
    int32_t counter = 0;
    for (int32_t batch = 0; batch < batch_size; batch++) {
        cout << "BATCH: " << batch << endl;
        for (int32_t channel = 0; channel < channels; channel++) {
            cout << "\tBATCH " << batch << ", CHANNEL: " << channel << endl;
            for (int32_t y = 0; y < y_size; y++) {
                cout << "\t\t[";
                for (int32_t x = 0; x < x_size; x++) {
                    float value1 = get(batch, channel, y, x);
                    float value2 = values[counter];

                    if (value1 != value2) {
                        cerr << "\n\nERROR: get and increment did not return same value!" << endl;
                        exit(1);
                    }

                    cout << " " << value2;
                    counter++;
                }
                cout << " ]" << endl;
            }
        }
    }
    cout << endl;
}

ostream& operator<<(ostream& stream, const Tensor& tensor) {
    stream << "[tensor | " << tensor.batch_size << " x " << tensor.channels << " x " << tensor.y_size << " x " << tensor.x_size << "]";

    return stream;
}

ostream& operator<<(ostream& stream, const Tensor *tensor) {
    // this will call the non-pointer version by dereferencing
    // the pointer to tensor
    stream << *tensor;
    return stream;
}


