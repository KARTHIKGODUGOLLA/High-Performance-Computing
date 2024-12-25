#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;


#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <string>
using std::string;

#include "device.hxx"
#include "exceptions.hxx"
#include "tensor.cuh"
#include "convolution_2d.cuh"
#include "test_common.hxx"

bool tensors_close_enough(string t1_name, Tensor *t1, string t2_name, Tensor *t2) {
    float t1_value;
    float t2_value;

    bool mismatch = false;

    for (int32_t batch = 0; batch < t1->batch_size; batch++) {
        for (int32_t channel = 0; channel < t1->channels; channel++) {
            for (int32_t y = 0; y < t1->y_size; y++) {
                for (int32_t x = 0; x < t1->x_size; x++) {
                    t1_value = t1->get(batch, channel, y, x);
                    t2_value = t2->get(batch, channel, y, x);

                    if (!(close_enough(t1_value, t2_value))) {
                        cout << "tensors not close enough on batch: " << batch << ", channel: " << channel << ", y: " << y << ", x: " << x << ", " << t1_name << " value: " << t1_value << ", " << t2_name << " value: " << t2_value << endl;
                        mismatch = true;
                    }
                }
            }
        }
    }

    return !mismatch;
}

bool convolutions_close_enough(string c1_name, Convolution2D *c1, string c2_name, Convolution2D *c2) {
    float c1_value;
    float c2_value;

    bool mismatch = false;

    for (int32_t in_channel = 0; in_channel < c1->in_channels; in_channel++) {
        for (int32_t out_channel = 0; out_channel < c1->out_channels; out_channel++) {
            for (int32_t y = 0; y < c1->y_size; y++) {
                for (int32_t x = 0; x < c1->x_size; x++) {
                    c1_value = c1->get(in_channel, out_channel, y, x);
                    c2_value = c2->get(in_channel, out_channel, y, x);

                    if (!(close_enough(c1_value, c2_value))) {
                        cout << "convolutions not close enough on in channel: " << in_channel << ", out_channel: " << out_channel << ", y: " << y << ", x: " << x << ", " << c1_name << " value: " << c1_value << ", " << c2_name << " value: " << c2_value << endl;
                        mismatch = true;
                    }
                }
            }
        }
    }

    return !mismatch;
}


void randomize_tensor(Tensor *t) {
    for (int32_t batch = 0; batch < t->batch_size; batch++) {
        for (int32_t channel = 0; channel < t->channels; channel++) {
            for (int32_t y = 0; y < t->y_size; y++) {
                for (int32_t x = 0; x < t->x_size; x++) {
                    t->set(batch, channel, y, x, (drand48() * 2.0) - 1.0);
                }
            }
        }
    }
}

void randomize_convolution(Convolution2D *c) {
    for (int32_t in_channel = 0; in_channel < c->in_channels; in_channel++) {
        for (int32_t out_channel = 0; out_channel < c->out_channels; out_channel++) {
            for (int32_t y = 0; y < c->y_size; y++) {
                for (int32_t x = 0; x < c->x_size; x++) {
                    c->set(in_channel, out_channel, y, x, (drand48() * 2.0) - 1.0);
                }
            }
        }
    }
}

/**
 * Creates a copy of the tensor on the given device.
 */
Tensor* copy_tensor(const Device &device, Tensor *t) {
    Tensor *copy = new Tensor(device, t->batch_size, t->channels, t->y_size, t->x_size);

    for (int32_t i = 0; i < t->size; i++) {
        copy->values[i] = t->values[i];
    }

    return copy;
}

/**
 * Copies weights from one c1 into to c2.
 */
void copy_weights(Convolution2D *c1, Convolution2D *c2) {
    for (int32_t i = 0; i < c1->n_weights; i++) {
        c2->weights[i] = c1->weights[i];
    }
}


void large_validation(int32_t batch_size, int32_t in_channels, int32_t in_size, int32_t out_channels, int32_t out_size, int32_t conv_size, int32_t padding) {
    Device cpu_device(-1);
    Device gpu_device(0);
    //Create initial tensors and do a slow forward pass through the convolution as a baseline.

    //these tensors should only have one value
    Tensor *input_tensor = new Tensor(cpu_device, batch_size, in_channels, in_size, in_size);
    Tensor *output_tensor = new Tensor(cpu_device, batch_size, out_channels, out_size, out_size);
    randomize_tensor(input_tensor);
    randomize_tensor(output_tensor);

    Tensor *fast_cpu_input_tensor = copy_tensor(cpu_device, input_tensor);
    Tensor *slow_gpu_input_tensor = copy_tensor(gpu_device, input_tensor);
    Tensor *fast_gpu_input_tensor = copy_tensor(gpu_device, input_tensor);

    Tensor *fast_cpu_output_tensor = copy_tensor(cpu_device, output_tensor);
    Tensor *slow_gpu_output_tensor = copy_tensor(gpu_device, output_tensor);
    Tensor *fast_gpu_output_tensor = copy_tensor(gpu_device, output_tensor);

    Convolution2D *convolution = new Convolution2D(cpu_device, conv_size, conv_size, padding, input_tensor, output_tensor);
    randomize_convolution(convolution);

    Convolution2D *fast_cpu_convolution = new Convolution2D(cpu_device, conv_size, conv_size, padding, fast_cpu_input_tensor, fast_cpu_output_tensor);
    copy_weights(convolution, fast_cpu_convolution);

    Convolution2D *slow_gpu_convolution = new Convolution2D(gpu_device, conv_size, conv_size, padding, slow_gpu_input_tensor, slow_gpu_output_tensor);
    copy_weights(convolution, slow_gpu_convolution);

    Convolution2D *fast_gpu_convolution = new Convolution2D(gpu_device, conv_size, conv_size, padding, fast_gpu_input_tensor, fast_gpu_output_tensor);
    copy_weights(convolution, fast_gpu_convolution);

    auto slow_cpu_t1 = high_resolution_clock::now();
    convolution->forward_slow();
    auto slow_cpu_t2 = high_resolution_clock::now();

    cudaDeviceSynchronize();
    auto fast_cpu_t1 = high_resolution_clock::now();
    fast_cpu_convolution->forward();
    auto fast_cpu_t2 = high_resolution_clock::now();

    auto slow_gpu_t1 = high_resolution_clock::now();
    slow_gpu_convolution->forward_slow();
    auto slow_gpu_t2 = high_resolution_clock::now();

    auto fast_gpu_t1 = high_resolution_clock::now();
    fast_gpu_convolution->forward();
    auto fast_gpu_t2 = high_resolution_clock::now();

    if (!tensors_close_enough("slow cpu", output_tensor, "fast cpu", fast_cpu_output_tensor)) {
        cout << "output tensor (slow cpu):" << endl;
        output_tensor->print_values();

        cout << "output tensor (fast cpu):" << endl;
        fast_cpu_output_tensor->print_values();

        cout << "output tensors (slow cpu vs fast cpu) were not close enough!" << endl;
        exit(1);
    }

    if (!tensors_close_enough("slow cpu", output_tensor, "slow gpu", slow_gpu_output_tensor)) {
        cout << "output tensor (slow cpu):" << endl;
        output_tensor->print_values();

        cout << "output tensor (slow gpu):" << endl;
        slow_gpu_output_tensor->print_values();

        cout << "output tensors (slow cpu vs slow gpu) were not close enough!" << endl;
        exit(1);
    }

    if (!tensors_close_enough("slow cpu", output_tensor, "fast gpu", fast_gpu_output_tensor)) {
        cout << "output tensor (slow cpu):" << endl;
        output_tensor->print_values();

        cout << "output tensor (fast gpu):" << endl;
        fast_gpu_output_tensor->print_values();

        cout << "output tensors (slow cpu vs fast gpu) were not close enough!" << endl;
        exit(1);
    }

    duration<double, std::milli> slow_cpu_time = slow_cpu_t2 - slow_cpu_t1;
    duration<double, std::milli> fast_cpu_time = fast_cpu_t2 - fast_cpu_t1;
    duration<double, std::milli> slow_gpu_time = slow_gpu_t2 - slow_gpu_t1;
    duration<double, std::milli> fast_gpu_time = fast_gpu_t2 - fast_gpu_t1;


    cout << "slow cpu runtime: " << slow_cpu_time.count() << " ms." << endl;
    cout << "fast cpu runtime: " << fast_cpu_time.count() << " ms." << endl;
    cout << "slow gpu runtime: " << slow_gpu_time.count() << " ms." << endl;
    cout << "fast gpu runtime: " << fast_gpu_time.count() << " ms." << endl;

    delete input_tensor;
    delete output_tensor;
    delete convolution;

    delete fast_cpu_input_tensor;
    delete fast_cpu_output_tensor;
    delete fast_cpu_convolution;

    delete slow_gpu_input_tensor;
    delete slow_gpu_output_tensor;
    delete slow_gpu_convolution;

    delete fast_gpu_input_tensor;
    delete fast_gpu_output_tensor;
    delete fast_gpu_convolution;
}

int main(int argc, char** argv) {
    int32_t batch_size = 5;
    int32_t in_channels = 8;
    int32_t in_size = 8;
    int32_t out_channels = 16;
    int32_t out_size = 8;
    int32_t conv_size = 3;
    int32_t padding = 1;

    try {
        large_validation(batch_size, in_channels, in_size, out_channels, out_size, conv_size, padding);
    } catch (int error) {
        cout << "ERROR was: " << ERROR_NAMES[error] << endl;
        exit(1);
    }

    batch_size = 20;
    in_channels = 16;
    in_size = 16;
    out_channels = 32;
    out_size = 16;
    conv_size = 3;
    padding = 1;

    try {
        large_validation(batch_size, in_channels, in_size, out_channels, out_size, conv_size, padding);
    } catch (int error) {
        cout << "ERROR was: " << ERROR_NAMES[error] << endl;
        exit(1);
    }

    return 0;
}

