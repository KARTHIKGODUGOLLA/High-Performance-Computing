#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include "device.hxx"
#include "exceptions.hxx"
#include "tensor.cuh"
#include "convolution_2d.cuh"
#include "test_common.hxx"

void test_constructor(const Device &device) {
    cout << "device is: " << device << endl;

    Tensor *input_tensor = new Tensor(device, 10, 5, 5, 5);
    Tensor *output_tensor = new Tensor(device, 10, 5, 5, 5);

    cout << "input tensor: " << input_tensor << endl;
    cout << "output tensor: " << output_tensor << endl;

    Convolution2D *convolution = new Convolution2D(device, 3, 3, 1, input_tensor, output_tensor);
    cout << "convolution: " << convolution << endl;

    int32_t expected_n_weights = input_tensor->channels * output_tensor->channels * 3 * 3;
    if (convolution->n_weights != expected_n_weights) {
        cerr << "ERROR: number of convolution weihts was incorrect, was " << convolution->n_weights << " and should have been: " << expected_n_weights << endl;
        exit(1);
    }

    delete input_tensor;
    delete output_tensor;
    delete convolution;
}

void test_invalid_convolution_size() {
    Device device = Device(-1);
    cout << "device is: " << device << endl;

    Tensor *input_tensor = new Tensor(device, 10, 5, 5, 4);
    Tensor *output_tensor = new Tensor(device, 10, 5, 5, 5);

    cout << "input tensor: " << input_tensor << endl;
    cout << "output tensor: " << output_tensor << endl;

    Convolution2D *convolution = NULL;

    bool exception_caught = false;
    try {
        convolution = new Convolution2D(device, 3, 3, 1, input_tensor, output_tensor);
        cout << "convolution: " << convolution << endl;
    } catch (int32_t error_code) {
        if (error_code == INVALID_CONVOLUTION_SIZE) {
            cout << "correctly caught INVALID_CONVOLUTION_SIZE" << endl;
            exception_caught = true;
        }
    }

    if (!exception_caught) {
        cerr << "ERROR: constructing the convolution should have thrown an INVALID_CONVOLUTION_SIZE exception" << endl;
        exit(1);
    }

    delete input_tensor;
    delete output_tensor;
    if (convolution != NULL) {
        delete convolution;
    }
}

void test_invalid_batch_size() {
    Device device = Device(-1);
    cout << "device is: " << device << endl;

    Tensor *input_tensor = new Tensor(device, 7, 5, 5, 5);
    Tensor *output_tensor = new Tensor(device, 10, 5, 5, 5);

    cout << "input tensor: " << input_tensor << endl;
    cout << "output tensor: " << output_tensor << endl;

    Convolution2D *convolution = NULL;

    bool exception_caught = false;
    try {
        convolution = new Convolution2D(device, 3, 3, 1, input_tensor, output_tensor);
        cout << "convolution: " << convolution << endl;
    } catch (int32_t error_code) {
        if (error_code == BATCH_SIZE_MISMATCH) {
            cout << "correctly caught BATCH_SIZE_MISMATCH" << endl;
            exception_caught = true;
        }
    }

    if (!exception_caught) {
        cerr << "ERROR: constructing the convolution should have thrown an INVALID_CONVOLUTION_SIZE exception" << endl;
        exit(1);
    }

    delete input_tensor;
    delete output_tensor;
    if (convolution != NULL) {
        delete convolution;
    }
}

void test_1_batch_1_to_1_channel(const Device &device, bool slow) {
    cout << "device is: " << device << endl;

    //these tensors should only have one value
    Tensor *input_tensor = new Tensor(device, 1, 1, 1, 1);
    Tensor *output_tensor = new Tensor(device, 1, 1, 1, 1);

    cout << "input tensor: " << input_tensor << endl;
    input_tensor->reset();
    input_tensor->values[0] = 1.0;
    input_tensor->print_values();

    cout << "output tensor: " << output_tensor << endl;
    output_tensor->reset();
    output_tensor->values[0] = 1.0;
    output_tensor->print_values();

    Convolution2D *convolution = NULL;

    convolution = new Convolution2D(device, 3, 3, 1, input_tensor, output_tensor);
    cout << "convolution: " << convolution << endl;
    convolution->reset();

    //set the convolution with some values for testing
    float value = 0.1;
    for (int32_t y = 0; y < convolution->y_size; y++) {
        for (int32_t x = 0; x < convolution->x_size; x++) {
            convolution->set(0, 0, y, x, value);
            value += 0.1;
        }
        value *= 2.0;
    }
    convolution->print_weights();

    if (slow) {
        convolution->forward_slow();
    } else {
        convolution->forward();
    }

    cout << "output tensor after convolution: " << output_tensor << endl;
    output_tensor->print_values();
    cout << "output_tensor->values[0]: " << output_tensor->values[0] << endl;

    if (!close_enough(output_tensor->values[0], 1.9)) {
        cerr << "ERROR: foward convolution was broken" << endl;
        exit(1);
    }

    delete input_tensor;
    delete output_tensor;
    delete convolution;
}

void test_1_batch_1_to_1_channel_3x3(const Device &device, bool slow) {
    cout << "device is: " << device << endl;

    //these tensors should only have one value
    Tensor *input_tensor = new Tensor(device, 1, 1, 3, 3);
    Tensor *output_tensor = new Tensor(device, 1, 1, 1, 1);

    cout << "input tensor: " << input_tensor << endl;
    input_tensor->reset();
    for (int32_t y = 0; y < 3; y++) {
        for (int32_t x = 0; x < 3; x++) {
            input_tensor->set(0, 0, y, x, 1.0);
        }
    }
    input_tensor->print_values();

    cout << "output tensor: " << output_tensor << endl;
    output_tensor->reset();
    output_tensor->values[0] = 1.0;
    output_tensor->print_values();

    Convolution2D *convolution = NULL;

    //no padding will take a 3x3 tensor down to a 1x1 tensor with a 3x3 convolution
    convolution = new Convolution2D(device, 3, 3, 0, input_tensor, output_tensor);
    cout << "convolution: " << convolution << endl;
    convolution->reset();

    //set the convolution with some weights for testing
    float weight = 0.1;
    for (int32_t y = 0; y < convolution->y_size; y++) {
        for (int32_t x = 0; x < convolution->x_size; x++) {
            convolution->set(0, 0, y, x, weight);
            weight += 0.1;
        }
        weight *= 2.0;
    }
    convolution->print_weights();

    if (slow) {
        convolution->forward_slow();
    } else {
        convolution->forward();
    }

    cout << "output tensor after convolution: " << output_tensor << endl;
    output_tensor->print_values();

    if (!close_enough(output_tensor->values[0], 11.2)) {
        cerr << "ERROR: foward convolution was broken" << endl;
        exit(1);
    }

    delete input_tensor;
    delete output_tensor;
    delete convolution;
}

void test_1_batch_1_to_2_channel_3x3(const Device &device, bool slow) {
    cout << "device is: " << device << endl;

    //these tensors should only have one value
    Tensor *input_tensor = new Tensor(device, 1, 1, 3, 3);
    Tensor *output_tensor = new Tensor(device, 1, 2, 1, 1);

    cout << "input tensor: " << input_tensor << endl;
    input_tensor->reset();
    for (int32_t y = 0; y < 3; y++) {
        for (int32_t x = 0; x < 3; x++) {
            input_tensor->set(0, 0, y, x, 1.0);
        }
    }
    input_tensor->print_values();

    cout << "output tensor: " << output_tensor << endl;
    output_tensor->reset();
    output_tensor->values[0] = 1.0;
    output_tensor->values[1] = 0.5;
    output_tensor->print_values();

    Convolution2D *convolution = NULL;

    //no padding will take a 3x3 tensor down to a 1x1 tensor with a 3x3 convolution
    convolution = new Convolution2D(device, 3, 3, 0, input_tensor, output_tensor);
    cout << "convolution: " << convolution << endl;
    convolution->reset();

    //set the convolution with some weights for testing
    float weight = 0.1;
    for (int32_t channel = 0; channel < 2; channel++) {
        for (int32_t y = 0; y < convolution->y_size; y++) {
            for (int32_t x = 0; x < convolution->x_size; x++) {
                convolution->set(0, channel, y, x, weight);
                weight += 0.1;
            }
            weight *= 2.0;
        }
        weight = 0.2;
    }
    convolution->print_weights();

    if (slow) {
        convolution->forward_slow();
    } else {
        convolution->forward();
    }

    cout << "output tensor after convolution: " << output_tensor << endl;
    output_tensor->print_values();

    if (!close_enough(output_tensor->values[0], 11.2)) {
        cerr << "ERROR: foward convolution was broken, got " << output_tensor->values[0] << ", expected: " << 11.2 << endl;
        exit(1);
    }

    if (!close_enough(output_tensor->values[1], 12.8)) {
        cerr << "ERROR: foward convolution was broken, got " << output_tensor->values[1] << ", expected: " << 13.3 << endl;
        exit(1);
    }

    delete input_tensor;
    delete output_tensor;
    delete convolution;
}

void test_1_batch_2_to_2_channel_3x3(const Device &device, bool slow) {
    cout << "device is: " << device << endl;

    //these tensors should only have one value
    Tensor *input_tensor = new Tensor(device, 1, 2, 3, 3);
    Tensor *output_tensor = new Tensor(device, 1, 2, 1, 1);

    cout << "input tensor: " << input_tensor << endl;
    input_tensor->reset();
    for (int32_t channel = 0; channel < 2; channel++) {
        for (int32_t y = 0; y < 3; y++) {
            for (int32_t x = 0; x < 3; x++) {
                input_tensor->set(0, channel, y, x, (channel + 1.0) / 2.0);
            }
        }
    }
    input_tensor->print_values();

    cout << "output tensor: " << output_tensor << endl;
    output_tensor->reset();
    output_tensor->values[0] = 1.0;
    output_tensor->values[1] = 0.5;
    output_tensor->print_values();

    Convolution2D *convolution = NULL;

    //no padding will take a 3x3 tensor down to a 1x1 tensor with a 3x3 convolution
    convolution = new Convolution2D(device, 3, 3, 0, input_tensor, output_tensor);
    cout << "convolution: " << convolution << endl;
    convolution->reset();

    //set the convolution with some weights for testing
    float weight;
    for (int32_t in_channel = 0; in_channel < 2; in_channel++) {
        weight = 0.1;
        for (int32_t out_channel = 0; out_channel < 2; out_channel++) {
            for (int32_t y = 0; y < convolution->y_size; y++) {
                for (int32_t x = 0; x < convolution->x_size; x++) {
                    convolution->set(in_channel, out_channel, y, x, weight);
                    weight += 0.1;
                }
                weight *= 2.0;
            }
            weight = 0.2;
        }
    }
    convolution->print_weights();

    if (slow) {
        convolution->forward_slow();
    } else {
        convolution->forward();
    }

    cout << "output tensor after convolution: " << output_tensor << endl;
    output_tensor->print_values();

    if (!close_enough(output_tensor->values[0], 16.3)) {
        cerr << "ERROR: foward convolution was broken, got " << output_tensor->values[0] << ", expected: " << 16.3 << endl;
        exit(1);
    }

    if (!close_enough(output_tensor->values[1], 18.95)) {
        cerr << "ERROR: foward convolution was broken, got " << output_tensor->values[1] << ", expected: " << 18.95 << endl;
        exit(1);
    }

    delete input_tensor;
    delete output_tensor;
    delete convolution;
}

void test_2_batch_2_to_2_channel_3x3(const Device &device, bool slow) {
    cout << "device is: " << device << endl;

    //these tensors should only have one value
    Tensor *input_tensor = new Tensor(device, 2, 2, 3, 3);
    Tensor *output_tensor = new Tensor(device, 2, 2, 1, 1);

    cout << "input tensor: " << input_tensor << endl;
    input_tensor->reset();
    for (int32_t batch = 0; batch < 2; batch++) {
        for (int32_t channel = 0; channel < 2; channel++) {
            for (int32_t y = 0; y < 3; y++) {
                for (int32_t x = 0; x < 3; x++) {
                    input_tensor->set(batch, channel, y, x, ((channel + 1.0) / 2.0) + batch);
                }
            }
        }
    }
    input_tensor->print_values();

    cout << "output tensor: " << output_tensor << endl;
    output_tensor->reset();
    output_tensor->values[0] = 1.0;
    output_tensor->values[1] = 0.5;
    output_tensor->values[2] = 2.0;
    output_tensor->values[3] = 3.0;
    output_tensor->print_values();

    Convolution2D *convolution = NULL;

    //no padding will take a 3x3 tensor down to a 1x1 tensor with a 3x3 convolution
    convolution = new Convolution2D(device, 3, 3, 0, input_tensor, output_tensor);
    cout << "convolution: " << convolution << endl;
    convolution->reset();

    //set the convolution with some weights for testing
    float weight;
    for (int32_t in_channel = 0; in_channel < 2; in_channel++) {
        weight = 0.1;
        for (int32_t out_channel = 0; out_channel < 2; out_channel++) {
            for (int32_t y = 0; y < convolution->y_size; y++) {
                for (int32_t x = 0; x < convolution->x_size; x++) {
                    convolution->set(in_channel, out_channel, y, x, weight);
                    weight += 0.1;
                }
                weight *= 2.0;
            }
            weight = 0.2;
        }
    }
    convolution->print_weights();

    if (slow) {
        convolution->forward_slow();
    } else {
        convolution->forward();
    }

    cout << "output tensor after convolution: " << output_tensor << endl;
    output_tensor->print_values();

    if (!close_enough(output_tensor->values[0], 16.3)) {
        cerr << "ERROR: foward convolution was broken, got " << output_tensor->values[0] << ", expected: " << 16.3 << endl;
        exit(1);
    }

    if (!close_enough(output_tensor->values[1], 18.95)) {
        cerr << "ERROR: foward convolution was broken, got " << output_tensor->values[1] << ", expected: " << 18.95 << endl;
        exit(1);
    }

    if (!close_enough(output_tensor->values[2], 37.7)) {
        cerr << "ERROR: foward convolution was broken, got " << output_tensor->values[2] << ", expected: " << 37.7 << endl;
        exit(1);
    }

    if (!close_enough(output_tensor->values[3], 46.05)) {
        cerr << "ERROR: foward convolution was broken, got " << output_tensor->values[3] << ", expected: " << 46.05 << endl;
        exit(1);
    }

    delete input_tensor;
    delete output_tensor;
    delete convolution;
}






int main(int argc, char** argv) {
    Device cpu_device(-1);
    Device gpu_device(0);

    test_constructor(cpu_device);
    test_constructor(gpu_device);

    test_invalid_convolution_size();
    test_invalid_batch_size();

    // test the slow easy version
    bool slow = true;
    cout << "\n\n********** TEST test_1_batch_1_to_1_channel CPU SLOW" << endl;
    test_1_batch_1_to_1_channel(cpu_device, slow);
    cout << "\n\n********** TEST test_1_batch_1_to_1_channel_3x3 CPU SLOW" << endl;
    test_1_batch_1_to_1_channel_3x3(cpu_device, slow);
    cout << "\n\n********** TEST test_1_batch_1_to_2_channel_3x3 CPU SLOW" << endl;
    test_1_batch_1_to_2_channel_3x3(cpu_device, slow);
    cout << "\n********** TEST test_1_batch_2_to_2_channel_3x3 CPU SLOW" << endl;
    test_1_batch_2_to_2_channel_3x3(cpu_device, slow);
    cout << "\n********** TEST test_2_batch_2_to_2_channel_3x3 CPU SLOW" << endl;
    test_2_batch_2_to_2_channel_3x3(cpu_device, slow);

    // test the fast hard version
    slow = false;
    cout << "\n\n********** TEST test_1_batch_1_to_1_channel CPU FAST" << endl;
    test_1_batch_1_to_1_channel(cpu_device, slow);
    cout << "\n\n********** TEST test_1_batch_1_to_1_channel_3x3 CPU FAST" << endl;
    test_1_batch_1_to_1_channel_3x3(cpu_device, slow);
    cout << "\n\n********** TEST test_1_batch_1_to_2_channel_3x3 CPU FAST" << endl;
    test_1_batch_1_to_2_channel_3x3(cpu_device, slow);
    cout << "\n\n********** TEST test_1_batch_2_to_2_channel_3x3 CPU FAST" << endl;
    test_1_batch_2_to_2_channel_3x3(cpu_device, slow);
    cout << "\n\n********** TEST test_2_batch_2_to_2_channel_3x3 CPU FAST" << endl;
    test_2_batch_2_to_2_channel_3x3(cpu_device, slow);

    //run these on the GPU with the global memory only version
    slow = true;
    cout << "\n\n********** TEST test_1_batch_1_to_1_channel GPU SLOW" << endl;
    test_1_batch_1_to_1_channel(gpu_device, slow);
    cout << "\n\n********** TEST test_1_batch_1_to_1_channel GPU SLOW" << endl;
    test_1_batch_1_to_1_channel(gpu_device, slow);
    cout << "\n\n********** TEST test_1_batch_1_to_1_channel_3x3 GPU SLOW" << endl;
    test_1_batch_1_to_1_channel_3x3(gpu_device, slow);
    cout << "\n\n********** TEST test_1_batch_1_to_2_channel_3x3 GPU SLOW" << endl;
    test_1_batch_1_to_2_channel_3x3(gpu_device, slow);
    cout << "\n\n********** TEST test_1_batch_2_to_2_channel_3x3 GPU SLOW" << endl;
    test_1_batch_2_to_2_channel_3x3(gpu_device, slow);
    cout << "\n\n********** TEST test_2_batch_2_to_2_channel_3x3 GPU SLOW" << endl;
    test_2_batch_2_to_2_channel_3x3(gpu_device, slow);

    //run these on the GPU with the shared memory version
    slow = false;
    cout << "\n\n********** TEST test_1_batch_1_to_1_channel GPU FAST" << endl;
    test_1_batch_1_to_1_channel(gpu_device, slow);
    cout << "\n\n********** TEST test_1_batch_1_to_1_channel GPU FAST" << endl;
    test_1_batch_1_to_1_channel(gpu_device, slow);
    cout << "\n\n********** TEST test_1_batch_1_to_1_channel_3x3 GPU FAST" << endl;
    test_1_batch_1_to_1_channel_3x3(gpu_device, slow);
    cout << "\n\n********** TEST test_1_batch_1_to_2_channel_3x3 GPU FAST" << endl;
    test_1_batch_1_to_2_channel_3x3(gpu_device, slow);
    cout << "\n\n********** TEST test_1_batch_2_to_2_channel_3x3 GPU FAST" << endl;
    test_1_batch_2_to_2_channel_3x3(gpu_device, slow);
    cout << "\n\n********** TEST test_2_batch_2_to_2_channel_3x3 GPU FAST" << endl;
    test_2_batch_2_to_2_channel_3x3(gpu_device, slow);

    return 0;
}
