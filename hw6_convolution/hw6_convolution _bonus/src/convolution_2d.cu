#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h> //for printf in cuda kernels


#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include "device.hxx"
#include "exceptions.hxx"
#include "tensor.cuh"
#include "convolution_2d.cuh"


__global__ void gpu__convolve_2d(
    int16_t n_convolution_weights,
    int16_t out_channels,
    int16_t in_batch_width,
    int16_t in_channel_width,
    int16_t in_y_size,
    int16_t in_x_size,
    int16_t out_batch_width,
    int16_t out_channel_width,
    int16_t out_y_size,
    int16_t out_x_size,
    int16_t conv_y_size,
    int16_t conv_x_size,
    int16_t padding,
    float *input_values,
    float *output_values,
    float *weights
) {
    int16_t input_channel = blockIdx.x;
    int16_t batch = threadIdx.x;
    int16_t output_channel = threadIdx.y;
    //printf("running kernel for input channel: %d, batch: %d and output channel: %d\n", input_channel, batch, output_channel);

    // the only way to pass in a dynamic shared memory is to use
    // an extern __shared__ array like this
    extern __shared__ float shared_memory[];

    //get pointers to the N values in shared memory (used for the weights) for this input to
    //output convolution
    //and the next M values in shared memory (used for the input channel)
    float *convolve_shared_weights = shared_memory;
    float *shared_input_values = shared_memory + (out_channels * n_convolution_weights);

     // Load weights into shared memory
    for (int i = batch; i < n_convolution_weights; i += blockDim.x) {
        convolve_shared_weights[output_channel * n_convolution_weights + i] =
            weights[input_channel * out_channels * n_convolution_weights +
                    output_channel * n_convolution_weights + i];
    }

    // Load input values into shared memory
    for (int i = threadIdx.x; i < in_channel_width; i += blockDim.x) {
        shared_input_values[batch * in_channel_width + i] =
            input_values[batch * in_batch_width + input_channel * in_channel_width + i];
    }

    //make sure you sync the threads after loading the shared memory
    __syncthreads();


    //TODO: do the convolution operation here for the given batch, input channel and output channel
    //when summing up to the output channel make sure you use CUDA's atomicAdd function to prevent
    //race conditions when adding to the output channel's values
     for (int out_y = 0; out_y < out_y_size; ++out_y) {
        for (int out_x = 0; out_x < out_x_size; ++out_x) {
            float sum = 0.0f;

            for (int conv_y = 0; conv_y < conv_y_size; ++conv_y) {
                for (int conv_x = 0; conv_x < conv_x_size; ++conv_x) {
                    int in_y = out_y + conv_y - padding;
                    int in_x = out_x + conv_x - padding;

                    // Ensure input coordinates are valid
                    if (in_y >= 0 && in_y < in_y_size && in_x >= 0 && in_x < in_x_size) {
                        int input_idx = batch * in_channel_width + in_y * in_x_size + in_x;
                        int weight_idx = output_channel * n_convolution_weights +
                                         conv_y * conv_x_size + conv_x;

                        sum += shared_input_values[input_idx] * convolve_shared_weights[weight_idx];
                    }
                }
            }
            int output_idx = batch * out_batch_width + output_channel * out_channel_width +
                             out_y * out_x_size + out_x;
            atomicAdd(&output_values[output_idx], sum);

        }
    }
}


__global__ void gpu__convolve_2d_slow(
    int16_t n_convolution_weights,
    int16_t out_channels,
    int16_t in_batch_width,
    int16_t in_channel_width,
    int16_t in_y_size,
    int16_t in_x_size,
    int16_t out_batch_width,
    int16_t out_channel_width,
    int16_t out_y_size,
    int16_t out_x_size,
    int16_t conv_y_size,
    int16_t conv_x_size,
    int16_t padding,
    float *input_values,
    float *output_values,
    float *weights
) {
    int16_t input_channel = blockIdx.x;
    int16_t batch = threadIdx.x;
    int16_t output_channel = threadIdx.y;
    //printf("running kernel for input channel: %d, batch: %d and output channel: %d\n", input_channel, batch, output_channel);

    //TODO: implement the convolution operation just using global memory
    //when summing up to the output channel make sure you use CUDA's atomicAdd function to prevent
    //race conditions when adding to the output channel's values
    //convolution operation
    for (int16_t out_y = 0; out_y < out_y_size; ++out_y) {
        for (int16_t out_x = 0; out_x < out_x_size; ++out_x) {
            float sum = 0.0;

            for (int16_t conv_y = 0; conv_y < conv_y_size; ++conv_y) {
                for (int16_t conv_x = 0; conv_x < conv_x_size; ++conv_x) {
                    int16_t in_y = out_y + conv_y - padding;
                    int16_t in_x = out_x + conv_x - padding;

                    if (in_y >= 0 && in_y < in_y_size && in_x >= 0 && in_x < in_x_size) {
                        int input_idx = (batch * in_batch_width) +
                                        (input_channel * in_channel_width) +
                                        (in_y * in_x_size) + in_x;

                        int weight_idx = (input_channel * out_channels * n_convolution_weights) +
                                         (output_channel * n_convolution_weights) +
                                         (conv_y * conv_x_size) + conv_x;

                        sum += input_values[input_idx] * weights[weight_idx];
		    }
		}
	    }
            int output_idx = (batch * out_batch_width) +
                             (output_channel * out_channel_width) +
                             (out_y * out_x_size) + out_x;

            atomicAdd(&output_values[output_idx], sum);
        }
    }
}



Convolution2D::Convolution2D(const Device &_device, uint32_t _y_size, uint32_t _x_size, uint32_t _padding, Tensor *_input_tensor, Tensor *_output_tensor) : device(_device), y_size(_y_size), x_size(_x_size), padding(_padding), input_tensor(_input_tensor), output_tensor(_output_tensor) {

    //first verify that the size of the convolution and padding will appropriately work
    //between the input and output tensors

    int32_t expected_output_x = (input_tensor->x_size - x_size) + 1 + (padding * 2);
    int32_t expected_output_y = (input_tensor->y_size - y_size) + 1 + (padding * 2);

    if (expected_output_x != output_tensor->x_size || expected_output_y != output_tensor->y_size) {
        throw INVALID_CONVOLUTION_SIZE;
    }

    if (input_tensor->batch_size != output_tensor->batch_size) {
        throw BATCH_SIZE_MISMATCH;
    }

    n_convolution_weights = x_size * y_size;
    n_weights = n_convolution_weights * input_tensor->channels * output_tensor->channels;

    //grab these once to save on pointer dereferences
    batch_size = input_tensor->batch_size; //both tensors should have the same batch size
    in_channels = input_tensor->channels;
    in_y_size = input_tensor->y_size;
    in_x_size = input_tensor->x_size;

    out_channels = output_tensor->channels;
    out_y_size = output_tensor->y_size;
    out_x_size = output_tensor->x_size;

    //precompute widths so we can re-use these without extra compute
    in_batch_width = in_channels * in_y_size * in_x_size;
    in_channel_width = in_y_size * in_x_size;

    out_batch_width = out_channels * out_y_size * out_x_size;
    out_channel_width = out_y_size * out_x_size;

    if (device.is_cpu()) {
        weights = new float[n_weights];
        errors = new float[n_weights];
    } else {
        //allocate the memory on the GPU
        cudaSetDevice(device.get_gpu());
        cudaMallocManaged(&weights, n_weights * sizeof(float));
        cudaMallocManaged(&errors, n_weights * sizeof(float));
    }
}


Convolution2D::~Convolution2D() {
    if (device.is_cpu()) {
        delete weights;
        delete errors;
    } else {
        cudaFree(weights);
        cudaFree(errors);
    }
}

void Convolution2D::forward() {
    if (device.is_cpu()) {
        uint32_t weight_offset = 0;

        float sum;

        int32_t conv_weight_offset; //used to go over the same set of weights for a convolution repeatedly
        int32_t current_in_x;
        int32_t current_in_y;

        //TODO: implement the convolution *WITHOUT* using the get/set/accumulate methods

        //iterate over every pair of input to output channels, there will be one
        //convolution between each. i.e., the input channels are fully connected
        //to the output channels.
        for (uint32_t input_channel = 0; input_channel < in_channels; input_channel++) {
            for (uint32_t output_channel = 0; output_channel < out_channels; output_channel++) {

                //instead of doing the batch first, each input to output channel will
                //use the same weights across the batch, so we can keep them in memory
                //better by iterating by batch after determining the input to output
                //channel
                for (uint32_t batch = 0; batch < batch_size; batch++) {
                    //TODO: calculate input and output offsets for the input and output channel values
                    uint32_t input_offset = batch * in_batch_width + input_channel * in_channel_width;
                    uint32_t output_offset = batch * out_batch_width + output_channel * out_channel_width;
                    //iterate over the input tensor's channel, we use the output
                    //sizes because if the input tensor is larger than the output
                    //the padding and convolution will make sure we hit every
                    //input cell
                    for (int32_t out_y = 0; out_y < out_y_size; out_y++) {
                        for (int32_t out_x = 0; out_x < out_x_size; out_x++) {

                            //TODO: do the convolution here
                            float sum = 0.0;
                            for (int32_t conv_y = 0; conv_y < y_size; conv_y++) {
                                for (int32_t conv_x = 0; conv_x < x_size; conv_x++) {
                                    // Compute input coordinates considering padding
                                    int32_t in_y = out_y + conv_y - padding;
                                    int32_t in_x = out_x + conv_x - padding;
                                    // make sure the input x and y values are actually inside the tensor                                    
                                    if (in_y >= 0 && in_y < in_y_size && in_x >= 0 && in_x < in_x_size) {
                                        float input_value = input_tensor->values[input_offset + in_y * in_x_size + in_x];

                                        uint32_t weight_index = (input_channel * out_channels + output_channel) * n_convolution_weights +
                                                                conv_y * x_size + conv_x;
                                        float weight_value = weights[weight_index];
                                        sum += input_value * weight_value;
                                    }
                                }
                            }
                            // Add the calculated sum to the output tensor's initial value
                            output_tensor->values[output_offset + out_y * out_x_size + out_x] += sum;
                        }
                    }

                }
            }
        }

    } else {
        //launch the CUDA kernel to do the forward pass using
        //shared memory

        //we will have one block per input channel
        int32_t number_blocks = in_channels;
        //in each block we will have out_channels (y dimension) threads per batch (x dimension)
        dim3 threads_per_block(batch_size, out_channels);
        //this will let each block share the weights across the batch for the convolution to each 
        //output channel for the input channel, and also have each block share the input channel's
        //values

        //this will allocate the shared memory that we will use to save the weights and each input channel
        //we will have the convolution weights for each output channel plus the input channel values for
        //the whole batch.
        int32_t shared_memory_size = ((n_convolution_weights * out_channels) + (batch_size * input_tensor->channel_width)) * sizeof(float);

        cout << "starting FAST GPU 2D convolution with number blocks: " << number_blocks << " and threads per block.x: " << threads_per_block.x << ", y: " << threads_per_block.y << " and z: " << threads_per_block.z << endl;
        cout << "batch_size: " << batch_size << ", out_channels: " << out_channels << ", shared memory size (n floats): " << (shared_memory_size / sizeof(float)) << endl;

        gpu__convolve_2d<<<number_blocks, threads_per_block, shared_memory_size>>>(
            n_convolution_weights,
            out_channels,
            in_batch_width,
            in_channel_width,
            in_y_size,
            in_x_size,
            out_batch_width,
            out_channel_width,
            out_y_size,
            out_x_size,
            y_size,
            x_size,
            padding,
            input_tensor->values,
            output_tensor->values,
            weights
        );
        cudaDeviceSynchronize();

        cudaError_t code = cudaGetLastError();
        if (code != cudaSuccess) {   
            cerr << "GPUassert: " << cudaGetErrorString(code) << endl;
            exit(1);
        }   

    }
}

void Convolution2D::forward_slow() {
    if (device.is_cpu()) {
        float sum;

        int32_t current_in_x;
        int32_t current_in_y;

        //iterate over every pair of input to output channels, there will be one
        //convolution between each. i.e., the input channels are fully connected
        //to the output channels.
        for (uint32_t input_channel = 0; input_channel < in_channels; input_channel++) {
            for (uint32_t output_channel = 0; output_channel < out_channels; output_channel++) {

                //instead of doing the batch first, each input to output channel will
                //use the same weights across the batch, so we can keep them in memory
                //better by iterating by batch after determining the input to output
                //channel
                for (uint32_t batch = 0; batch < batch_size; batch++) {
                    //iterate over the input tensor's channel, we use the output
                    //sizes because if the input tensor is larger than the output
                    //the padding and convolution will make sure we hit every
                    //input cell
                    for (int32_t in_y = 0; in_y < out_y_size; in_y++) {
                        for (int32_t in_x = 0; in_x < out_x_size; in_x++) {

                            //reset the weight offset so we can
                            //iterate over the same set of weights
                            //for each convolution
                            sum = 0.0;
                            for (int32_t conv_y = 0; conv_y < y_size; conv_y++) {
                                for (int32_t conv_x = 0; conv_x < x_size; conv_x++) {
                                    current_in_y = in_y + conv_y - padding;
                                    current_in_x = in_x + conv_x - padding;
                                    
                                    // make sure the input x and y values are actually inside
                                    // the tensor, if they are outside they refer to a padding
                                    // cell
                                    if (current_in_y >= 0 && current_in_y < in_y_size && current_in_x >= 0 && current_in_x < in_x_size) {
                                        sum += input_tensor->get(batch, input_channel, current_in_y, current_in_x) * get(input_channel, output_channel, conv_y, conv_x);
                                    }
                                }
                            }

                            //cout << "setting output_tensor->values[" << batch << "][" << output_channel << "][" << in_y << "][" << in_x << "] += " << sum << endl;

                            //the starting x and y indexes of the input tensor are where
                            //we'll set the output tensor values
                            output_tensor->accumulate(batch, output_channel, in_y, in_x, sum);
                        }
                    }

                }
            }
        }
    } else {
        // run the GPU kernel without using shared memory

        //we will have one block per input channel
        int32_t number_blocks = in_channels;
        //in each block we will have in_channels (y dimension) threads per batch (x dimension)
        dim3 threads_per_block(batch_size, out_channels);

        cout << "starting SLOW GPU 2D convolution with number blocks: " << number_blocks << " and threads per block.x: " << threads_per_block.x << ", y: " << threads_per_block.y << " and z: " << threads_per_block.z << endl;

        cout << "batch_size: " << batch_size << ", out_channels: " << out_channels << endl;

        gpu__convolve_2d_slow<<<number_blocks, threads_per_block>>>(
            n_convolution_weights,
            out_channels,
            in_batch_width,
            in_channel_width,
            in_y_size,
            in_x_size,
            out_batch_width,
            out_channel_width,
            out_y_size,
            out_x_size,
            y_size,
            x_size,
            padding,
            input_tensor->values,
            output_tensor->values,
            weights
        );
        cudaDeviceSynchronize();

        cudaError_t code = cudaGetLastError();
        if (code != cudaSuccess) {   
            cerr << "GPUassert: " << cudaGetErrorString(code) << endl;
            exit(1);
        }   

    }
}


float Convolution2D::get(int32_t in_channel, int32_t out_channel, int32_t y, int32_t x) {
    //there will be out_channels convolutions, each with n_convolution_weights weights
    //for every input channel.
    int32_t channel_width = out_channels * n_convolution_weights;

    return weights[(in_channel * channel_width) + (out_channel * n_convolution_weights) + (y * x_size) + x];
}

void Convolution2D::set(int32_t in_channel, int32_t out_channel, int32_t y, int32_t x, float value) {
    //there will be out_channels convolutions, each with n_convolution_weights weights
    //for every input channel.
    int32_t channel_width = out_channels * n_convolution_weights;

    weights[(in_channel * channel_width) + (out_channel * n_convolution_weights) + (y * x_size) + x] = value;
}


void Convolution2D::backward() {
    if (device.is_cpu()) {

    } else {
        //launch the CUDA kernel to do the backward pass

    }
}

void Convolution2D::reset() {
    if (device.is_cpu()) {
        memset(errors, 0, n_weights * sizeof(float));
    } else {
        cudaMemset(errors, 0, n_weights * sizeof(float));
    }
}

void Convolution2D::print_weights() {
    int32_t counter = 0;
    for (int32_t in_channel = 0; in_channel < in_channels; in_channel++) {
        cout << "INPUT CHANNEL: " << in_channel << endl;
        for (int32_t out_channel = 0; out_channel < out_channels; out_channel++) {
            cout << "\tINPUT CHANNEL:" << in_channel << " to OUTPUT CHANNEL: " << out_channel << endl;
            for (int32_t y = 0; y < y_size; y++) {
                cout << "\t\t[";
                for (int32_t x = 0; x < x_size; x++) {
                    float value1 = get(in_channel, out_channel, y, x);
                    float value2 = weights[counter];

                    if (value1 != value2) {
                        cerr << "\n\nERROR: get and increment did not return same weight!" << endl;
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


ostream& operator<<(ostream& stream, const Convolution2D& convolution) {
    stream << "[convolution " << convolution.x_size << " x " << convolution.y_size << " | input channels: " << convolution.input_tensor->channels << ", output channels: "  << convolution.output_tensor->channels << ", n weights: " << convolution.n_weights << "]";

    return stream;
}

ostream& operator<<(ostream& stream, const Convolution2D *convolution) {
    // this will call the non-pointer version by dereferencing
    // the pointer to convolution
    stream << *convolution;
    return stream;
}

