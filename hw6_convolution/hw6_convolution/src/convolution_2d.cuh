#pragma once

#include "device.hxx"

#include <iostream>
using std::ostream;

#include <string>
using std::string;


class Convolution2D {
    public:
        //Used to determine if we are using this tensor on a CPU or
        //on a GPU
        Device device;

        //The y dimension of the convolution
        uint32_t y_size;

        //The x dimension of the convolution
        uint32_t x_size;

        //The padding to be used on the input tensor (around the x and y dimensions)
        uint32_t padding;

        //The input tensor to the convolution operation
        Tensor *input_tensor;

        //The output tensor to put results in
        Tensor *output_tensor;

        //The number of weights in each convolution, this will be (x_size * y_size).
        uint32_t n_convolution_weights;

        //The total number of tensor weights, which will be 
        //convolution_weights * (input_tensor.channels * output_tensor.channels)
        uint32_t n_weights;

        uint32_t batch_size; //both tensors should have the same batch size

        //Grab these so we can save on pointer dereferences later
        uint32_t in_channels;
        uint32_t in_y_size;
        uint32_t in_x_size;

        uint32_t out_channels;
        uint32_t out_y_size;
        uint32_t out_x_size;

        int32_t in_batch_width;
        int32_t in_channel_width;

        int32_t out_batch_width;
        int32_t out_channel_width;

        //The convolution weights (one convolution for each input to output 
        //channel combination).
        float *weights;

        //The backward pass errors for the weights to use in gradient descent, this
        //will be the same size as the number of weights.
        float *errors;

        /**
         * Creates a 2D convolutional layer between the input and output tensor. This
         * will create one convolution (of size x_size * y_size) between each input and
         * ouptut channel (i.e., input and output channels will be fully connected).
         *
         * \param device is used to know if we are going to use this tensor on a CPU
         *      or on a GPU
         * \param _y_size
         * \param _x_size
         * \param _padding
         * \param _input_tensor
         * \param _output_tensor
         */
        Convolution2D(const Device &_device, uint32_t _y_size, uint32_t _x_size, uint32_t _padding, Tensor *_input_tensor, Tensor *_output_tensor);

        /**
         * Destructor for the convolution class.  Deletes CPU or GPU memory depending on
         * where the convolution was made.
         */
        ~Convolution2D();

        /**
         * Performs a forward pass of this convolution from the input to the output tensor.
         */
        void forward();

        /**
         * Performs a forward pass of this convolution from the input to the output tensor. This uses
         * the slower assign method to set the values but we can use this to validate the fast
         * version of forward (and the GPU version of forward).
         */
        void forward_slow();

        /**
          * Gets a weight in this convolution operation.
          * \param in_channel is the channel the convoltion goes from in the input tensor
          * \param out_channel is the channel the convoltion goes to in the output tensor
          * \param y is the y index of the weight in the convolution
          * \param x is the x index of the weight in the convolution
          * \return the weight at the y and x position for the convolution going from
          *     in_channel to out_channel.
          */
        float get(int32_t in_channel, int32_t out_channel, int32_t y, int32_t x);

        /**
          * Sets a weight in this convolution operation.
          * \param in_channel is the channel the convoltion goes from in the input tensor
          * \param out_channel is the channel the convoltion goes to in the output tensor
          * \param y is the y index of the weight in the convolution
          * \param x is the x index of the weight in the convolution
          */
        void set(int32_t in_channel, int32_t out_channel, int32_t y, int32_t x, float value);

        /**
         * Performs a backward pass of this convolution from the output to the input
         * tensor to set the error values for gradient calculation.
         */
        void backward();

        /**
         * Resets the error values for the convolution for another forward/back ward pass.
         */

        void reset();

        /**
         * Prints all the values in the convolutional operation.
         */
        void print_weights();

        /**
         * Overloads the << operator so we can nicely print out the convolution class
         * to cout, cerr or any stream.
         */
        friend ostream& operator<<(ostream& stream, const Convolution2D& convolution);

        /**
         * Overloads the << operator so we can nicely print out the convolution class
         * to cout, cerr or any stream. This one will accept a pointer instead of
         * a non-pointer version of the class.
         */
        friend ostream& operator<<(ostream& stream, const Convolution2D *convolution);
};
