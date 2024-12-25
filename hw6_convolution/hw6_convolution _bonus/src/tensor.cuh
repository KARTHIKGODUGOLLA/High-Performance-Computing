#pragma once

#include "device.hxx"

#include <iostream>
using std::ostream;

#include <string>
using std::string;


class Tensor {
    public:
        //Used to determine if we are using this tensor on a CPU or
        //on a GPU
        Device device;

        //The first dimension of the tensor (batch size)
        uint32_t batch_size;

        //The second dimension of the tensor (number of channels)
        uint32_t channels;

        //The third dimension of the tensor (number of rows)
        uint32_t y_size;

        //The fourth dimension of the tensor (number of columns)
        uint32_t x_size;

        //how many values in a channel (y_size * x_size)
        int32_t channel_width;

        //how many values in a batch (channels * y_size * x_size)
        int32_t batch_width;

        //The total size of the tensor - batch_size * channels * y_size * x_size
        uint32_t size;

        float *values;
        float *errors;

        /**
         * Creates a 4D tensor for our convolutional neural network. See parameter
         * definitions above.
         *
         * \param device is used to know if we are going to use this tensor on a CPU
         *      or on a GPU
         * \param _batch_size
         * \param _channels
         * \param _y_size
         * \param _x_size
         */
        Tensor(const Device &_device, uint32_t _batch_size, uint32_t _channels, uint32_t _y_size, uint32_t _x_size);

        /**
         * Destructor for the tensor class.  Deletes CPU or GPU memory depending on
         * where the tensor was made.
         */
        ~Tensor();

        /**
          * Gets a value in this tensor.
          * \param batch is the batch the value is in
          * \param channel is the channel the value is in
          * \param y is the y index of the value
          * \param x is the x index of the value
          * \return the given value 
          */
        float get(int32_t batch, int32_t channel, int32_t y, int32_t x);

        /**
          * Sets a value in this tensor.
          * \param batch is the batch the value is in
          * \param channel is the channel the value is in
          * \param y is the y index of the value
          * \param x is the x index of the value
          * \param value is the value to be set
          */
        void set(int32_t batch, int32_t channel, int32_t y, int32_t x, float value);

        /**
          * Adds a value to a value in this tensor.
          * \param batch is the batch the value is in
          * \param channel is the channel the value is in
          * \param y is the y index of the value
          * \param x is the x index of the value
          * \param value is the value to add to the value in the tensor at the given position
          */
        void accumulate(int32_t batch, int32_t channel, int32_t y, int32_t x, float value);


        /**
         * Resets the values and errors to 0 for another forward and backward pass.
         */
        void reset();

        /**
         * Prints all the values in the tensor.
         */
        void print_values();

        /**
         * Overloads the << operator so we can nicely print out the tensor class
         * to cout, cerr or any stream.
         */
        friend ostream& operator<<(ostream& stream, const Tensor& tensor);

        /**
         * Overloads the << operator so we can nicely print out the tensor class
         * to cout, cerr or any stream. This one will accept a pointer instead of
         * a non-pointer version of the class.
         */
        friend ostream& operator<<(ostream& stream, const Tensor *tensor);
};
