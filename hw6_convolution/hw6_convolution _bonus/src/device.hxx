#pragma once

#include <cstdint>

#include <iostream>
using std::ostream;


class Device {
    public:
        //Specifies which device is being used, -1 for CPU otherwise GPUs
        int32_t device_number;

        /**
         * Creates a class which specifies which device we are using.
         * \param _device_number is which device is being used.
         */
        Device(int32_t _device_number);

        /**
         * \return true if this device is specfiying a CPU.
         */
        bool is_cpu() const;

        /**
         * \return true if this device is specfiying a GPU.
         */
        bool is_gpu() const;

        /**
         * \return which GPU is being used, or -1 for CPU.
         */
        int32_t get_gpu() const;
        /**
         * Overloads the << operator so we can nicely print out the device class
         * to cout, cerr or any stream.
         */
        friend ostream& operator<<(ostream& stream, const Device& device);
};
