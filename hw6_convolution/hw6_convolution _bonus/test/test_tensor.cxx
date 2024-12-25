#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include "device.hxx"
#include "tensor.cuh"

int main(int argc, char** argv) {
    //try first on the CPU
    Device device(-1);
    cout << "device is: " << device << endl;

    if (!device.is_cpu()) {
        cerr << "ERROR: device should have been CPU." << endl;
        exit(1);
    }

    Tensor *tensor = new Tensor(device, 10, 5, 5, 5);

    cout << tensor << endl;

    if (tensor->size != 10 * 5 * 5 * 5) {
        cerr << "ERROR: tensor size was incorrect, was " << tensor->size << " and should have been: " << (10 * 5 * 5 * 5) << endl;
        exit(1);
    }

    delete tensor;

    //now try on the GPU

    device = Device(0);
    cout << "device is: " << device << endl;

    tensor = new Tensor(device, 10, 5, 5, 5);

    cout << tensor << endl;

    if (tensor->size != 10 * 5 * 5 * 5) {
        cerr << "ERROR: tensor size was incorrect, was " << tensor->size << " and should have been: " << (10 * 5 * 5 * 5) << endl;
        exit(1);
    }

    delete tensor;

    return 0;
}
