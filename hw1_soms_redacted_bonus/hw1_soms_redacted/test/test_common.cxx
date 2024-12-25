#include <cmath>
using std::fabs;

#include <iomanip>
using std::setprecision;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <string>
using std::string;

#include <vector>
using std::vector;

void print_vector(string name, const vector<double> &v) {
    cout << name << ": ";
    bool first = true;
    for (double value : v) {
        if (first) {
            first = false;
        } else {
            cout << ", ";
        }
        cout << setprecision(15) << value;
    }
    cout << endl;

}

void compare_vectors(string name, const vector<double> &actual, const vector<double> &expected) {

    if (actual.size() != expected.size()) {
        cerr << name << " size (" << actual.size() << ") was not the expected size: " << expected.size() << endl;
        exit(1);
    }

    bool mismatch = false;
    for (int32_t i = 0; i < actual.size(); i++) {
        double diff = actual[i] - expected[i];
        if (fabs(diff) > 1e-10) {
            cerr << name << " actual[" << i << "]: " << actual[i] << " did not match expected[" << i << "]: " << expected[i] << ", difference: " << diff << endl;
            mismatch = true;
        }
    }

    if (mismatch) {
        cerr << name << " vectors did not match" << endl;
        exit(1);
    }
}

