#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <vector>
using std::vector;

#include "sample.hxx"
#include "test_common.hxx"

int main(int argc, char** argv) {
    string line = "some label,5,10,-2,3.0";

    Sample sample(line);
    print_vector("simple line", sample.values);

    vector<double> expected{5, 10, -2, 3.0};
    compare_vectors("simple line", sample.values, expected);

    string expected_label = "some label";
    if (expected_label != sample.label) {
        cerr << "error, sample label was '" << sample.label << "' and it should have been '" << expected_label << "'" << endl;
        exit(1);
    }

    return 0;
}
