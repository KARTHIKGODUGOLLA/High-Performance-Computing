#ifndef TEST_COMMON_HXX
#define TEST_COMMON_HXX

#include <string>
using std::string;

#include <vector>
using std::vector;

void print_vector(string name, const vector<double> &v);

void compare_vectors(string name, const vector<double> &actual, const vector<double> &expected);

#endif
