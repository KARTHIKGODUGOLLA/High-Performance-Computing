#include <cmath>
using std::fabs;

#include <iomanip>
using std::setprecision;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <vector>
using std::vector;

#include "som.hxx"
#include "test_common.hxx"

void compare_doubles(string name, double actual, double expected) {
    double diff = fabs(actual - expected);
    if (diff > 1e-10) {
        cerr << "distance from sample to " << name << " did not match, difference: " << setprecision(15) << diff << endl;
        cerr << "actual: " << setprecision(15) << actual << ", expected: " << setprecision(15) << expected << endl;
        exit(1);
    }
}

int main(int argc, char** argv) {
    //make a SOM that we won't really use
    SelfOrganizingMap *som = new SelfOrganizingMap({"a", "b"}, 3, 2, 2, 100);

    //hardcode the cell values just to be sure (for OS/compilers which have a different
    //mersenne twister and/or normal_distribution implementation)
    // som->print_cell_code();
    som->cells[0][0]->values = {1.73826037986052, -0.893258708396503, 0.0885711706115432};
    som->cells[0][1]->values = {-1.17970328773162, -0.398300045331517, -0.363775610896062};
    som->cells[1][0]->values = {-1.43973701249902, -0.234715169222093, 2.30447256451043};
    som->cells[1][1]->values = {0.432816580412273, 0.0154593309715393, -0.12263552044841};


    Sample *sample = new Sample("a,-5,-1,2.5");
    
    cout << sample << endl;

    double distance00 = som->distance_function(som->cells[0][0], sample);
    print_vector("cell[0][0]", som->cells[0][0]->values);
    cout << "cell [0][0] distance: " << setprecision(15) << distance00 << endl;
    compare_doubles("cell[0][0]", distance00, 7.15755095331755);

    double distance01 = som->distance_function(som->cells[0][1], sample);
    print_vector("cell[1][0]", som->cells[0][1]->values);
    cout << "cell [0][1] distance: " << setprecision(15) << distance01 << endl;
    compare_doubles("cell[0][1]", distance01, 4.81226771437124);

    double distance10 = som->distance_function(som->cells[1][0], sample);
    print_vector("cell[1][0]", som->cells[1][0]->values);
    cout << "cell [1][0] distance: " << setprecision(15) << distance10 << endl;
    compare_doubles("cell[1][0]", distance10, 3.64682936129692);

    double distance11 = som->distance_function(som->cells[1][1], sample);
    print_vector("cell[1][1]", som->cells[1][1]->values);
    cout << "cell [1][1] distance: " << setprecision(15) << distance11 << endl;
    compare_doubles("cell[1][1]", distance11, 6.11758700161897);


    return 0;
}
