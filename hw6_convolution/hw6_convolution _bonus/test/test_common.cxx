#include <cmath>
using std::fabs;

#include <iostream>
using std::cout;
using std::endl;

#include "test_common.hxx"

bool close_enough(float f1, float f2) {
    float diff = fabs(f1 - f2);
    if (diff >= CLOSE_ENOUGH_LIMIT) {
        cout << "diff: " << diff << endl;
        return false;
    }
    return true;
}



bool close_enough_fancy(float f1, float f2) {
    // handle edge cases when one of the values is 0
    if (f1 == 0) return fabs(f2) < CLOSE_ENOUGH_LIMIT;
    if (f2 == 0) return fabs(f1) < CLOSE_ENOUGH_LIMIT;

    float f1_abs = fabs(f1);
    float f2_abs = fabs(f2);
    
    //calculate relative similarity
    float diff;
    if (f1_abs > f2_abs) {
        diff = (fabs(f1 - f2) / f1_abs);
    } else {
        diff = (fabs(f1 - f2) / f2_abs);
    }

    if (diff >= CLOSE_ENOUGH_LIMIT) {
        cout << "diff: " << diff << endl;
        return false;
    }
    return true;
}
