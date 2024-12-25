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

int main(int argc, char** argv) {
    //make a SOM that we won't really use
    SelfOrganizingMap *som = new SelfOrganizingMap({"a", "b", "c"}, 4, 5, 5, 100);

    //hardcode the cell values just to be sure (for OS/compilers which have a different
    //mersenne twister and/or normal_distribution implementation)
    // som->print_cell_code();
    som->cells[0][0]->values = {1.73826037986052, -0.893258708396503, 0.0885711706115432, -1.17970328773162};
    som->cells[0][1]->values = {-0.398300045331517, -0.363775610896062, -1.43973701249902, -0.234715169222093};
    som->cells[0][2]->values = {2.30447256451043, 0.432816580412273, 0.0154593309715393, -0.12263552044841};
    som->cells[0][3]->values = {-0.113794672925254, 0.266016351285991, -1.44077119847131, 1.01106615039517};
    som->cells[0][4]->values = {1.37699633576283, 0.980569134835605, 0.480924128190207, -0.230285083936049};
    som->cells[1][0]->values = {0.271005111969217, -1.26691060368642, 0.475369035857103, 2.14076325027999};
    som->cells[1][1]->values = {-1.08417397072667, 0.24652490317828, -0.436348770302628, -2.38158262270916};
    som->cells[1][2]->values = {-1.29543982514288, -2.07241190952353, -0.057026634866721, -0.464023541851654};
    som->cells[1][3]->values = {1.49451523078073, 0.275929705649751, -2.303374273956, 0.566820563559807};
    som->cells[1][4]->values = {-1.18738682436331, 1.37700070725779, 0.107968584264031, 0.0793930236411706};
    som->cells[2][0]->values = {1.12558935916785, 0.506951109004468, 0.354829224512055, -1.00805247190325};
    som->cells[2][1]->values = {0.113451131215394, -0.702668180013166, -0.686599321052349, -1.0411337074394};
    som->cells[2][2]->values = {0.946310399640558, -2.55944766477786, 0.680001872955055, 0.790671631082328};
    som->cells[2][3]->values = {0.477704832257768, 0.716605809732281, 0.0844800708824679, -0.66623991572264};
    som->cells[2][4]->values = {0.705921160360947, -0.338760491236097, 0.393248072653692, -1.6017614218413};
    som->cells[3][0]->values = {1.8379774768721, -0.601657681736595, 0.421248419159312, -0.0199423727351812};
    som->cells[3][1]->values = {-0.216272352134686, -0.19530937841333, 0.00473325334647421, -0.688252870999717};
    som->cells[3][2]->values = {-0.200902683466444, 0.4246901819566, -2.26909895169173, -0.757833527215193};
    som->cells[3][3]->values = {0.194815239765472, -0.95657021932124, -1.34536714653089, -1.42373815367588};
    som->cells[3][4]->values = {0.60472385741705, -0.358771523010233, -0.559486321836126, 0.637805715580074};
    som->cells[4][0]->values = {-0.107882594037835, 0.0562009745327589, -0.415846919756975, 0.0889739385121715};
    som->cells[4][1]->values = {-1.87701390002209, 1.88400150674621, 1.5251802037426, -0.866172508790739};
    som->cells[4][2]->values = {-0.0527620854626043, 0.387493455159739, -0.886748410636903, 0.112464210496467};
    som->cells[4][3]->values = {1.99693122132596, -0.83371603888623, 0.95965712334584, 0.679220878044328};
    som->cells[4][4]->values = {1.80954954457325, -0.0264569750482067, 0.373661077298714, 0.59910474929387};

    Sample *sample = new Sample("a,-5,-1,2.5,3.2");
    cout << sample << endl;

    CellIndex bmu_index = som->get_best_matching_unit(sample);
    cout << "bmu index: " << bmu_index << endl;

    CellIndex expected_index(1, 0);
    if (bmu_index != expected_index) {
        cerr << "error, BMU index " << bmu_index << " did not match expected index: " << expected_index << endl;
        exit(1);
    }
    delete sample;

    sample = new Sample("a,0.5,0.3,-2.5,1.2");
    cout << sample << endl;

    bmu_index = som->get_best_matching_unit(sample);
    cout << "bmu index: " << bmu_index << endl;

    expected_index = CellIndex(1, 3);
    if (bmu_index != expected_index) {
        cerr << "error, BMU index " << bmu_index << " did not match expected index: " << expected_index << endl;
        exit(1);
    }
    delete sample;

    delete som;

    return 0;
}
