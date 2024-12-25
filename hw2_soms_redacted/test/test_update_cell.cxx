#include <algorithm>
using std::find;

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

void compare_indexes(const vector<CellIndex> &actual, const vector<CellIndex> &expected) {
    if (actual.size() != expected.size()) {
        cerr << "error, number of neighbor indexes: " << actual.size() << " did not match the expected number: " << expected.size() << endl;
        exit(1);
    }

    for (CellIndex i : expected) {
        if (find(actual.begin(), actual.end(), i) == actual.end()) {
            cerr << "error, expected cell index " << i << " was not found in the computed neighbor indexes" << endl;
            exit(1);
        }
    }
}

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

    vector<CellIndex> neighbor_indexes;

    cout << endl << endl << "testing neighbor radius 1" << endl;

    int32_t neighbor_radius = 1;
    som->get_neighbors(bmu_index, neighbor_radius, neighbor_indexes);

    for (CellIndex neighbor_index : neighbor_indexes) {
        cout << "neighbor_index: " << neighbor_index << endl;
    }

    //test updating the BMU
    CellIndex cell_index = CellIndex(1, 0);
    som->update_cell(sample, bmu_index, cell_index, 0.3);
    Cell *cell = som->cells[cell_index.y][cell_index.x];
    cout << cell << endl;
    if (cell->label_counts["a"] != 1) {
        cerr << "error, cell label counts for 'a' should be 1" << endl;
        exit(1);
    }

    if (cell->label_counts["b"] != 0) {
        cerr << "error, cell label counts for 'a' should be 0" << endl;
        exit(1);
    }

    if (cell->label_counts["c"] != 0) {
        cerr << "error, cell label counts for 'c' should be 0" << endl;
        exit(1);
    }
    print_vector("bmu cell 1", cell->values);
    compare_vectors("bmu cell 1", cell->values, {-1.31029642162155, -1.1868374225805, 1.08275832509997, 2.458534275196});


    //make sure update label counts is working
    sample->label = "b";
    som->update_cell(sample, bmu_index, cell_index, 0.3);
    cout << cell << endl;
    if (cell->label_counts["a"] != 1) {
        cerr << "error, cell label counts for 'a' should be 1" << endl;
        exit(1);
    }

    if (cell->label_counts["b"] != 1) {
        cerr << "error, cell label counts for 'a' should be 1" << endl;
        exit(1);
    }

    if (cell->label_counts["c"] != 0) {
        cerr << "error, cell label counts for 'c' should be 0" << endl;
        exit(1);
    }
    print_vector("bmu cell 2", cell->values);
    compare_vectors("bmu cell 2", cell->values, {-2.41720749513508, -1.13078619580635, 1.50793082756998, 2.6809739926372});


    sample->label = "b";
    som->update_cell(sample, bmu_index, cell_index, 0.3);
    cout << cell << endl;
    if (cell->label_counts["a"] != 1) {
        cerr << "error, cell label counts for 'a' should be 1" << endl;
        exit(1);
    }

    if (cell->label_counts["b"] != 2) {
        cerr << "error, cell label counts for 'a' should be 1" << endl;
        exit(1);
    }

    if (cell->label_counts["c"] != 0) {
        cerr << "error, cell label counts for 'c' should be 0" << endl;
        exit(1);
    }
    print_vector("bmu cell 3", cell->values);
    compare_vectors("bmu cell 3", cell->values, {-3.19204524659456, -1.09155033706444, 1.80555157929899, 2.83668179484604});


    cell_index = CellIndex(0, 0);
    som->update_cell(sample, bmu_index, cell_index, 0.1);
    cell = som->cells[cell_index.y][cell_index.x];
    cout << cell << endl;
    if (cell->label_counts["a"] != 0) {
        cerr << "error, cell label counts for 'a' should be 1" << endl;
        exit(1);
    }

    if (cell->label_counts["b"] != 0) {
        cerr << "error, cell label counts for 'a' should be 1" << endl;
        exit(1);
    }

    if (cell->label_counts["c"] != 0) {
        cerr << "error, cell label counts for 'c' should be 0" << endl;
        exit(1);
    }
    print_vector("cell 2", cell->values);
    compare_vectors("cell 2", cell->values, {1.40134736086749, -0.898595772976678, 0.209142612080966, -0.960718123345034});


    cell_index = CellIndex(2, 1);
    som->update_cell(sample, bmu_index, cell_index, 0.7);
    cell = som->cells[cell_index.y][cell_index.x];
    cout << cell << endl;
    if (cell->label_counts["a"] != 0) {
        cerr << "error, cell label counts for 'a' should be 1" << endl;
        exit(1);
    }

    if (cell->label_counts["b"] != 0) {
        cerr << "error, cell label counts for 'a' should be 1" << endl;
        exit(1);
    }

    if (cell->label_counts["c"] != 0) {
        cerr << "error, cell label counts for 'c' should be 0" << endl;
        exit(1);
    }
    print_vector("cell 3", cell->values);
    compare_vectors("cell 3", cell->values, {-0.7814028167473, -0.754701248510862, -0.128944439868188, -0.298935308637509});


    delete sample;
    delete som;

    return 0;
}
