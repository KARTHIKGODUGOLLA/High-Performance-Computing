#include <cmath>
using std::abs;
using std::fabs;
using std::pow;
using std::sqrt;

#include <iostream>
using std::cout;
using std::endl;

#include <random>
using std::mt19937;
using std::normal_distribution;
using std::random_device;

#include <utility>
using std::pair;

#include <vector>
using std::vector;

#include "dataset.hxx"
#include "cell.hxx"
#include "som.hxx"
#include "mutex"
#include "thread"

CellIndex::CellIndex(): x(-1), y(-1) {
}

CellIndex::CellIndex(int32_t _y, int32_t _x): y(_y), x(_x) {
}

bool CellIndex::operator==(const CellIndex& other) const {
    return y == other.y && x == other.x;
}

ostream& operator<<(ostream& stream, const CellIndex& cell_index) {
    stream << "[y: " << cell_index.y << ", x: " << cell_index.x << "]";
    return stream;
}


SelfOrganizingMap::SelfOrganizingMap(vector<string> _labels, uint32_t _sample_size, uint32_t _height, uint32_t _width, int32_t seed) : labels(_labels), sample_size(_sample_size), width(_height), height(_width) {
    random_number_generator = mt19937(seed);

    initialize_cells();
}

SelfOrganizingMap::SelfOrganizingMap(vector<string> _labels, uint32_t _sample_size, uint32_t _height, uint32_t _width) : labels(_labels), sample_size(_sample_size), height(_height), width(_width) {
    random_device rd;
    random_number_generator = mt19937(rd());

    initialize_cells();
}


void SelfOrganizingMap::initialize_cells() {
    // initialize all the cells in the 2D vectors to NULL
    cells = vector< vector<Cell*> >(height, vector<Cell*>(width, NULL));

    normal_distribution<double> distribution(0.0, 1.0);

    // set each cell such that its values are generated randomly from
    // a normal distribution with mean = 0 and standard deviation = 1
    // (re)set all the label counts for each cell to 0.
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            cells[y][x] = new Cell(sample_size, random_number_generator, distribution);
            cells[y][x]->reset_label_counts(labels);
            cout << "cell[" << y << "][" << x << "]: " << cells[y][x] << endl;
        }
    }
}

SelfOrganizingMap::~SelfOrganizingMap() {
    /**********
     * TODO: Implement the destructor
     **********/
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            delete cells[y][x];
        }
    }
    /********** ENDTODO **********/
}


double SelfOrganizingMap::distance_function(const Cell *cell, const Sample *sample) {
    double euclidianDistance = 0.0;

    /**********
     * TODO: Implement the Euclidian distance function between the cell and
     * sample values.
     **********/
    for (uint32_t i = 0; i < sample->size(); i++) {
        euclidianDistance += pow(cell->values[i] - (*sample)[i], 2); // Euclidean distance
    }
    euclidianDistance=sqrt(euclidianDistance);
    /********** ENDTODO **********/
    return euclidianDistance;
}

std::mutex mutx;
CellIndex SelfOrganizingMap::get_best_matching_unit(const Sample *sample) {
    //TODO: Homework 2 - Perform this loop concurrently where
    //a thread will be created for each cell in the SOM so each
    //distance can be calcualted simultaneously.

    CellIndex best_cell_index;
    vector<vector<double>> cell_distances(height, vector<double>(width, std::numeric_limits<double>::max()));
     auto compute_distance = [&](uint32_t y, uint32_t x) {
        double distance = distance_function(cells[y][x], sample);
        //using mutex to safely upda the shared distaance
        std::lock_guard<std::mutex> lock(mutx);
        cell_distances[y][x] = distance;
    };
    vector<std::thread> threads;
    for (uint32_t row = 0; row < height; row++) {
        for (uint32_t col = 0; col < width; col++) {
            threads.emplace_back(compute_distance, row, col);
        }
    }
    //waiting for all threds to finish execution
    for (auto &thread : threads) {
        thread.join();
    }
    //finding the BMU by checking the smallest distance cell to the sample
    double minimum_distance = std::numeric_limits<double>::max();
    for (uint32_t row = 0; row < height; row++) {
        for (uint32_t col = 0; col < width; col++) {
            double current_distance = cell_distances[row][col];
            if (current_distance < minimum_distance) {
                minimum_distance = current_distance;
                best_cell_index = CellIndex(row, col);
            }
        }
    }

    /**********
     * TODO: Homework 1 - implement getting the best matching unit sequentially
     * (using for loops instead of threads).
     **********/
    // double min_distance = std::numeric_limits<double>::max();
    // for (uint32_t y = 0; y < height; y++) {
    //     for (uint32_t x = 0; x < width; x++) {
    //         //calculating the distance between cell and the sample
    //         double distance = distance_function(cells[y][x], sample);
    //         //Update the distance if we find the closer cell
    //         if (distance < min_distance) { 
    //             min_distance = distance;
    //             best_cell_index = CellIndex(y, x);
    //         }
    //     }
    // }
    /********** ENDTODO **********/
    return best_cell_index;
}

double SelfOrganizingMap::neighborhood_function(const CellIndex &index1, const CellIndex &index2) {
    double theta = 0.0;
    /**********
     * TODO: Homework 1 - implement the SOM neighborhood function (theta).
     **********/
     int distance = abs(index1.y - index2.y) + abs(index1.x - index2.x);
     theta = 1.0 / pow(2, distance);
    return theta;
}

void SelfOrganizingMap::get_neighbors(const CellIndex &bmu_index, int32_t neighbor_radius, vector<CellIndex> &neighbor_indexes) {
    /**********
     * TODO: Homework 1 - implement getting all the neighbor indexes of the BMU
     **********/
    neighbor_indexes.clear();    
    if (neighbor_radius == -1) {
        for (int row = 0; row < height; ++row) { //adding ll the cell indexs to the neighbors list
            for (int col = 0; col < width; ++col) {
                neighbor_indexes.push_back({row, col});
            }
        }
        return;
    }
    for (int32_t dy = -neighbor_radius; dy <= neighbor_radius; dy++) {
        for (int32_t dx = -neighbor_radius; dx <= neighbor_radius; dx++) {
            int32_t neighborY = bmu_index.y + dy;
            int32_t neighborX = bmu_index.x + dx;
            // checking if the neighbor cordinates are withing the bounds
            if (neighborY >= 0 && neighborY < height && neighborX >= 0 && neighborX < width) {
                neighbor_indexes.push_back(CellIndex(neighborY, neighborX));
            }
        }
    }
    /********** ENDTODO **********/
}


void SelfOrganizingMap::update_cell(const Sample *sample, const CellIndex &bmu_index, const CellIndex &cell_index, double learning_rate) {
    /**********
     * TODO: Homework 1 - implement the SOM cell update process
     **********/
    Cell *cell = cells[cell_index.y][cell_index.x];

    double theta = neighborhood_function(bmu_index, cell_index);

    for (uint32_t i = 0; i < cell->values.size(); i++) {
        // Updating the values with theta from the neighborhood function
        cell->values[i] += theta * learning_rate * ((*sample)[i] - cell->values[i]);
    }

    if (cell_index == bmu_index) {//updating the label counts
        cell->label_counts[sample->label]++;
    }
    /********** ENDTODO **********/
}


void SelfOrganizingMap::train(Dataset *dataset, uint32_t epochs, int32_t neighbor_radius, double learning_rate, double learning_rate_schedule) {

    CellIndex bmu_index;
    vector<CellIndex> neighbor_indexes;

    for (uint32_t epoch = 0; epoch < epochs; epoch++) {
        dataset->shuffle();

        for (int32_t y = 0; y < height; y++) {
            for (int32_t x = 0; x < width; x++) {
                cells[y][x]->reset_label_counts(labels);
            }
        }

        //TODO: Homework 2 BONUS: Update this loop to be able to
        //update multiple samples simultaneously, locking cells
        //so that their contents can't be accessed or modified 
        //simultaneously by the get_best_matching_unit, get_neighbors
        //and update_cells methods.
        for (const Sample* sample : dataset->samples) {
            bmu_index = get_best_matching_unit(sample);
            // cout << "BMU index: " << bmu_index << endl;

            get_neighbors(bmu_index, neighbor_radius, neighbor_indexes);
            /*
            for (CellIndex index : neighbor_indexes) {
                cout << "\tneighbor index: " << index << endl;
            }
            */

            for (CellIndex neighbor_index : neighbor_indexes) {
                update_cell(sample, bmu_index, neighbor_index, learning_rate);
            }
        }

        learning_rate = learning_rate * learning_rate_schedule;

        cout << endl << endl;
        cout << "SOM after epoch " << epoch << ", learning rate: " << learning_rate << endl;

        for (int32_t y = 0; y < height; y++) {
            for (int32_t x = 0; x < width; x++) {
                cout << "    ";
                cells[y][x]->print_label_counts(cout);
            }
            cout << endl;
        }
    }
}

// void SelfOrganizingMap::print_cell_code() {
//         for (int32_t y = 0; y < height; y++) {
//             for (int32_t x = 0; x < width; x++) {
//                 cout << "som->cells[" << y << "][" << x << "]->values = {";

//                 bool first = true;

//                 for (int32_t i = 0; i < cells[y][x]->values.size(); i++) {
//                     if (first) {
//                         first = false;
//                     } else {
//                         cout << ", ";
//                     }
//                     cout << setprecision(15) << cells[y][x]->values[i];
//                 }
//                 cout << "};" << endl;

//                 cells[y][x]->reset_label_counts(labels);
//             }
//         }
// }
