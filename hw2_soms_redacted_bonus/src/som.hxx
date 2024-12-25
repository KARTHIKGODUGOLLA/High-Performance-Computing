#ifndef SOM_HXX
#define SOM_HXX

#include <random>
using std::mt19937;

#include <utility>
using std::pair;

#include <vector>
using std::vector;

#include "cell.hxx"
#include "dataset.hxx"

class CellIndex {
    public:
        // The y-index of a cell in the SOM
        int32_t y;

        // The x-index of a cell in the SOM
        int32_t x;

        /**
         * A default constructor to an uninitalized CellIndex.
         * x and y will be initialized to -1.
         */
        CellIndex();


        /**
         * A simple object to represent an x-y index
         * of a cell in the SOM.
         * 
         * \param y is the y-index of the cell
         * \param x is the x-index of the cell
         */
        CellIndex(int32_t _y, int32_t _x);

        /**
         * \return true if both cell indexes have the same x and y values.
         */
        bool operator==(const CellIndex &other) const;

        /**
         * Overloads the << operator so we can nicely print out the CellIndex class
         * to cout, cerr or any stream.
         */
        friend ostream& operator<<(ostream& stream, const CellIndex& cell_index);
};


class SelfOrganizingMap {
    public:
        //random number generator to use for initializing cell values
        mt19937 random_number_generator;

        //the size (number of values) in each cell, which should be the
        //same as the number of values in each dataset sample
        uint32_t sample_size;

        //a list of the unique labels in the dataset
        vector<string> labels;

        //the number of cells in the SOM in the y dimension
        uint32_t height;

        //the number of cells in the SOM in the x dimension
        uint32_t width;

        //2D vector of the SOM cells (units)
        vector< vector<Cell*> > cells;

        /**
         * Initialize a self organizing map with the given hyperparameters.
         */
        SelfOrganizingMap(vector<string> _labels, uint32_t _sample_size, uint32_t _height, uint32_t _width, uint32_t _batch_size);
        SelfOrganizingMap(vector<string> _labels, uint32_t _sample_size, uint32_t _height, uint32_t _width, int32_t seed, uint32_t _batch_size);

        /**
         * Helper function for the constructors to initialize the cells.
         */
        void initialize_cells();

        /**
         * Destructor for the SelfOrganizingMap class. Needs to call
         * delete on all the cell pointers.
         */
        ~SelfOrganizingMap();

        /**
         * Calculates the distance between a sample and a cell in the
         * self organizing map using Euclidian distance.
         *
         * \param cell is a cell in the SOM
         * \param sample is a sample from the dataset
         * \return the Euclidian distance between the cell and the sample
         */
        double distance_function(const Cell *cell, const Sample *sample);

        /**
         * Finds the index of the best matching unit in the SOM (the
         * unit which has the closest distance to the sample.
         *
         * \param sample is the sample being matched to the SOM.
         * \return the cell index of the best matching unit for the sample.
         */
        CellIndex get_best_matching_unit(const Sample *sample);

        /**
         * Complex the neighborhood function (i.e, theta) used in the cell
         * update equation.
         *
         * \param index1 is the first cell index
         * \param index2 is the second cell index
         *
         * \return theta, the value for the neighborhood function given the difference
         *      of the two cell indexes.
         */
        double neighborhood_function(const CellIndex &index1, const CellIndex &index2);

        /**
         * This method will find all neighbors of the best matching unit and then
         * set the neighbor indexes array to all those neighbors AND the best matching
         * unit.
         * 
         * \param bmu_index is the index of the best matching unit.
         * \param neighbor_radius is the maximum distance (in terms of indexes) for
         *      a cell to be included as a neighbor. if neighbor radius is < 0, then
         *      all cells will be included as neighbors. So for example, if the
         *      neighbor radius is 2, then any cell (including the BMU) with 
         *      (|bmu_x - cell_x| <= 2 AND |bmu_y - cell_y|) <= 2 will be added to
         *      the neighbor index vector.
         * \param neighbors (output) will be updated to be a vector of the best 
         *      matching unit and all of its neighbors within the given neighbor
         *      radius.
         */
        void get_neighbors(const CellIndex &bmu_index, int32_t neighbor_radius, vector<CellIndex> &neighbor_indexes);

        /**
         * This method will update a cell according to the self organizing map weight
         * update rule:
         *  cell[i] += theta * learning_rate * (sample[i] - cell[i])
         *
         * Where theta is calculated using the neighborhood function given the best
         * matching unit index and the cell index. Make sure you call the cell->bmu_match 
         * method to update the label counts for this epoch to track how well the SOM is 
         * learning, if the passed neighbor is the BMU.
         *
         * \param sample is the dataset sample we are training with
         * \param bmu_index is the index of the best matching unit
         * \param cell_index is the index of the cell to update.
         * \param learning_rate is the current learning rate (which is being updated
         *      during the training loop each epoch according to the schedule).
         */
        void update_cell(const Sample *sample, const CellIndex &bmu_index, const CellIndex &cell_index, double learning_rate);


        void train(Dataset *dataset, uint32_t epochs, int32_t neighbor_radius, double learning_rate, double learning_rate_schedule);

        uint32_t batch_size;
        // void print_cell_code();
};

#endif
