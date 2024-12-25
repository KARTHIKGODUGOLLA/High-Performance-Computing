#ifndef CELL_HXX
#define CELL_HXX

#include <iostream>
using std::ostream;

#include <map>
using std::map;

#include <random>
using std::mt19937;
using std::normal_distribution;

#include <string>
using std::string;

#include <vector>
using std::vector;

class Cell {
    public:
        //A vector to store the cell's values
        vector<double> values;

        //a map to keep track of how many samples with what label
        //were mapped to this cell as their best matching unit. we
        //can use this to determine how well the SOM is clustering
        //our data.
        map<string, double> label_counts;

        /**
         * Initializes a cell with the given number of parameters and the
         * provided normal distribution random number generator.
         *
         * \param size is the number of values to have in the cell.
         * \param generator is a mersenne twister random number generator.
         * \param distribution A normal distribution random number generator.
         */
        Cell(uint32_t size, mt19937 &generator, normal_distribution<double> &distribution);

        /**
         * \return the number of values in the cell.
         */
        uint32_t size() const;

        /**
         * Will set or reset the counts in the label_counts map to 0 for each
         * label.
         *
         * \param labels the vector of unique labels for the dataset the SOM is
         *      being trained on.
         */
        void reset_label_counts(const vector<string> &labels);

        /**
         * When the cell gets matched to as a best matching unit, increase the
         * label count for that class. With this we can track how many of each
         * class mapped to which units (to see how well our data is clustering).
         *
         * \param label is the label to increase the label count for.
         */
        void bmu_match(const string &label);

        /**
         * Overloading the [] operator for the cell class so we can more
         * easily (and safely) access its values.
         *
         * \param index is the index of the value to return.
         */
        double& operator[](uint32_t index);

        /**
         * Overloading the [] operator for the sample class so we can more
         * easily (and safely) access its values. This allows us to use this
         * on const instances of samples (e.g., in the << operator).
         *
         * \param index is the index of the value to return.
         */
        const double& operator[](uint32_t index) const;

        /**
         * Prints out just the label counts so we can visualize how well the SOM
         * has learned the data.
         *
         * \param stream is the stream to write the label counts to.
         */
        void print_label_counts(ostream& stream) const;

        /**
         * Overloads the << operator so we can nicely print out the cell class
         * to cout, cerr or any stream.
         */
        friend ostream& operator<<(ostream& stream, const Cell& cell);

        /**
         * Overloads the << operator so we can nicely print out the cell class
         * to cout, cerr or any stream. This one will accept a pointer instead of
         * a non-pointer version of the class.
         */
        friend ostream& operator<<(ostream& stream, const Cell *cell);

};
#endif
