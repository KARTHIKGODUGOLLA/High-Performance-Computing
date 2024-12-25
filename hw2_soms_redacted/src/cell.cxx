#include <iomanip>
using std::setw;
using std::left;

#include <iostream>
using std::ostream;

#include <random>
using std::mt19937;
using std::normal_distribution;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "cell.hxx"



Cell::Cell(uint32_t size, mt19937 &generator, normal_distribution<double> &distribution) {
    values.reserve(size);
    for (uint32_t i = 0; i < size; i++) {
        values.push_back(distribution(generator));
    }
}

void Cell::reset_label_counts(const vector<string> &labels) {
    for (string label : labels) {
        label_counts[label] = 0;
    }
}

uint32_t Cell::size() const {
    return values.size();
}

void Cell::bmu_match(const string &label) {
    label_counts[label] += 1;
}

double& Cell::operator[](uint32_t index) {
    return values.at(index);
}

const double& Cell::operator[](uint32_t index) const {
    return values.at(index);
}


void Cell::print_label_counts(ostream& stream) const {
    stream << "[";

    bool first = true;
    for (auto const& [label, count] : label_counts) {
        if (first) {
            first = false;
        } else {
            stream << " ";
        }

        stream << label << ":" << left << setw(4) << count;
    }

    stream << "]";
}


ostream& operator<<(ostream& stream, const Cell& cell) {
    stream << "[cell | label counts:";

    for (auto const& [label, count] : cell.label_counts) {
        stream << " " << label << ":" << count;
    }

    stream << " | values: ";

    for (uint32_t i = 0; i < cell.values.size(); i++) {
        if (i > 0) stream << ", ";
        stream << cell[i];
    }

    stream << "]";
    return stream;
}

ostream& operator<<(ostream& stream, const Cell *cell) {
    // this will call the non-pointer version by dereferencing
    // the pointer to cell
    stream << *cell;
    return stream;
}

