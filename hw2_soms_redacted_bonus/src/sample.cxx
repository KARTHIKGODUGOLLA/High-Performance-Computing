#include <iostream>
using std::ostream;
using std::endl;

#include <string>
using std::string;
using std::getline;
using std::stod;

#include <sstream>
using std::istringstream;

#include "sample.hxx"

Sample::Sample(const string &line) {
    /**********
     * TODO: Implement the constructor to populate the the label
     * and values fields of the Sample class.
     * hint: use getline and a stringstream
     **********/
        std::stringstream strStream(line);
        std::string elem;

        //extracting the first element and reamining with "," as delimter
        if (std::getline(strStream, elem, ',')) {
            label = elem;
        }
        while (std::getline(strStream, elem, ',')) {
            double value = std::stod(elem);
            values.push_back(value); 
        }
    /********** ENDTODO **********/
}


Sample::~Sample() {
}

uint32_t Sample::size() const {
    return values.size();
}

double& Sample::operator[](uint32_t index) {
    return values.at(index);
}

const double& Sample::operator[](uint32_t index) const {
    return values.at(index);
}

ostream& operator<<(ostream& stream, const Sample& sample) {
    stream << "[sample | label: " << sample.label << " | ";

    for (uint32_t i = 0; i < sample.values.size(); i++) {
        if (i > 0) stream << ", ";
        stream << sample[i];
    }

    stream << "]";
    return stream;
}

ostream& operator<<(ostream& stream, const Sample *sample) {
    // this will call the non-pointer version by dereferencing
    // the pointer to sample
    stream << *sample;
    return stream;
}

