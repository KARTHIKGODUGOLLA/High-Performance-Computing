#include <algorithm>
using std::find;
using std::shuffle;

#include <cmath>
using std::sqrt;

#include <iostream>
using std::cout;
using std::endl;

#include <fstream>
using std::ifstream;

#include <random>
using std::mt19937;
using std::random_device;

#include <string>
using std::getline;

#include "dataset.hxx"
#include "sample.hxx"

void Dataset::read_samples_from_file(const string &filename) {
    ifstream input_stream(filename);

    string line;
    while (getline(input_stream, line)) {
        Sample *sample = new Sample(line);
        samples.push_back(sample);
        // cout << sample << endl;

        if (find(labels.begin(), labels.end(), sample->label) == labels.end()) {
            labels.push_back(sample->label);
        }
    }

    cout << "dataset labels:";
    for (string label : labels) {
        cout << " " << label;
    }
    cout << endl;
}

Dataset::Dataset(const string &filename, int32_t seed) {
    random_number_generator = mt19937(seed);

    read_samples_from_file(filename);
}

Dataset::Dataset(const string &filename) {
    random_device rd;
    random_number_generator = mt19937(rd());

    read_samples_from_file(filename);
}


Dataset::~Dataset() {
    /**********
     * TODO: Implement the destructor
     **********/
    for (Sample* sample : samples) {
        delete sample;
    }
    samples.clear();
    /********** ENDTODO **********/
}

int32_t Dataset::sample_size() {
    if (samples.size() == 0) {
        return -1;
    } else {
        //all samples should be the same size
        return samples[0]->size();
    }
}

void Dataset::get_statistics(vector<double> &sample_means, vector<double> &sample_stds) {
    /**********
     * TODO: Resize the input sample means and calculate
     * the mean for each sample value.
     ***********/
    size_t total_samples = samples.size();
    if (total_samples == 0) return;

    size_t sample_size = samples[0]->size();
    //resizing the inp vectors to accomodate mean and std
    sample_means.resize(sample_size);
    sample_stds.resize(sample_size);

    std::fill(sample_means.begin(), sample_means.end(), 0.0);
    std::vector<double> variances(sample_size, 0.0);
    //summing up all the samples
    for (Sample* sample : samples) {
        for (size_t i = 0; i < sample_size; ++i) {
            sample_means[i] += (*sample)[i];
        }
    }
    //calculate mean for each sample value
     for (size_t i = 0; i < sample_size; ++i) {
        sample_means[i] /= total_samples;
    }

    for (Sample* sample : samples) {
        for (size_t i = 0; i < sample_size; ++i) {
            double difference=(*sample)[i]-sample_means[i];
            variances[i] +=difference*difference;
        }
    }

    // Calculate standard deviations
    for (size_t i = 0; i < sample_size; ++i) {
        sample_stds[i] = std::sqrt(variances[i] / total_samples);
    }
    /********** ENDTODO **********/
}

void Dataset::normalize(const vector<double> &sample_means, vector<double> &sample_stds) {
    /**********
     * TODO: Normalize the samples using z-score normalization
     * with the provided means and standard deviations.
     ***********/
    //checking if the samples, means or stds are empty
    if (samples.empty() || sample_means.empty() || sample_stds.empty()) return;
    size_t sample_size = samples[0]->size();
    
    for (Sample* sample : samples) {
        for (size_t i = 0; i < sample_size; ++i) {
            if (sample_stds[i] != 0) {
                //normalizing the samples using z-score normalization
                (*sample)[i] = ((*sample)[i] - sample_means[i]) / sample_stds[i];
            }
            else{
                //do nothing
            }
        }
    }

    //cout << "normalized samples: " << endl;

    /********** ENDTODO **********/
}

void Dataset::shuffle() {
    // need to specify the namespace to avoid the compiler being confused
    // with the class's shuffle method.
    std::shuffle(samples.begin(), samples.end(), random_number_generator);
}


Sample* Dataset::operator[](uint32_t index) {
    return samples.at(index);
}
