#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <string>
using std::string;
using std::stoi;
using std::stod;

#include <vector>
using std::vector;

#include "dataset.hxx"
#include "som.hxx"


int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    bool use_hex_grid = false;
    if (arguments.size() == 9 && arguments[8] == "--hex") {
        use_hex_grid = true;
        cout << "Using hexagonal grid" << endl;
    }
    //check to see command line arguments were correctly specified
    if (arguments.size() < 8|| arguments.size() > 9) {
        string executable_name = arguments[0];
        cerr << "Incorrect arguments. Usage:" << endl;
        cerr << "./" << executable_name << " <dataset_filename> <som_width> <som_height> <learning_rate> <learning_rate_schedule> <neighbor_radius> <epochs> <--hex>" << endl;
        exit(1);
    }

    //parse command line arguments
    string dataset_filename = arguments[1];
    int32_t som_width = stoi(arguments[2]);
    int32_t som_height = stoi(arguments[3]);
    double learning_rate = stod(arguments[4]);
    double learning_rate_schedule = stod(arguments[5]);
    int32_t neighbor_radius = stoi(arguments[6]);
    int32_t epochs = stoi(arguments[7]);

    cout << "input dataset filename: '" << dataset_filename << "'" << endl;
    cout << "som width: " << som_width << endl;
    cout << "som height: " << som_height << endl;
    cout << "learning rate: " << learning_rate << endl;
    cout << "learning rate schedule: " << learning_rate_schedule << endl;

    Dataset *dataset = new Dataset(dataset_filename);

    // normalize the dataset
    vector<double> sample_means;
    vector<double> sample_stds;
    dataset->get_statistics(sample_means, sample_stds);
    dataset->normalize(sample_means, sample_stds);

    dataset->shuffle();
    for (int i = 0; i < 5; i++) {
        cout << (*dataset)[i] << endl;
    }

    SelfOrganizingMap *som = new SelfOrganizingMap(dataset->labels, dataset->sample_size(), som_width, som_height);

    som->train(dataset, epochs, neighbor_radius, learning_rate, learning_rate_schedule,use_hex_grid);

    delete dataset;
    delete som;
}
