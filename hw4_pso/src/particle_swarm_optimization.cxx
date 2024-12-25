#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <string>
#include <limits>
#include <algorithm>
#include "objectives/objective_function.hxx"
#include "objectives/sphere.hxx"
#include "objectives/ackley.hxx"
#include "objectives/griewank.hxx"
#include "objectives/rastrigin.hxx"
#include "objectives/rosenbrock.hxx"
#include "objectives/neural_network.hxx"

using namespace std;

void print_particle(const vector<double>& particle, double fitness) {
    cout << "fitness: " << fixed << setw(12) << setprecision(7) << fitness << " particle [";
    for (double val : particle) {
        cout << " " << fixed << setw(12) << setprecision(7) << val;
    }
    cout << "]" << endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 8 && argc != 9) {
        if (rank == 0) {
            cerr << "Incorrect arguments.\n"
                 << argv[0] << " <population size> <w> <c1> <c2> <max_epochs> <objective_function> <number_parameters>\n"
                 << "Possible objective functions: sphere, ackley, griewank, rastrigin, rosenbrock, neural_network\n"
                 << "Usage for neural network objective function:\n"
                 << argv[0] << " <population size> <w> <c1> <c2> <max_epochs> neural_network <dataset_filename> <number_layers>\n";
        }
        MPI_Finalize();
        exit(1);
    }

    int population_size = stoi(argv[1]);
    double w = stod(argv[2]);
    double c1 = stod(argv[3]);
    double c2 = stod(argv[4]);
    int max_epochs = stoi(argv[5]);
    string objective_function_name(argv[6]);

    int n_parameters = 0;
    ObjectiveFunction* of = nullptr;

    if (objective_function_name == "neural_network") {
        string dataset_filename = argv[7];
        int n_layers = stoi(argv[8]);
        of = new NeuralNetwork(-20.0, 20.0, dataset_filename, n_layers);
        n_parameters = of->get_n_parameters();
    } else {
        n_parameters = stoi(argv[7]);

        if (objective_function_name == "sphere") {
            of = new Sphere(n_parameters, -5.12, 5.12);
            vector<double> best_parameters(n_parameters, 0);
            double best_fitness = of->evaluate(best_parameters);
            cout << "best fitness for optimal parameters: " << best_fitness << endl;            
        } else if (objective_function_name == "ackley") {
            of = new Ackley(n_parameters, -30, 30);
            vector<double> best_parameters(n_parameters, 0);
            double best_fitness = of->evaluate(best_parameters);
            cout << "best fitness for optimal parameters: " << best_fitness << endl;
        } else if (objective_function_name == "griewank") {
            of = new Griewank(n_parameters, -600, 600);
            vector<double> best_parameters(n_parameters, 0);
            double best_fitness = of->evaluate(best_parameters);
            cout << "best fitness for optimal parameters: " << best_fitness << endl;
        } else if (objective_function_name == "rastrigin") {
            of = new Rastrigin(n_parameters, -2.048, 2.048);
            vector<double> best_parameters(n_parameters, 0);
            double best_fitness = of->evaluate(best_parameters);
            cout << "best fitness for optimal parameters: " << best_fitness << endl;
        } else if (objective_function_name == "rosenbrock") {
            of = new Rosenbrock(n_parameters, -5.12, 5.12);
            vector<double> best_parameters(n_parameters, 0);
            double best_fitness = of->evaluate(best_parameters);
            cout << "best fitness for optimal parameters: " << best_fitness << endl;
        }
        else
        {
            if (rank == 0)
            {
                cerr << "Unknown object function '" << objective_function_name << "'" << endl;
                cerr << "Options:" << endl;
                cerr << "\tsphere" << endl;
                cerr << "\tackley" << endl;
                cerr << "\tgriewank" << endl;
                cerr << "\trastrigin" << endl;
                cerr << "\tneural_network" << endl;
                exit(1);
            }
            MPI_Finalize();
            exit(1);
        }
    }

    vector<double> min_bounds = of->get_min_bounds();
    vector<double> max_bounds = of->get_max_bounds();

    // RNG setup
    random_device rd;
    mt19937 random_number_generator(rd());
    uniform_real_distribution<double> distribution(0.0, 1.0);

    // Divide workload evenly among processes
    int local_population_size = population_size / size + (rank < population_size % size ? 1 : 0);
    vector<vector<double>> particles(local_population_size, vector<double>(n_parameters));
    vector<vector<double>> velocities(local_population_size, vector<double>(n_parameters));
    vector<vector<double>> local_bests(local_population_size, vector<double>(n_parameters));
    vector<double> local_best_fitnesses(local_population_size, numeric_limits<double>::max());

    vector<double> global_best(n_parameters, 0);
    double global_best_fitness = numeric_limits<double>::max();

    // Initialize local particles
    for (int i = 0; i < local_population_size; ++i) {
        for (int j = 0; j < n_parameters; ++j) {
            double r = distribution(random_number_generator);
            particles[i][j] = min_bounds[j] + r * (max_bounds[j] - min_bounds[j]);
            velocities[i][j] = 0.05 * (max_bounds[j] - min_bounds[j]) * distribution(random_number_generator);
            local_bests[i][j] = particles[i][j];
        }
    }

    for (int epoch = 0; epoch < max_epochs; ++epoch) {
         //Evaluate fitness and update local bests
        for (int i = 0; i < local_population_size; ++i) {
            double fitness = of->evaluate(particles[i]);
            if (fitness < local_best_fitnesses[i]) {
                local_best_fitnesses[i] = fitness;
                local_bests[i] = particles[i];
            }
        }

        // Gather all local best fitnesses
        vector<double> all_local_best_fitnesses(size * local_population_size, numeric_limits<double>::max());
        MPI_Allgather(local_best_fitnesses.data(), local_population_size, MPI_DOUBLE,
                      all_local_best_fitnesses.data(), local_population_size, MPI_DOUBLE, MPI_COMM_WORLD);

        //Gather all local best particles
        vector<double> flat_local_bests(local_population_size * n_parameters);
        for (int i = 0; i < local_population_size; ++i) {
            copy(local_bests[i].begin(), local_bests[i].end(), flat_local_bests.begin() + i * n_parameters);
        }

        vector<double> all_flat_bests(population_size * n_parameters);
        MPI_Allgather(flat_local_bests.data(), local_population_size * n_parameters, MPI_DOUBLE,
                      all_flat_bests.data(), local_population_size * n_parameters, MPI_DOUBLE, MPI_COMM_WORLD);

        //Determine the global best particle
        for (int i = 0; i < population_size; ++i) {
            if (all_local_best_fitnesses[i] < global_best_fitness) {
                global_best_fitness = all_local_best_fitnesses[i];
                global_best.assign(all_flat_bests.begin() + i * n_parameters, all_flat_bests.begin() + (i + 1) * n_parameters);
            }
        }

        //Update particles based on the global best
        for (int i = 0; i < local_population_size; ++i) {
            double r1 = distribution(random_number_generator), r2 = distribution(random_number_generator);
            for (int j = 0; j < n_parameters; ++j) {
                velocities[i][j] = w * velocities[i][j] +
                                   c1 * r1 * (local_bests[i][j] - particles[i][j]) +
                                   c2 * r2 * (global_best[j] - particles[i][j]);
                particles[i][j] += velocities[i][j];
                particles[i][j] = max(min(particles[i][j], max_bounds[j]), min_bounds[j]);
            }
        }

        if (rank == 0) {
            cout << "epoch " << epoch << " global best fitness: " << fixed << setprecision(7) << global_best_fitness << endl;
            print_particle(global_best, global_best_fitness);
        }
    }

    delete of;
    MPI_Finalize();
    return 0;
}