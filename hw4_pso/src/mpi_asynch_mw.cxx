#include <mpi.h>
#include <limits>
using std::numeric_limits;

#include <iomanip>
using std::fixed;
using std::right;
using std::setprecision;
using std::setw;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <random>
using std::mt19937;
using std::random_device;
using std::uniform_real_distribution;

#include <string>
using std::stod;
using std::stoi;
using std::string;

#include <vector>
using std::vector;

#include "objectives/objective_function.hxx"
#include "objectives/sphere.hxx"
#include "objectives/ackley.hxx"
#include "objectives/griewank.hxx"
#include "objectives/rastrigin.hxx"
#include "objectives/rosenbrock.hxx"
#include "objectives/neural_network.hxx"

void print_particle(const vector<double> &particle, const vector<double> &velocity, double fitness){
    cout << "fitness: " << right << fixed << setw(12) << setprecision(7) << fitness << ", particle[";
    for (int32_t j = 0; j < particle.size(); j++)
    {
        cout << right << fixed << setw(12) << setprecision(7) << particle[j];
    }
    
    if (velocity.size() > 0)
    {
        cout << " velocity [";
        for (int32_t j = 0; j < velocity.size(); j++)
        {
            cout << right << fixed << setw(12) << setprecision(7) << velocity[j];
        }
        cout << "]";
    }
    cout << endl;
}

int main(int argc, char **argv)
{
    int comm_sz; /* number of processes  */
    int my_rank; /* process rank         */

    // Initialize the command line arguments on every process
    MPI_Init(&argc, &argv);

    // Get the number of processes in MPI_COMM_WORLD, and put
    // it in the 'comm_sz" variable; ie., how many processes
    // are running this program (the same as what you put in the
    //-np argument).
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // Get the rank of this particular process in MPI_COMM_WORLD,
    // and put it in the 'my_rank' variable -- ie., what number
    // is this process
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    vector<string> arguments = vector<string>(argv, argv + argc);

    // Check to see command line arguments were correctly specified
    if (arguments.size() != 8 && arguments.size() != 9)
    {
        if (my_rank == 0)
        {
            string executable_name = arguments[0];
            cout << "Incorrect arguments." << endl;

            cout << "Usage for benchmark objective functions:" << endl;
            cout << "./" << executable_name << " <population size> <w> <c1> <c2> <max_epochs> <objective_function> <number_parameters" << endl;
            cout << "possible objective functions: sphere, ackley, griewank, rastrigin, rosenbrock, neural_network" << endl;

            cout << "Usage for neural network objective functions:" << endl;
            cout << "./" << executable_name << " <population size> <w> <c1> <c2> <max_epochs> neural_network <dataset_filename> <number_layers>" << endl;
        }
        MPI_Finalize();
        return -1;
    }
    else
    {
        int32_t population_size = stoi(arguments[1]);
        double w = stod(arguments[2]);
        double c1 = stod(arguments[3]);
        double c2 = stod(arguments[4]);
        int32_t max_particles = stoi(arguments[5]);
        string objective_function_name(arguments[6]);

        int32_t n_parameters = 0;
        ObjectiveFunction *of = NULL;
        if (my_rank == 0)
            cout << "creating objective function: '" << objective_function_name << "'" << endl;
        if (objective_function_name == "neural_network")
        {
            string dataset_filename = arguments[7];
            int32_t n_layers = stoi(arguments[8]);

            of = new NeuralNetwork(-20.0, 20.0, dataset_filename, n_layers);
            n_parameters = of->get_n_parameters();
        }
        else
        {
            n_parameters = stoi(arguments[7]);

            // Initialize the objective functions with the traditional parameter bounds from
            // https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume24/ortizboyer05a-html/node6.html
            if (objective_function_name == "sphere")
            {
                of = new Sphere(n_parameters, -5.12, 5.12);

                vector<double> best_parameters(n_parameters, 0);
                double best_fitness = of->evaluate(best_parameters);
                cout << "best fitness for optimal parameters: " << best_fitness << endl;
            }
            else if (objective_function_name == "ackley")
            {
                of = new Ackley(n_parameters, -30, 30);

                vector<double> best_parameters(n_parameters, 0);
                double best_fitness = of->evaluate(best_parameters);
                cout << "best fitness for optimal parameters: " << best_fitness << endl;
            }
            else if (objective_function_name == "griewank")
            {
                of = new Griewank(n_parameters, -600, 600);

                vector<double> best_parameters(n_parameters, 0);
                double best_fitness = of->evaluate(best_parameters);
                cout << "best fitness for optimal parameters: " << best_fitness << endl;
            }
            else if (objective_function_name == "rastrigin")
            {
                of = new Rastrigin(n_parameters, -2.048, 2.048);

                vector<double> best_parameters(n_parameters, 0);
                double best_fitness = of->evaluate(best_parameters);
                cout << "best fitness for optimal parameters: " << best_fitness << endl;
            }
            else if (objective_function_name == "rosenbrock")
            {
                of = new Rosenbrock(n_parameters, -5.12, 5.12);

                vector<double> best_parameters(n_parameters, 1.0);
                double best_fitness = of->evaluate(best_parameters);
                cout << "best fitness for optimal parameters: " << best_fitness << endl;
            }
            else
            {
                cerr << "Unknown objective function '" << objective_function_name << "'" << endl;
                cerr << "Options:" << endl;
                cerr << "\tsphere" << endl;
                cerr << "\tackley" << endl;
                cerr << "\tgriewank" << endl;
                cerr << "\trastrigin" << endl;
                cerr << "\tneural_network" << endl;

                exit(1);
            }
        }

        //master process
        if (my_rank == 0)
        {
            vector<double> min_bounds = of->get_min_bounds();
            vector<double> max_bounds = of->get_max_bounds();

            for (int32_t i = 0; i < n_parameters; i++)
            {
                cout << "parameter [" << i << "] min bound: " << min_bounds[i]
                     << ", max bound: " << max_bounds[i] << endl;
            }
            // set up the random number generator
            random_device rd;
            mt19937 random_number_generator(rd());
            uniform_real_distribution<double> distribution(0.0, 1.0);

            // initialize the particles for particle swarm optimization
            vector<vector<double>> particles(population_size, vector<double>(n_parameters));
            vector<vector<double>> velocities(population_size, vector<double>(n_parameters));
            vector<vector<double>> local_bests(population_size, vector<double>(n_parameters));

            // initialize each fitness to the maximum possible double value
            // so the first time we compute a particle fitness it will become
            // the local best fitness
            vector<double> local_best_fitnesses(population_size, numeric_limits<double>::max());

            int32_t global_best_index = -1;
            vector<double> global_best(n_parameters);
            double global_best_fitness = numeric_limits<double>::max();

            for (int32_t i = 0; i < population_size; i++)
            {

                // randomly initialize the particle values and local bests
                for (int32_t j = 0; j < n_parameters; j++)
                {
                    double r = distribution(random_number_generator);
                    particles[i][j] = ((max_bounds[j] - min_bounds[j]) * r) + min_bounds[j];
                    local_bests[i][j] = particles[i][j];

                    // give each particle a small random initial velocity
                    velocities[i][j] = 0.05 * (((max_bounds[j] - min_bounds[j]) * r) + min_bounds[j]);
                }
                // print_particle(particles[i], velocities[i], local_best_fitnesses[i]);
            }
            cout << "beginning particle swarm optimization with paramerers:" << endl;
            cout << "\tpopulation_size: " << population_size << endl;
            cout << "\tn_parameters: " << n_parameters << endl;
            cout << "\tw: " << w << endl;
            cout << "\tc1: " << c1 << endl;
            cout << "\tc2: " << c2 << endl;

            int total_processes;
            MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

            int32_t total_particles_sent = 0;

            while(total_particles_sent < max_particles)
            {
                int token;

                MPI_Request request;
                MPI_Status status;

                MPI_Irecv(&token, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, &status);
                int tag = status.MPI_TAG;
                int source = status.MPI_SOURCE;

                if (tag == 1)
                {
                    MPI_Request req;
                    MPI_Request particle_request;
                    MPI_Request velocity_req;
                    int particle_index = total_particles_sent % population_size;
                    MPI_Isend(&particle_index, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &req);

                    MPI_Isend(particles[particle_index].data(), n_parameters,
                     MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &particle_request);
                    MPI_Isend(velocities[particle_index].data(), n_parameters,
                     MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &velocity_req);
                    total_particles_sent++;
                }
                else if (tag == 2)
                {
                    int idx = token;
                    double fitness;
                    MPI_Request requests[3];

                    MPI_Irecv(&fitness, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &requests[0]);
                    MPI_Irecv(particles[idx].data(), n_parameters, MPI_DOUBLE, 
                    source, tag, MPI_COMM_WORLD, &requests[1]);
                    MPI_Irecv(velocities[idx].data(), n_parameters, MPI_DOUBLE,
                     source, tag, MPI_COMM_WORLD, &requests[2]);
                    MPI_Status status[3];
                    MPI_Waitall(3, requests, status);

                    if (fitness < local_best_fitnesses[idx])
                    {
                        for (int32_t i = 0; i < n_parameters; i++)
                        {
                            local_bests[idx][i] = particles[idx][i];
                        }
                        local_best_fitnesses[idx] = fitness;
                    }
                    if (fitness < global_best_fitness)
                    {
                        for (int32_t j = 0; j < n_parameters; j++)
                        {
                            global_best[j] = particles[idx][j];
                        }
                        global_best_fitness = fitness;
                        global_best_index = idx;
                        cout << "new GLOBAL BEST " << setw(3) << idx << " ";
                        print_particle(particles[idx], velocities[idx], local_best_fitnesses[idx]);
                    }

                    // get the random numbers for moving the particle
                    double r1 = distribution(random_number_generator);
                    double r2 = distribution(random_number_generator);

                    // calculate the particle's new velocity and move it
                    for (int32_t j = 0; j < n_parameters; j++)
                    {
                        velocities[idx][j] = (w * velocities[idx][j]) + (r1 * c1 * (local_bests[idx][j] - particles[idx][j])) + (r2 * c2 * (global_best[j] - particles[idx][j]));

                        double new_position = particles[idx][j] + velocities[idx][j];

                        // if the particle moves out of bounds, move it back into bounds and adjust the velocity accordingly
                        if (new_position < min_bounds[j])
                        {
                            // cout << "particle " << particles[i][j] << " moved to " << new_position << " out of bounds, velocity was " << velocities[i][j];
                            new_position = min_bounds[j] * 0.9;
                            velocities[idx][j] = new_position - particles[idx][j];
                            // cout << " new position " << new_position << " and velocity " << velocities[i][j] << endl;
                        }

                        if (new_position > max_bounds[j])
                        {
                            // cout << "particle " << particles[i][j] << " moved to " << new_position << " out of bounds, velocity was " << velocities[i][j];
                            new_position = max_bounds[j] * 0.9;
                            velocities[idx][j] = new_position - particles[idx][j];
                            // cout << " new position " << new_position << " and velocity " << velocities[i][j] << endl;
                        }

                        particles[idx][j] = new_position;
                    }
                }

                // to ensure global_best_index is valid before accessing related data
                bool valid_global_best_index=false;
                valid_global_best_index=((total_particles_sent + 1) 
                          % (max_particles / 10) == 0);
                if (valid_global_best_index)
                {
                    cout << "global best fitness " << global_best_fitness << " global best index "
                         << global_best_index << endl;
                    print_particle(global_best, velocities[global_best_index], local_best_fitnesses[global_best_index]);
                }
            }

            //terminate
            for (int i = 1; i < total_processes; ++i)
            {
                int buff = 0;
                MPI_Send(&buff, 1, MPI_INT, i, 3, MPI_COMM_WORLD);
            }

            cout << " global best " << global_best_fitness << " at index " << global_best_index << " ";
            print_particle(global_best, velocities[global_best_index], global_best_fitness);
        }
        else
        {
            int my_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
            while (true)
            {
                int index = 0;
                MPI_Request requests[2];
                MPI_Status status;

                MPI_Isend(&index, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &requests[0]);
                MPI_Irecv(&index, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &requests[1]);
                MPI_Wait(&requests[1], &status);

                vector<double> particle(n_parameters);
                vector<double> velocity(n_parameters);

                int tag = status.MPI_TAG;
                if (tag == 1)
                {
                    int idx = index;

                    MPI_Request particles_request[2];
                    MPI_Status particle_status[2];

                    MPI_Irecv(particle.data(), n_parameters, MPI_DOUBLE, 0, 1, 
                    MPI_COMM_WORLD, &particles_request[0]);
                    MPI_Irecv(velocity.data(), n_parameters, MPI_DOUBLE, 0, 1, 
                    MPI_COMM_WORLD, &particles_request[1]);
                    MPI_Waitall(2, particles_request, particle_status);

                    double fitness = of->evaluate(particle);
                    MPI_Request reqs[4];

                    MPI_Isend(&idx, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &reqs[0]);
                    MPI_Isend(&fitness, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &reqs[1]);
                    MPI_Isend(particle.data(), n_parameters, MPI_DOUBLE, 0, 2, 
                    MPI_COMM_WORLD, &reqs[2]);
                    MPI_Isend(velocity.data(), n_parameters, MPI_DOUBLE, 0, 2, 
                    MPI_COMM_WORLD, &reqs[3]);
                }
                //termination check
                else if (tag == 3)
                {
                    break;
                }
            }
        }
    }

    MPI_Finalize();
    return 0;
}