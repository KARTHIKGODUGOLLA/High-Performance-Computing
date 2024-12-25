#include <cuda_runtime.h>
#include <GL/glut.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>

using namespace std;

// Maximum block size
const int MAX_THREADS_PER_BLOCK = 1024;

// Simulation parameters
int n_bodies = 1000;       
float time_step = 0.01f;  
float damping = 0.99f;     
float softening_sq = 0.1f; 
int block_size = 1024;    

// Device pointers
float *d_x, *d_y, *d_z;       
float *d_mass;              
float *d_x_next, *d_y_next, *d_z_next; 
float *d_x_velocity, *d_y_velocity, *d_z_velocity; 
float *d_x_acceleration, *d_y_acceleration, *d_z_acceleration;

// Host arrays
float *x, *y, *z;
float *mass;
int window_width = 800;
int window_height = 800;

// CUDA kernel to compute accelerations
__global__ void computeAccelerations(
    int n_bodies, float *x, float *y, float *z, float *mass,
    float *x_acc, float *y_acc, float *z_acc, float softening_sq) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;

    float acc_x = 0.0f, acc_y = 0.0f, acc_z = 0.0f;

    for (int j = 0; j < n_bodies; j++) {
        if (i == j) continue;

        float dist_x = x[j] - x[i];
        float dist_y = y[j] - y[i];
        float dist_z = z[j] - z[i];

        float dist_sqr = dist_x * dist_x + dist_y * dist_y + dist_z * dist_z + softening_sq;
        float inv_dist_cube = rsqrtf(dist_sqr * dist_sqr * dist_sqr);

        // Reduce gravitational strength for very close distances
        if (dist_sqr < 0.01f) inv_dist_cube *= 0.1f;

        acc_x += dist_x * mass[j] * inv_dist_cube;
        acc_y += dist_y * mass[j] * inv_dist_cube;
        acc_z += dist_z * mass[j] * inv_dist_cube;
    }

    x_acc[i] = acc_x;
    y_acc[i] = acc_y;
    z_acc[i] = acc_z;
}

// CUDA kernel to update positions and velocities
__global__ void updatePositionsAndVelocities(
    int n_bodies, float time_step, float damping,
    float *x, float *y, float *z,
    float *x_next, float *y_next, float *z_next,
    float *vx, float *vy, float *vz,
    float *ax, float *ay, float *az, float *mass) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies) return;

    vx[i] = (vx[i] + (ax[i] / mass[i]) * time_step) * damping;
    vy[i] = (vy[i] + (ay[i] / mass[i]) * time_step) * damping;
    vz[i] = (vz[i] + (az[i] / mass[i]) * time_step) * damping;

    x_next[i] = x[i] + vx[i] * time_step;
    y_next[i] = y[i] + vy[i] * time_step;
    z_next[i] = z[i] + vz[i] * time_step;
}

// OpenGL display function
void display() {
    int num_blocks = (n_bodies + block_size - 1) / block_size;

    // Launch kernel to compute accelerations
    computeAccelerations<<<num_blocks, block_size>>>(
        n_bodies, d_x, d_y, d_z, d_mass, d_x_acceleration, d_y_acceleration, d_z_acceleration, softening_sq);

    // Launch kernel to update positions and velocities
    updatePositionsAndVelocities<<<num_blocks, block_size>>>(
        n_bodies, time_step, damping, d_x, d_y, d_z, d_x_next, d_y_next, d_z_next,
        d_x_velocity, d_y_velocity, d_z_velocity, d_x_acceleration, d_y_acceleration, d_z_acceleration, d_mass);

    // Swap device arrays for the next iteration
    swap(d_x, d_x_next);
    swap(d_y, d_y_next);
    swap(d_z, d_z_next);

    // Copy updated positions back to host for rendering
    cudaMemcpy(x, d_x, n_bodies * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, n_bodies * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(z, d_z, n_bodies * sizeof(float), cudaMemcpyDeviceToHost);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glBegin(GL_POINTS);
    for (int i = 0; i < n_bodies; i++) {
        glVertex3f(x[i], y[i], z[i]);
    }
    glEnd();

    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
}

// command-line arguments
void parseArguments(int argc, char **argv) {
    bool block_size_defined = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--window_size") == 0 && i + 2 < argc) {
            window_width = atoi(argv[++i]);
            window_height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n_bodies") == 0 && i + 1 < argc) {
            n_bodies = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--time_step") == 0 && i + 1 < argc) {
            time_step = atof(argv[++i]);
        } else if (strcmp(argv[i], "--block_size") == 0 && i + 1 < argc) {
            block_size = atoi(argv[++i]);
            block_size_defined = true;
            if (block_size > MAX_THREADS_PER_BLOCK) {
                cerr << "Error: Block size cannot exceed " << MAX_THREADS_PER_BLOCK << "." << endl;
                exit(1);
            }
        }
    }

    if (!block_size_defined) {
        block_size = MAX_THREADS_PER_BLOCK;
        if (n_bodies > block_size) {
            cerr << "Error: Number of bodies exceeds the maximum supported block size of 1024." << endl;
            exit(1);
        }
    }
}


int main(int argc, char **argv) {
    parseArguments(argc, argv);
    // Initialize bodies with random positions and velocities
    // Initialize host arrays
    mass = new float[n_bodies];
    x = new float[n_bodies];
    y = new float[n_bodies];
    z = new float[n_bodies];
    float *vx = new float[n_bodies];
    float *vy = new float[n_bodies];
    float *vz = new float[n_bodies];

    for (int i = 0; i < n_bodies; i++) {
        mass[i] = 1.0f;
        x[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f; 
        y[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        z[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;

        vx[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
        vy[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
        vz[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_mass, n_bodies * sizeof(float));
    cudaMalloc(&d_x, n_bodies * sizeof(float));
    cudaMalloc(&d_y, n_bodies * sizeof(float));
    cudaMalloc(&d_z, n_bodies * sizeof(float));
    cudaMalloc(&d_x_next, n_bodies * sizeof(float));
    cudaMalloc(&d_y_next, n_bodies * sizeof(float));
    cudaMalloc(&d_z_next, n_bodies * sizeof(float));
    cudaMalloc(&d_x_velocity, n_bodies * sizeof(float));
    cudaMalloc(&d_y_velocity, n_bodies * sizeof(float));
    cudaMalloc(&d_z_velocity, n_bodies * sizeof(float));
    cudaMalloc(&d_x_acceleration, n_bodies * sizeof(float));
    cudaMalloc(&d_y_acceleration, n_bodies * sizeof(float));
    cudaMalloc(&d_z_acceleration, n_bodies * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_mass, mass, n_bodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, n_bodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n_bodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, n_bodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_velocity, vx, n_bodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_velocity, vy, n_bodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z_velocity, vz, n_bodies * sizeof(float), cudaMemcpyHostToDevice);

    // OpenGL setup
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("CUDA N-Body Simulation");
    glutDisplayFunc(display);

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glPointSize(2.0f);

    glutMainLoop();

    // Cleanup
    delete[] mass;
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] vx;
    delete[] vy;
    delete[] vz;

    cudaFree(d_mass);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_x_next);
    cudaFree(d_y_next);
    cudaFree(d_z_next);
    cudaFree(d_x_velocity);
    cudaFree(d_y_velocity);
    cudaFree(d_z_velocity);
    cudaFree(d_x_acceleration);
    cudaFree(d_y_acceleration);
    cudaFree(d_z_acceleration);

    return 0;
}

