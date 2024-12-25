/* W B Langdon at MUN 10 May 2007
 * Program to demonstarte use of OpenGL's glDrawPixels
 */

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef __APPLE__
#  include <OpenGL/gl.h>
#  include <OpenGL/glu.h>
#  include <GLUT/glut.h>
#else
#  include <GL/gl.h>
#  include <GL/glu.h>
#  include <GL/glut.h>
#endif

#include <string>
#include <queue>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "math.h"

using std::cin;
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::ostream;
using std::setw;
using std::right;
using std::left;
using std::fixed;
using std::vector;
using std::priority_queue;
using std::setprecision;

typedef double (*ProbabilityDensityFunction)(float mass, float radius, float scale);
typedef double (*CumulativeMassFunction)(float mass, float radius, float scale);
typedef double (*RotationalVelocityFunction)(float mass, float radius, float scale);

float gravitational_constant = 1.0;

int n_bodies;

float scale;
float max_radius;

float radius;
float total_mass;
float time_step;
float softening;
float softening_sq;
float damping;


/**
 *  You're going to need to have the mass, x y z, x y z next, x y z acceleration on the GPUs. The CPU
 *  should only need the x y and z to print the stars to the screen.
 */
float* mass;

float* x;
float* y;
float* z;

float* x_next;
float* y_next;
float* z_next;

float* x_velocity;
float* y_velocity;
float* z_velocity;

float* x_acceleration;
float* y_acceleration;
float* z_acceleration;

int window_size;
int window_width;
int window_height;


/**
 *  This code handles the camera, you can ignore it.
 *  It lets you zoom in (by holding control, click and moving the mouse), move left/right up/down by 
 *  holding shift, click and moving the mouse, and rotate by moving the mouse.
 */
int     ox                  = 0;
int     oy                  = 0;
int     buttonState         = 0; 
float   camera_trans[]      = {0, -0.2, -10};
float   camera_rot[]        = {0, 0, 0};
float   camera_trans_lag[]  = {0, -0.2, -10};
float   camera_rot_lag[]    = {0, 0, 0};
const float inertia         = 0.1f;


void reshape(int w, int h) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 1000.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
}

void mouse_button(int button, int state, int x, int y) {
    int mods;

    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    mods = glutGetModifiers();
    if (mods & GLUT_ACTIVE_SHIFT) 
    {
        buttonState = 2;
    } 
    else if (mods & GLUT_ACTIVE_CTRL) 
    {
        buttonState = 3;
    }

    ox = x; oy = y;

    glutPostRedisplay();
}

void mouse_motion(int x, int y) {
    float dx = (float)(x - ox);
    float dy = (float)(y - oy);

    if (buttonState == 3) 
    {
        // left+middle = zoom
        camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
    } 
    else if (buttonState & 2) 
    {
        // middle = translate
        camera_trans[0] += dx / 100.0f;
        camera_trans[1] -= dy / 100.0f;
    }
    else if (buttonState & 1) 
    {
        // left = rotate
        camera_rot[0] += dy / 5.0f;
        camera_rot[1] += dx / 5.0f;
    }

    ox = x; oy = y;
    glutPostRedisplay();
}


static void swap_array(float **a1, float **a2) {
    float *temp;

    temp = a1[0];
    a1[0] = a2[0];
    a2[0] = temp;
}

/**
 *  The display function gets called repeatedly, updating the visualization of the simulation
 */
void display() {
    float dist_x, dist_y, dist_z;
    float dist_sqr;
    float dist_sixth;
    float inv_dist_cube;
    float si, sj;

    cout << "in display loop!" << endl;

    //The n-body simulation works similar to the heat diffusion simulation.  You want to
    //parallelize these for loops on the GPU and then transfer the data for the x, y and z
    //of the bodies back to the CPU so they can be displayed.
    for (int i = 0; i < n_bodies; i++) {
        x_acceleration[i] = 0;
        y_acceleration[i] = 0;
        z_acceleration[i] = 0;
    }

    //You want to paralleize this, each GPU thread will perform the inner loop over ALL
    //other bodies, not just i+1 to the end.
    for (int i = 0; i < n_bodies; i++) {
        for (int j = i + 1; j < n_bodies; j++) {
            dist_x = x[j] - x[i];
            dist_y = y[j] - y[i];
            dist_z = z[j] - z[i];

            dist_sqr = (dist_x * dist_x) + (dist_y * dist_y) + (dist_z * dist_z) + softening_sq;
            dist_sixth = dist_sqr * dist_sqr * dist_sqr;
            inv_dist_cube = (float)1.0 / sqrt(dist_sixth);

            si = mass[j] * inv_dist_cube;
            sj = mass[i] * inv_dist_cube;

            x_acceleration[i] += dist_x * si;
            y_acceleration[i] += dist_y * si;
            z_acceleration[i] += dist_z * si;

            //to do this on the GPU you should only accumulate the accelleration values for the
            //particular particle the thread is working on.
            x_acceleration[j] -= dist_x * sj;
            y_acceleration[j] -= dist_y * sj;
            z_acceleration[j] -= dist_z * sj;
        }

        x_velocity[i] = (x_velocity[i] + ((x_acceleration[i] / mass[i]) * time_step)) * damping;
        y_velocity[i] = (y_velocity[i] + ((y_acceleration[i] / mass[i]) * time_step)) * damping;
        z_velocity[i] = (z_velocity[i] + ((z_acceleration[i] / mass[i]) * time_step)) * damping;

        x_next[i] = x[i] + x_velocity[i] * time_step;
        y_next[i] = y[i] + y_velocity[i] * time_step;
        z_next[i] = z[i] + z_velocity[i] * time_step;
    }
    //wait for the kernel to complete here
    //move memory back to the CPU if not using managed malloc

    //You should swap the GPU pointers if you are not using managed
    //malloc, otherwise you will need to swap the managed arrays.
    swap_array(&x, &x_next);
    swap_array(&y, &y_next);
    swap_array(&z, &z_next);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

//    cout << "trans: " << camera_trans[0] << ", " << camera_trans[1] << ", " << camera_trans[2] << " -- rot: " << camera_rot[0] << ", " << camera_rot[1] << ", " << camera_rot[2] << endl;
    for (int c = 0; c < 3; ++c)
    {
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
    }

    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

    glBegin(GL_POINTS);
    for (int i = 0; i < n_bodies; i++) {
        glVertex3f(x[i], y[i], z[i]);
    }
    glEnd();

    glFlush();
    glutSwapBuffers();

    glutPostRedisplay();
}


/**
 *  These are different models of galaxies, you can ignore them if you
 *  want.  They just initialize the stars differently.
 */

double plummer_density_profile(float total_mass, float radius, float scale) {
    return ((3 * total_mass) / (4 * M_PI * scale * scale * scale)) * pow((1 + (radius * radius) / (scale * scale)), -5.0 / 2.0);
}

double plummer_cumulative_mass(float total_mass, float radius, float scale) {
    return (total_mass * radius * radius * radius) / pow((scale * scale) + (radius * radius), 3.0/2.0);
}

double plummer_rotational_velocity(float total_mass, float radius, float scale) {
    return sqrt( (gravitational_constant * total_mass * radius * radius) / pow((radius * radius) + (scale * scale), 2.0));
}


double hernquist_density_profile(float total_mass, float radius, float scale) {
    return (total_mass / (2.0 * M_PI)) * scale / (radius * pow(scale + radius, 3.0));
}

double hernquist_cumulative_mass(float total_mass, float radius, float scale) {
    float scale_radius = scale + radius;
    float scale_radius_sq = scale_radius * scale_radius;

    return (total_mass * radius * radius) / scale_radius_sq;
}

double hernquist_rotational_velocity(float total_mass, float radius, float scale) {
    return sqrt( (2.0 * gravitational_constant * total_mass) / radius );
}


double jaffe_density_profile(float total_mass, float radius, float scale) {
    return 0;
}

double jaffe_cumulative_mass(float total_mass, float radius, float scale) {
    return (total_mass * radius) / (scale + radius);
}

double jaffe_rotational_velocity(float total_mass, float radius, float scale) {
    return sqrt( (2.0 * gravitational_constant * total_mass) / radius );
}


double uniform_density_profile(float total_mass, float radius, float scale) {
    if (radius > max_radius) return 0;
    else return 3 * total_mass / (4.0 * M_PI * radius * radius * radius);
}

double uniform_cumulative_mass(float total_mass, float radius, float scale) {
    if (radius >= max_radius) return 1;
    else return total_mass * (radius * radius * radius) / (max_radius * max_radius * max_radius);
}

double uniform_rotational_velocity(float total_mass, float radius, float scale) {
    return sqrt( (2.0 * gravitational_constant * total_mass) / radius );
}

void usage(char *executable) {
    cerr << "Usage for n-body simulation:" << endl;
    cerr << "    " << executable << " <argument list>" << endl;
    cerr << "Possible arguments:" << endl;
    cerr << "   --window_size <x pixels (int)> <y pixels (int)> : window size" << endl;
    cerr << "   --n_bodies <int>                                : number of bodies in the simulation" << endl;
    cerr << "   --max_radius <float>                            : maximum radius for bodies to be generated in (default 1.0)" << endl;
    cerr << "   --total_mass <float>                            : total mass of the bodies (default 1.0) " << endl;
    cerr << "   --scale <float>                                 : scale factor for initial body distribution (default 1.0)" << endl;
    cerr << "   --time_step <float>                             : time step of the simulation" << endl;
    cerr << "   --softening <float>                             : softening factor for the bodies to prevent collisions (default 1.0)" << endl;
    cerr << "   --damping <float>                               : damping factor for velocity, will scale the velocity of the particles (default 1.0)" << endl;
    cerr << "   --model <string>                                : initial distribution model, options: plummer, hernquist or jaffe" << endl;
    exit(1);
}

int main(int argc, char** argv) {
    max_radius = 1.0;
    total_mass = 1.0;
    scale = 1.0;
    softening = 1.0;

    n_bodies = -1;
    time_step = -1;

    damping = 1.0;

    char *model_str = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--window_size") == 0) {
            window_width = atoi(argv[++i]);
            window_height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n_bodies") == 0) {
            n_bodies = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max_radius") == 0) {
            max_radius = atof(argv[++i]);
        } else if (strcmp(argv[i], "--total_mass") == 0) {
            total_mass = atof(argv[++i]);
        } else if (strcmp(argv[i], "--scale") == 0) {
            scale = atof(argv[++i]);
        } else if (strcmp(argv[i], "--time_step") == 0) {
            time_step = atof(argv[++i]);
        } else if (strcmp(argv[i], "--softening") == 0) {
            softening = atof(argv[++i]);
        } else if (strcmp(argv[i], "--damping") == 0) {
            damping = atof(argv[++i]);
        } else if (strcmp(argv[i], "--model") == 0) {
            model_str = argv[++i];
        } else {
            cerr << "Unknown argument '" << argv[i] << "'." << endl;
            usage(argv[0]);
        }
    }
    softening_sq = softening * softening;

    if (time_step <= 0) {
        cerr << "Error! time_step (" << time_step << ") not specified (or misspecified), cannot be <= 0" << endl;
        usage(argv[0]);
    }

    if (n_bodies <= 0) {
        cerr << "Error! n_bodies (" << n_bodies << ") not specified (or misspecified), cannot be <= 0" << endl;
        usage(argv[0]);
    }

    if (model_str == NULL) {
        cerr << "Error! model not specified." << endl;
        usage(argv[0]);
    }

    ProbabilityDensityFunction probability_density_function = NULL;
    CumulativeMassFunction cumulative_mass_function = NULL;
    RotationalVelocityFunction rotational_velocity_function = NULL;

    if (strcmp(model_str, "uniform") == 0) {
        probability_density_function    = uniform_density_profile;
        cumulative_mass_function        = uniform_cumulative_mass;
        rotational_velocity_function    = uniform_rotational_velocity;

    } else if (strcmp(model_str, "plummer") == 0) {
        probability_density_function    = plummer_density_profile;
        cumulative_mass_function        = plummer_cumulative_mass;
        rotational_velocity_function    = plummer_rotational_velocity;

    } else if (strcmp(model_str, "hernquist") == 0) {
        probability_density_function    = hernquist_density_profile;
        cumulative_mass_function        = hernquist_cumulative_mass;
        rotational_velocity_function    = hernquist_rotational_velocity;

    } else if (strcmp(model_str, "jaffe") == 0) {
        probability_density_function    = jaffe_density_profile;
        cumulative_mass_function        = jaffe_cumulative_mass;
        rotational_velocity_function    = jaffe_rotational_velocity;

    } else {
        cerr << "Error! model '" << model_str << "' unknown." << endl;
        usage(argv[0]);
    }

    cout << "Arguments succesfully parsed." << endl;
    cout << "    window_width:  " << setw(10) << window_width << endl;
    cout << "    window_height: " << setw(10) << window_height << endl;
    cout << "    n_bodies:      " << setw(10) << n_bodies << endl;
    cout << "    max_radius:    " << setw(10) << max_radius << endl;
    cout << "    total_mass:    " << setw(10) << total_mass << endl;
    cout << "    scale:         " << setw(10) << scale << endl;
    cout << "    time_step:     " << setw(10) << time_step << endl;
    cout << "    softening:     " << setw(10) << softening << endl;
    cout << "    damping:       " << setw(10) << damping << endl;
    cout << "    model:         " << setw(10) << model_str << endl;

    window_size = window_width * window_height;

    //You will either need to create these with managed malloc or
    //create these on the CPU then make copies for the device/GPU
    //arrays.
    mass = new float[n_bodies];
    x = new float[n_bodies];
    y = new float[n_bodies];
    z = new float[n_bodies];
    x_next = new float[n_bodies];
    y_next = new float[n_bodies];
    z_next = new float[n_bodies];
    x_velocity = new float[n_bodies];
    y_velocity = new float[n_bodies];
    z_velocity = new float[n_bodies];
    x_acceleration = new float[n_bodies];
    y_acceleration = new float[n_bodies];
    z_acceleration = new float[n_bodies];


    for (float i = 0; i < max_radius; i += 0.1) {
        cout << "radius: " << setw(10) << i << ", pdf: " << setw(10) << probability_density_function(total_mass, i, scale) << endl;
    }    
    cout << "the probability density at the maximum radius is: " << probability_density_function(total_mass, max_radius, scale) << endl;

    for (float i = 0; i < max_radius; i += 0.1) {
        cout << "radius: " << setw(10) << i << ", cmf: " << setw(10) << cumulative_mass_function(total_mass, i, scale) << endl;
    }    
    cout << "the cumulative mass at the maximum radius is: " << cumulative_mass_function(total_mass, max_radius, scale) << endl;

    float mass_sum = 0;
    float circular_velocity;
    float velocity_direction1;
    float velocity_direction2;

    //This initializes the positions of the stars according to the
    //different models.
    int accepted = 0;
    while (accepted < n_bodies) {
        float phi = drand48() * 2.0 * M_PI;
        float costheta = (drand48() * 2.0) - 1.0;
        float u = drand48();

        float theta = acos( costheta );
        float random_radius = max_radius * pow(u, 1.0/3.0);

        float m_R = cumulative_mass_function(total_mass, random_radius, scale);       //try replacing cumulative with prob density function

        float random_mass = drand48() * total_mass;

        if (m_R  < random_mass) {
//            cout << "accepted a radius: " << random_radius << ", (m_R) " << m_R << " <= " << random_mass << " (random_mass)" << endl;

            mass[accepted] = (float)total_mass / (float)n_bodies;
            mass_sum += random_mass;
//            cout << "rand_radians1: " << rand_radians1 << ", rand_radians2: " << rand_radians2 << endl;

            x[accepted] = random_radius * sin(theta) * cos(phi);
            y[accepted] = random_radius * sin(theta) * sin(phi);
            z[accepted] = random_radius * cos(theta);

            circular_velocity = rotational_velocity_function(total_mass, random_radius, scale);

            if (drand48() < 0.5) {
                velocity_direction1 = phi + (M_PI / 2.0);
            } else {
                velocity_direction1 = phi - (M_PI / 2.0);
            }

            if (drand48() < 0.5) {
                velocity_direction2 = theta + (M_PI / 2.0);
            } else {
                velocity_direction2 = theta - (M_PI / 2.0);
            }

            x_velocity[accepted] = circular_velocity * sin(velocity_direction1) * cos(velocity_direction2);
            y_velocity[accepted] = circular_velocity * sin(velocity_direction1) * sin(velocity_direction2);
            z_velocity[accepted] = circular_velocity * cos(velocity_direction1);

//            x_velocity[accepted] = 0;
//            y_velocity[accepted] = 0;
//            z_velocity[accepted] = 0;


            /*
            cout << "velocity_x = " << x_velocity[accepted] << ", ";
            cout << "velocity_y = " << y_velocity[accepted] << ", ";
            cout << "velocity_z = " << z_velocity[accepted] << ", ";
            cout << endl;
            */

            accepted++;
        } else {
//            cout << "rejected a radius: " << random_radius << ", (m_R) " << m_R << " > " << random_mass << " (random_mass)" <<endl;
        }
    }

    vector<float> bins;
    float bin_size = 0.1;
    for (int i = 0; i < n_bodies; i++) {
        float radius = sqrt((x[i] * x[i]) + (y[i] * y[i]) + (z[i] * z[i]));

        int bin = radius / bin_size;

        if (bin > bins.size()) bins.resize(bin, 0);
        bins[bin]++;
    }

    for (int i = 0; i < bins.size(); i++) {
        if (i > 0) bins[i] +=  bins[i - 1];
        cout << "bin[" << setw(3) << i << "]: " << setw(7) << bins[i] * mass[0] << endl;
    }


    for (int i = 0; i < n_bodies / 4; i++) {
        x[i] += 5.0;
        z[i] += 5.0;
    }

    for (int i = n_bodies / 4; i < n_bodies / 2; i++) {
        x[i] += 5.0;
        y[i] += 5.0;
    }

    for (int i = n_bodies / 2; i < 3 * n_bodies / 4; i++) {
        y[i] += 5.0;
        z[i] += 5.0;
    }

    cout << endl;
    cout << "total mass: " << total_mass << endl;
    cout << "number of bodies: " << n_bodies << endl;
    cout << "mass per body: " << mass[0] << endl;
    cout << endl;

    cout << "Initialized nbody simulation!" << endl;
    cout << "window width: "    << window_width << endl;
    cout << "window height: "   << window_height << endl;
    cout << "window size : "    << window_size << endl;

    /**
     *  Generate the first event -- start a fire at a random (non-water) cell
     */

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("NBody Simulation");

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse_button);
    glutMotionFunc(mouse_motion);
    //glutKeyboardFunc(keyboard);
    //glutIdleFunc(idle);

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glPointSize(2);

    glutMainLoop();
}
