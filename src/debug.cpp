#include "snow/debug.hpp"

#include <iostream>

double print_timings(StepTimings timings)
{
    double measured =
        timings.stencil +
        timings.p2g +
        timings.forces_constitutive +
        timings.forces_accumulative +
        timings.grid_velocity +
        timings.grid_collisions +
        timings.stress_update +
        timings.cr +
        timings.g2p +
        timings.particle_collisions +
        timings.position_update;

    std::cout << "Stencil: " << timings.stencil << "ms  " << 100.0 * timings.stencil / measured <<"% \n";
    std::cout << "P2G: " << timings.p2g << "ms  " << 100.0 * timings.p2g / measured <<"% \n";
    std::cout << "Forces constitutive: " << timings.forces_constitutive << "ms  " << 100.0 * timings.forces_constitutive / measured <<"% \n";
    std::cout << "Forces accumulative: " << timings.forces_accumulative << "ms  " << 100.0 * timings.forces_accumulative / measured <<"% \n";
    std::cout << "Grid velocity: " << timings.grid_velocity << "ms  " << 100.0 * timings.grid_velocity / measured <<"% \n";
    std::cout << "Grid collisions: " << timings.grid_collisions << "ms  " << 100.0 * timings.grid_collisions / measured <<"% \n";
    std::cout << "Stress update: " << timings.stress_update << "ms  " << 100.0 * timings.stress_update / measured <<"% \n";
    std::cout << "Conjugate Residual: " << timings.cr << "ms  " << 100.0 * timings.cr / measured <<"% \n";
    std::cout << "Total iterations this frame: " << timings.iterations << '\n';
    std::cout << "G2P: " << timings.g2p << "ms  " << 100.0 * timings.g2p / measured <<"% \n";
    std::cout << "Particle collisions: " << timings.particle_collisions << "ms  " << 100.0 * timings.particle_collisions / measured <<"% \n";
    std::cout << "Position update: " << timings.position_update << "ms  " << 100.0 * timings.position_update / measured <<"% \n";
    return measured;
}