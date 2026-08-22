#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <chrono>

#include "snow/grid.hpp"
#include "snow/collision.hpp"
#include "snow/import.hpp"
#include "snow/export.hpp"
#include "snow/debug.hpp"

int main()
{
    double Youngs_modulus{4.8e5};
    double Poisson_ratio{0.2};

    double mu_0 = Youngs_modulus/ (2 * (1 + Poisson_ratio));
    double lambda_0 = (Youngs_modulus * Poisson_ratio)/((1 + Poisson_ratio) * (1 - 2 * Poisson_ratio));

    std::vector<Particle> particles{import_particles("input/groups.csv", "input/particles.csv")};
    std::vector<CollisionBody> colliders;

    std::cout << particles.size() << " particles\n";

    colliders.push_back(make_plane({0,0,0},{0,1,0}, 1));
    Eigen::Vector3d gravity{0,-10,0};
    std::vector<ParticleStencil> stencils(particles.size());
    setup(particles, 0.05);
    export_particles(particles, 0);
    Grid grid;
    for (int frame{1}; frame < 251; ++frame)
    {
        StepTimings timings;
        auto frame_start = std::chrono::steady_clock::now();

        for(int substep{0}; substep < 40; ++substep)
        {
            grid = semi_implicit_step(particles, colliders, stencils, gravity, 0.001, 0.05, mu_0, lambda_0, 10, 0.025, 0.0075, 0.95, 1e-9, 300, 0.5, timings);
        }

        auto frame_end = std::chrono::steady_clock::now();

        double frame_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        std::cout << "Frame " << frame << ": " << frame_time << "ms\n";
        double unnacounted_time = print_timings(timings);
        std::cout << "Unacounted for time: " << frame_time - unnacounted_time << "ms \n";
        export_particles(particles, frame);
        std::cout << '\n';
        std::cout << "Grid nodes: " << grid.size() << '\n' << "Buckets: " << grid.bucket_count() << '\n' << "Load factor: " << grid.load_factor() << '\n' << '\n';
    }
}