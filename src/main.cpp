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
    constexpr double grid_spacing = 0.05;
    constexpr double time_step = 0.0005;
    constexpr int substeps_per_frame = 80;
    constexpr int frame_count = 50;

// 80 x 0.0005 is 0.04 seconds, which is 25 fps
    
    double Youngs_modulus{2.5e5};
    double Poisson_ratio{0.2};
    constexpr double hardening = 10.0;
    constexpr double max_compression = 0.015;
    constexpr double max_stretch = 0.0075;
    constexpr double flip_ratio = 0.95;

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
    for (int frame{1}; frame <= frame_count; ++frame)
    {
        StepTimings timings;
        auto frame_start = std::chrono::steady_clock::now();

        for(int substep{0}; substep < substeps_per_frame; ++substep)
        {
            grid = step(particles, colliders, stencils, gravity, time_step, grid_spacing, mu_0, lambda_0, hardening, max_compression, max_stretch, flip_ratio, timings);
        }

        auto frame_end = std::chrono::steady_clock::now();

        double frame_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        std::cout << "Frame " << frame << ": " << frame_time << "ms\n";
        double unnacounted_time = print_timings(timings);
        std::cout << "Unacounted for time: " << frame_time - unnacounted_time << "ms \n" << '\n';
        export_particles(particles, frame);
    }
}
