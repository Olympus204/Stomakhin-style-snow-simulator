#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <stdexcept>
#include <unordered_set>

#include "snow/grid.hpp"
#include "snow/particle.hpp"
#include "snow/export.hpp"

void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void run_test(const char* name, void (*test)())
{
    std::cout << "Running " << name << "... ";
    test();
    std::cout << "passed\n";
}

double difference(double a, double b)
{
    double difference = std::abs(a - b);
    return difference;
}

void spline_test()
{
    double a = N(1,1);
    double b = N(1,2);
    double c = N(1,3);
    double d = N(1,10);
    double e = N(1.5, 0.75);
    double f = N(0.75, 1.5);

    require(difference(a, 2.0/3.0) < 1e-10, "spline error 1");
    require(difference(b, 1.0/6.0) < 1e-10, "spline error 2");
    require(c == 0, "spline error 3");
    require(d == 0, "spline error 4");
    require(difference(e,f) < 1e-10, "spline error 5");
}

void N_dash_test()
{
    double a = N_dash(4,4.3);
    double b = N_dash(4.3,4);
    double c = N_dash(1,1);
    double d = N_dash(0,1);
    double e = N_dash(0,2);
    double f = N_dash(0,0.5);

    require(difference(a, -1 * b) < 1e-10, "N` error 1");
    require(difference(c, 0) < 1e-10, "N` error 2");
    require(difference(d, -0.5) < 1e-10, "N` error 3");
    require(difference(e, 0) < 1e-10, "N` error 4");
    require(difference(f, -0.625) < 1e-10, "N` error 5");
}

void constitutive_differential_test()
{
    Particle particle{{0,0,0},{0,0,0},0,0};

    particle.F_E <<
        1.08,  0.03,  0.01,
       -0.02,  0.96,  0.04,
        0.01, -0.03,  1.03;

    particle.F_P =
        Eigen::Matrix3d::Identity();

    Eigen::Matrix3d delta_F;

    delta_F <<
         0.20, -0.10,  0.03,
         0.05,  0.12, -0.04,
        -0.02,  0.07, -0.08;

    double mu_0{2.0};
    double lambda_0{3.0};

    double epsilon{1e-6};
    std::vector<ParticleLinearisation> linearisations;
    linearisations.push_back(build_lineratisation(particle, mu_0, lambda_0, 0));
    Eigen::Matrix3d analytical = constitutive_differential(delta_F, linearisations[0]);

    Particle plus = particle;
    Particle minus = particle;

    plus.F_E = particle.F_E + epsilon * delta_F;
    ParticleLinearisation plus_linearisation = build_lineratisation(plus, mu_0, lambda_0, 0);
    minus.F_E = particle.F_E - epsilon * delta_F;
    ParticleLinearisation minus_linearisation = build_lineratisation(minus, mu_0, lambda_0, 0);

    Eigen::Matrix3d P_plus = force_constitutive(plus, plus_linearisation);
    Eigen::Matrix3d P_minus = force_constitutive(minus, minus_linearisation);
    Eigen::Matrix3d numerical = (P_plus - P_minus) / (2.0 * epsilon);
    double error = (analytical - numerical).norm();
    double relative_error = error / std::max(1.0, numerical.norm());
    require(relative_error < 1e-5, "constitutive differential error");
}

void constitutive_differential_stretch_test()
{
    Particle particle{{0,0,0},{0,0,0},0,0};

    particle.F_E <<
        1.12,  0,  0,
        0,  0.91,  0,
        0, 0,  1.04;

    particle.F_P =
        Eigen::Matrix3d::Identity();

    Eigen::Matrix3d delta_F;

    delta_F <<
         -0.15, 0.02,  0,
         0.01,  0.08, -0.03,
         0,  0.04, 0.11;

    double mu_0{2.0};
    double lambda_0{3.0};

    double epsilon{1e-6};
    std::vector<ParticleLinearisation> linearisations;
    linearisations.push_back(build_lineratisation(particle, mu_0, lambda_0, 0));
    Eigen::Matrix3d analytical = constitutive_differential(delta_F, linearisations[0]);

    Particle plus = particle;
    Particle minus = particle;

    plus.F_E = particle.F_E + epsilon * delta_F;
    ParticleLinearisation plus_linearisation = build_lineratisation(plus, mu_0, lambda_0, 0);
    minus.F_E = particle.F_E - epsilon * delta_F;
    ParticleLinearisation minus_linearisation = build_lineratisation(minus, mu_0, lambda_0, 0);

    Eigen::Matrix3d P_plus = force_constitutive(plus, plus_linearisation);
    Eigen::Matrix3d P_minus = force_constitutive(minus, minus_linearisation);
    Eigen::Matrix3d numerical = (P_plus - P_minus) / (2.0 * epsilon);
    double error = (analytical - numerical).norm();
    double relative_error = error / std::max(1.0, numerical.norm());
    require(relative_error < 1e-5, "Constitutive differential stretch test error");
}

void constitutive_differential_shear_test()
{
    Particle particle{{0,0,0},{0,0,0},0,0};

    particle.F_E <<
        1.02,  0.18,  -0.04,
        0.03,  0.97,  0.11,
        0.02, -0.06,  1.05;

    particle.F_P =
        Eigen::Matrix3d::Identity();

    Eigen::Matrix3d delta_F;

    delta_F <<
         0.04, -0.22,  0.13,
         0.17, -0.03, -0.09,
        -0.08,  0.11, 0.06;

    double mu_0{2.0};
    double lambda_0{3.0};

    double epsilon{1e-6};
    std::vector<ParticleLinearisation> linearisations;
    linearisations.push_back(build_lineratisation(particle, mu_0, lambda_0, 0));
    Eigen::Matrix3d analytical = constitutive_differential(delta_F, linearisations[0]);

    Particle plus = particle;
    Particle minus = particle;

    plus.F_E = particle.F_E + epsilon * delta_F;
    ParticleLinearisation plus_linearisation = build_lineratisation(plus, mu_0, lambda_0, 0);
    minus.F_E = particle.F_E - epsilon * delta_F;
    ParticleLinearisation minus_linearisation = build_lineratisation(minus, mu_0, lambda_0, 0);

    Eigen::Matrix3d P_plus = force_constitutive(plus, plus_linearisation);
    Eigen::Matrix3d P_minus = force_constitutive(minus, minus_linearisation);
    Eigen::Matrix3d numerical = (P_plus - P_minus) / (2.0 * epsilon);
    double error = (analytical - numerical).norm();
    double relative_error = error / std::max(1.0, numerical.norm());
    require(relative_error < 1e-5, "Constitutive differential shear test error");
}

void constitutive_differential_rigid_test()
{
    Particle particle{{0,0,0},{0,0,0},0,0};

    particle.F_E <<
        0.965, -0.258, 0.03,
        0.259, 0.966, -0.02,
        -0.02, 0.03,  1.01;

    particle.F_P =
        Eigen::Matrix3d::Identity();

    Eigen::Matrix3d delta_F;

    delta_F <<
         0.01, -0.12,  0.04,
         0.09, 0.02, -0.03,
        -0.05, 0.06, 0.01;

    double mu_0{2.0};
    double lambda_0{3.0};

    double epsilon{1e-6};
    std::vector<ParticleLinearisation> linearisations;
    linearisations.push_back(build_lineratisation(particle, mu_0, lambda_0, 0));
    Eigen::Matrix3d analytical = constitutive_differential(delta_F, linearisations[0]);

    Particle plus = particle;
    Particle minus = particle;

    plus.F_E = particle.F_E + epsilon * delta_F;
    ParticleLinearisation plus_linearisation = build_lineratisation(plus, mu_0, lambda_0, 0);
    minus.F_E = particle.F_E - epsilon * delta_F;
    ParticleLinearisation minus_linearisation = build_lineratisation(minus, mu_0, lambda_0, 0);

    Eigen::Matrix3d P_plus = force_constitutive(plus, plus_linearisation);
    Eigen::Matrix3d P_minus = force_constitutive(minus, minus_linearisation);
    Eigen::Matrix3d numerical = (P_plus - P_minus) / (2.0 * epsilon);
    double error = (analytical - numerical).norm();
    double relative_error = error / std::max(1.0, numerical.norm());
    require(relative_error < 1e-5, "Constitutive differential rigid test error");
}

void constitutive_differential_non_trivial_test()
{
    Particle particle{{0,0,0},{0,0,0},0,0};

    particle.F_E <<
        0.94, 0.08, 0.02,
        -0.03, 1.07, 0.05,
        0.01, -0.04, 0.98;

    particle.F_P <<
        0.92, 0, 0,
        0, 0.97, 0,
        0, 0,  1.01;

    Eigen::Matrix3d delta_F;

    delta_F <<
        0.07, -0.05, 0.09,
        -0.02, 0.13, 0.04,
        0.06, -0.08, -0.03;

    double mu_0{2.0};
    double lambda_0{3.0};

    double epsilon{1e-6};
    std::vector<ParticleLinearisation> linearisations;
    linearisations.push_back(build_lineratisation(particle, mu_0, lambda_0, 0));
    Eigen::Matrix3d analytical = constitutive_differential(delta_F, linearisations[0]);

    Particle plus = particle;
    Particle minus = particle;

    plus.F_E = particle.F_E + epsilon * delta_F;
    ParticleLinearisation plus_linearisation = build_lineratisation(plus, mu_0, lambda_0, 0);
    minus.F_E = particle.F_E - epsilon * delta_F;
    ParticleLinearisation minus_linearisation = build_lineratisation(minus, mu_0, lambda_0, 0);

    Eigen::Matrix3d P_plus = force_constitutive(plus, plus_linearisation);
    Eigen::Matrix3d P_minus = force_constitutive(minus, minus_linearisation);
    Eigen::Matrix3d numerical = (P_plus - P_minus) / (2.0 * epsilon);
    double error = (analytical - numerical).norm();
    double relative_error = error / std::max(1.0, numerical.norm());
    require(relative_error < 1e-5, "Constitutive differential non trivil test error");
}

void P2G_test()
{   
    Grid grid;
    Particle particle{{2.1,3.5,4.7}, {3,-2,0.5}, 2, 1};
    P2G(grid, particle, 0.5);
    double total_mass{0.0};
    Eigen::Vector3d total_momentum{0.0,0.0,0.0};
    for (const auto& [index, node] : grid)
    {
        total_mass += node.mass;
        total_momentum += node.momentum;
    }
    require(difference(total_mass, 2) < 1e-10, "P2G mass error");
    Eigen::Vector3d momentum_difference = total_momentum - Eigen::Vector3d{6, -4, 1};
    require(momentum_difference.norm() < 1e-10, "P2G momentum error");
}

void overlap_test()
{
    Grid grid;
    Particle a{{2.1,3.5,4.7}, {3,-2,0.5}, 2, 1};
    Particle b{{2.3,3.7,4.4}, {-1,4,2}, 1.5, 1};
    P2G(grid, a, 0.5);
    P2G(grid, b, 0.5);
    double total_mass{0.0};
    Eigen::Vector3d total_momentum{0.0,0.0,0.0};
    for (const auto& [index, node] : grid)
    {
        total_mass += node.mass;
        total_momentum += node.momentum;
    }
    require(difference(total_mass, 3.5) < 1e-10, "P2G overlap mass error");
    Eigen::Vector3d momentum_difference = total_momentum - Eigen::Vector3d{4.5,2,4};
    require(momentum_difference.norm() < 1e-10, "P2G overlap momentum error");
}

void translation_test()
{
    Grid grid_a;
    Grid grid_b;
    Particle particle_a{{2.15,3.35,4.65}, {1,2,3}, 1, 1};
    Particle particle_b{{2.65,3.35,4.65}, {1,2,3}, 1, 1};
    P2G(grid_a, particle_a, 0.5);
    P2G(grid_b, particle_b, 0.5);
    require(grid_a.size() == grid_b.size(), "translation grid size error");
    for (const auto& [index, node] : grid_a)
    {
        GridIndex translated_grid = {
            index.i + 1,
            index.j,
            index.k
        };
        if (grid_b.find(translated_grid) == grid_b.end())
        {
            require(false, "grid translation error");
        }
        require(difference(node.mass, grid_b.find(translated_grid)->second.mass) < 1e-10, "P2G translation mass error");
        Eigen::Vector3d momentum_difference = node.momentum - grid_b.find(translated_grid)->second.momentum;
        require(momentum_difference.norm() < 1e-10, "P2G translation momentum error");
    }
}

void volume_test()
{
    Grid grid;
    Particle particle{{1,2,3}, {1,2,3}, 1, 7};
    P2G(grid, particle, 0.5);
    calculate_volume(grid, particle, 0.5);
    require(difference(particle.V_p0, 1) < 1e-10, "P2G volume error");
}

void force_zero_test()
{
    Grid grid;
    Particle particle{{1,2,3}, {1,2,3}, 1, 7};
    P2G(grid, particle, 0.5);
    ParticleLinearisation linearisation = build_lineratisation(particle, 1, 1, 0);
    calculate_volume(grid, particle, 0.5);
    force_calculation(grid, particle, linearisation, 0.5);
    for (const auto& [index, node] : grid)
    {
        require(node.force.norm() < 1e-10, "zero force test error");
    }
}

void force_non_zero_test()
{
    Grid grid;
    Particle particle{{0.5,0.5,0.5}, {0,0,0}, 1, 1};
    particle.F_E.col(0) << 0.8, 0, 0;
    P2G(grid, particle, 1);
    ParticleLinearisation linearisation = build_lineratisation(particle, 1, 1, 0);
    force_calculation(grid, particle, linearisation, 1);
    require(difference(grid.find({0,0,0})->second.force[0], -0.06888020833) < 1e-10, "non zero force test error 1");
    require(difference(grid.find({0,0,0})->second.force[1], -0.02296006944) < 1e-10, "non zero force test error 2");
    require(difference(grid.find({0,0,0})->second.force[2], -0.02296006944) < 1e-10, "non zero force test error 3");
    Eigen::Vector3d total_force = Eigen::Vector3d::Zero();
    for (const auto& [index, node] : grid)
    {
        total_force += node.force;
    }
    require(difference(total_force.norm(), 0) < 1e-10, "Non zero force test error");
}

void velocity_test()
{
    Grid grid;
    grid[{0,0,0}].mass = 2;
    grid[{0,0,0}].momentum = {6,4,0};
    grid[{0,0,0}].force = {2,-4,1};
    grid[{0,0,1}].mass = 0;
    grid[{0,0,1}].force = {2,-4,1};
    Eigen::Vector3d gravity = {0,-9.81,0};
    grid_velocity(grid, gravity, 0.01);
    require(difference(grid.find({0,0,0})->second.velocity[0], 3.01) < 1e-10, "velocity test error 1");
    require(difference(grid.find({0,0,0})->second.velocity[1], 1.8819) < 1e-10, "velocity test error 2");
    require(difference(grid.find({0,0,0})->second.velocity[2], 0.005) < 1e-10, "velocity test error 3");
    require(difference(grid.find({0,0,1})->second.velocity[0], 0) < 1e-10, "velocity test error 4");
    require(difference(grid.find({0,0,1})->second.velocity[1], 0) < 1e-10, "velocity test error 5");
    require(difference(grid.find({0,0,1})->second.velocity[2], 0) < 1e-10, "velocity test error 6");
}

void collision_outside_grid_test()
{
    Grid grid;
    grid[{0,1,0}].velocity = {2,-3,0};
    std::vector<CollisionBody> colliders;
    colliders.push_back(make_plane({0,0,0}, {0,1,0}, 1));
    grid_collisions(grid, colliders, 0.1);
    Eigen::Vector3d velocity_difference = grid.find({0,1,0})->second.velocity - Eigen::Vector3d{2,-3,0};
    require(velocity_difference.norm() < 1e-10, "collision outside grid test error");
}

void escaping_collision_grid_test()
{
    Grid grid;
    grid[{0,-1,0}].velocity = {2,3,0};
    std::vector<CollisionBody> colliders;
    colliders.push_back(make_plane({0,0,0}, {0,1,0}, 1));
    grid_collisions(grid, colliders, 1);
    Eigen::Vector3d velocity_difference = grid.find({0,-1,0})->second.velocity - Eigen::Vector3d{2,3,0};
    require(velocity_difference.norm() < 1e-10, "escaping colision grid test error");
}

void sticking_grid_test()
{
    Grid grid;
    grid[{0,-1,0}].velocity = {0.2,-3,0};
    std::vector<CollisionBody> colliders;
    colliders.push_back(make_plane({0,0,0}, {0,1,0}, 0.6));
    grid_collisions(grid, colliders, 0.1);
    Eigen::Vector3d velocity_difference = grid.find({0,-1,0})->second.velocity - Eigen::Vector3d{0,0,0};
    require(velocity_difference.norm() < 1e-10, "sticking grid test error");
}

void sliding_grid_test()
{
    Grid grid;
    grid[{0,-1,0}].velocity = {2,-3,0};
    std::vector<CollisionBody> colliders;
    colliders.push_back(make_plane({0,0,0}, {0,1,0}, 0.2));
    grid_collisions(grid, colliders, 0.1);
    Eigen::Vector3d velocity_difference = grid.find({0,-1,0})->second.velocity - Eigen::Vector3d{1.4,0,0};
    require(velocity_difference.norm() < 1e-10, "sliding grid test error");
}

void rigid_translation_test()
{
    Grid grid;
    Particle particle{{0.15,0.15,0.15}, {0,0,0}, 0, 0};
    for(int i{0}; i < 4; ++i)
    {
        for(int j{0}; j < 4; ++j)
        {
            for (int k{0}; k < 4; ++k)
            {
                GridIndex index{i,j,k};
                grid[index].velocity = {3,-2,1};
            }
        }
    }
    stress_update(grid, particle, 0.1, 0.1, 0.1, 0.1);
    Eigen::Matrix3d elastic_change = particle.F_E - Eigen::Matrix3d::Identity();
    Eigen::Matrix3d plastic_change = particle.F_P - Eigen::Matrix3d::Identity();
    require(elastic_change.norm() < 1e-10, "rigid translation elastic error");
    require(plastic_change.norm() < 1e-10, "rigid translation plastic error");
}

void elastic_test()
{
    Grid grid;
    Particle particle{{0.15,0.15,0.15}, {0,0,0}, 0, 0};
    for(int i{0}; i < 4; ++i)
    {
        for(int j{0}; j < 4; ++j)
        {
            for (int k{0}; k < 4; ++k)
            {
                GridIndex index{i,j,k};
                grid[index].velocity = {3,-2,1};
            }
        }
    }
    particle.F_E = Eigen::Vector3d{0.99,1,1}.asDiagonal();
    stress_update(grid, particle, 0.1, 0.1, 0.025, 0.1);
    const Eigen::Matrix3d target_F_E = Eigen::Vector3d{0.99, 1.0, 1.0}.asDiagonal();
    const Eigen::Matrix3d target_F_P = Eigen::Matrix3d::Identity();

    Eigen::Matrix3d elastic_change = particle.F_E - target_F_E;
    Eigen::Matrix3d plastic_change = particle.F_P - target_F_P;
    require(elastic_change.norm() < 1e-10, "elastic test elastic error");
    require(plastic_change.norm() < 1e-10, "elastic test plastic error");
}

void plastic_test()
{
    Grid grid;
    Particle particle{{0.15,0.15,0.15}, {0,0,0}, 0, 0};
    for(int i{0}; i < 4; ++i)
    {
        for(int j{0}; j < 4; ++j)
        {
            for (int k{0}; k < 4; ++k)
            {
                GridIndex index{i,j,k};
                grid[index].velocity = {3,-2,1};
            }
        }
    }
    particle.F_E = Eigen::Vector3d{0.95,1,1}.asDiagonal();
    stress_update(grid, particle, 0.1, 0.1, 0.025, 0.1);
    const Eigen::Matrix3d target_F_E = Eigen::Vector3d{0.975, 1.0, 1.0}.asDiagonal();
    const Eigen::Matrix3d target_F_P = Eigen::Vector3d{0.95/0.975, 1.0, 1.0}.asDiagonal();
    const Eigen::Matrix3d trial_F = Eigen::Vector3d{0.95, 1.0, 1.0}.asDiagonal();

    Eigen::Matrix3d elastic_change = particle.F_E - target_F_E;
    Eigen::Matrix3d plastic_change = particle.F_P - target_F_P;
    Eigen::Matrix3d Trial_f_change = particle.F_E * particle.F_P - trial_F;
    require(elastic_change.norm() < 1e-10, "plastic test elastic error");
    require(plastic_change.norm() < 1e-10, "plastic test plastic error");
    require(Trial_f_change.norm() < 1e-10, "plastic test trial f error");
}

void linear_velocity_field_test()
{
    Grid grid;
    Particle particle{{0.15,0.15,0.15}, {0,0,0}, 0, 0};
    for(int i{0}; i < 4; ++i)
    {
        for(int j{0}; j < 4; ++j)
        {
            for (int k{0}; k < 4; ++k)
            {
                GridIndex index{i,j,k};
                double x_i = 0.1 * i;
                grid[index].velocity = {-0.1 * x_i,0,0};
            }
        }
    }
    stress_update(grid, particle, 0.1, 0.1, 0.025, 0.1);
    const Eigen::Matrix3d target_F_E = Eigen::Vector3d{0.99, 1.0, 1.0}.asDiagonal();
    const Eigen::Matrix3d target_F_P = Eigen::Matrix3d::Identity();

    Eigen::Matrix3d elastic_change = particle.F_E - target_F_E;
    Eigen::Matrix3d plastic_change = particle.F_P - target_F_P;
    require(elastic_change.norm() < 1e-10, "linear velocity field elastic error");
    require(plastic_change.norm() < 1e-10, "linear velocity field plastic error");
}

void deformation_differential_translation_test()
{
    Grid grid;

    Particle particle{{0.15, 0.15, 0.15}, {0, 0, 0}, 1, 1};

    particle.F_E <<
        1.1,  0.1,  0.0,
        0.0,  0.9,  0.05,
        0.02, 0.0,  1.05;

    for (int i{0}; i < 4; ++i)
    {
        for (int j{0}; j < 4; ++j)
        {
            for (int k{0}; k < 4; ++k)
            {
                GridIndex index{i,j,k};

                grid[index].delta_v =
                    Eigen::Vector3d{3,-2,1};
            }
        }
    }

    ParticleStencil stencil = build_stencil(particle, 0.1);

    for (StencilEntry& entry : stencil)
    {
        entry.node = &grid[entry.index];
    }

    Eigen::Matrix3d delta_F = deformation_differential(particle, stencil, 0.02);

    require(delta_F.norm() < 1e-10, "deformation differential translation test error");
}

void deformation_differential_linear_field_test()
{
    Grid grid;

    Particle particle{{0.15, 0.15, 0.15}, {0, 0, 0}, 1, 1};

    particle.F_E = Eigen::Matrix3d::Identity();

    for (int i{0}; i < 4; ++i)
    {
        for (int j{0}; j < 4; ++j)
        {
            for (int k{0}; k < 4; ++k)
            {
                GridIndex index{i,j,k};

                grid[index].delta_v =
                    Eigen::Vector3d{-0.2 * i * 0.1, 0, 0};
            }
        }
    }
    ParticleStencil stencil = build_stencil(particle, 0.1);

    for (StencilEntry& entry : stencil)
    {
        entry.node = &grid[entry.index];
    }

    Eigen::Matrix3d delta_F = deformation_differential(particle, stencil, 0.1);
    Eigen::Matrix3d delta_F_expected = Eigen::Matrix3d::Zero();
    delta_F_expected(0,0) = -0.02;

    require((delta_F - delta_F_expected).norm() < 1e-10, "deformation differential linear field test error");
}

void Aq_test()
{
    double mu_0 = 0;
    double lambda_0 = 0;
    Grid grid;
    std::vector<Particle> snow;
    snow.push_back({{0,0,0}, {0,0,0}, 1, 1});
    std::vector<ParticleStencil> stencils;
    stencils.push_back(build_stencil(snow[0], 0.1));
    P2G(grid, snow[0], stencils[0], 0.1);
    std::vector<GridNode*> krylov_nodes = index_nodes(grid);
    KrylovVector q;
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        q.push_back({i + 1, 2 * i + 3, i + 2});
    }
    std::vector<ParticleLinearisation> linearisations;
    linearisations.push_back(build_lineratisation(snow[0], mu_0, lambda_0, 0));
    KrylovVector Aq = apply_A(snow, stencils, linearisations, krylov_nodes, q, 0.1, 0.5);
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        require(((krylov_nodes[i]->mass * q[i]) - Aq[i]).norm() < 1e-10, "Aq test error");
    }
}

void Aq_linear_test()
{
    double mu_0 = 2;
    double lambda_0 = 3;
    Grid grid;
    std::vector<Particle> snow;
    snow.push_back({{0,0,0}, {0,0,0}, 1, 1});
    std::vector<ParticleStencil> stencils;
    stencils.push_back(build_stencil(snow[0], 0.1));
    P2G(grid, snow[0], stencils[0], 0.1);
    std::vector<GridNode*> krylov_nodes = index_nodes(grid);
    KrylovVector q1;
    KrylovVector q2;
    KrylovVector q12;
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        q1.push_back({i + 1, 2 * i + 3, i + 2});
        q2.push_back({i + 2, 3 * i + 2, 4 * i + 2});
    }
    for (std::size_t i{0}; i < q1.size(); ++i)
    {
        q12.push_back(q1[i] + q2[i]);
    }
    std::vector<ParticleLinearisation> linearisations;
    linearisations.push_back(build_lineratisation(snow[0], mu_0, lambda_0, 0));
    KrylovVector Aq1 = apply_A(snow, stencils, linearisations, krylov_nodes, q1, 0.1, 0.5);
    KrylovVector Aq2 = apply_A(snow, stencils, linearisations, krylov_nodes, q2, 0.1, 0.5);
    KrylovVector Aq12 = apply_A(snow, stencils, linearisations, krylov_nodes, q12, 0.1, 0.5);
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        require((Aq12[i] - (Aq1[i] + Aq2[i])).norm() < 1e-10, "Aq linear test error");
    }
}

void Aq_symmetry_test()
{
    double mu_0 = 2;
    double lambda_0 = 3;
    Grid grid;
    std::vector<Particle> snow;
    snow.push_back({{0,0,0}, {0,0,0}, 1, 1});
    std::vector<ParticleStencil> stencils;
    stencils.push_back(build_stencil(snow[0], 0.1));
    P2G(grid, snow[0], stencils[0], 0.1);
    std::vector<GridNode*> krylov_nodes = index_nodes(grid);
    KrylovVector x;
    KrylovVector y;
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        x.push_back({i + 1, 2 * i + 3, i + 2});
        y.push_back({i + 2, 3 * i + 2, 4 * i + 2});
    }
    std::vector<ParticleLinearisation> linearisations;
    linearisations.push_back(build_lineratisation(snow[0], mu_0, lambda_0, 0));
    KrylovVector Ax = apply_A(snow, stencils, linearisations, krylov_nodes, x, 0.1, 0.5);
    KrylovVector Ay = apply_A(snow, stencils, linearisations, krylov_nodes, y, 0.1, 0.5);
    double L{0.0};
    double R{0.0};
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        L += x[i].dot(Ay[i]);
        R += Ax[i].dot(y[i]);
    }
    require(abs(L - R) < 1e-10, "Aq symmetry test error");
}

void mass_scaling_test()
{
    double mu_0 = 0;
    double lambda_0 = 0;
    Grid grid;
    std::vector<Particle> snow;
    snow.push_back({{0,0,0}, {0,0,0}, 1, 1});
    std::vector<ParticleStencil> stencils;
    stencils.push_back(build_stencil(snow[0], 0.1));
    P2G(grid, snow[0], stencils[0], 0.1);
    std::vector<GridNode*> krylov_nodes = index_nodes(grid);
    KrylovVector q;
    std::vector<double> mass_scaling(krylov_nodes.size());
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        q.push_back({i + 1, 2 * i + 3, i + 2});
        mass_scaling[i] = 1 / std::sqrt(krylov_nodes[i]->mass);
    }
    std::vector<ParticleLinearisation> linearisations;
    linearisations.push_back(build_lineratisation(snow[0], mu_0, lambda_0, 0));
    KrylovVector Bq = apply_mass_scaled_A(snow, stencils, linearisations, krylov_nodes, q, mass_scaling, 0.1, 0);
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        require((q[i] - Bq[i]).norm() < 1e-10, "mass scaling test error");
    }
}

void mass_scaling_symmetry_test()
{
    double mu_0 = 2;
    double lambda_0 = 3;
    Grid grid;
    std::vector<Particle> snow;
    snow.push_back({{0,0,0}, {0,0,0}, 1, 1});
    std::vector<ParticleStencil> stencils;
    stencils.push_back(build_stencil(snow[0], 0.1));
    P2G(grid, snow[0], stencils[0], 0.1);
    std::vector<GridNode*> krylov_nodes = index_nodes(grid);
    KrylovVector x;
    KrylovVector y;
    std::vector<double> mass_scaling(krylov_nodes.size());
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        x.push_back({i + 1, 2 * i + 3, i + 2});
        y.push_back({i + 2, 3 * i + 2, 4 * i + 2});
        mass_scaling[i] = 1 / std::sqrt(krylov_nodes[i]->mass);
    }
    std::vector<ParticleLinearisation> linearisations;
    linearisations.push_back(build_lineratisation(snow[0], mu_0, lambda_0, 0));
    KrylovVector Bx = apply_mass_scaled_A(snow, stencils, linearisations, krylov_nodes, x, mass_scaling, 0.1, 0.5);
    KrylovVector By = apply_mass_scaled_A(snow, stencils, linearisations, krylov_nodes, y, mass_scaling, 0.1, 0.5);
    double L{0.0};
    double R{0.0};
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        L += x[i].dot(By[i]);
        R += Bx[i].dot(y[i]);
    }
    require((abs(L - R) / std::max({1.0, abs(L), abs(R)})) < 1e-10, "Mass scaling symmetry test error");
}

void CR_initial_residual_zero_test()
{
    double mu_0 = 0;
    double lambda_0 = 0;
    Grid grid;
    std::vector<Particle> snow;
    snow.push_back({{0,0,0}, {0,0,0}, 1, 0});
    std::vector<ParticleStencil> stencils;
    stencils.push_back(build_stencil(snow[0], 0.1));
    P2G(grid, snow[0], stencils[0], 0.1);
    for (auto& [index, node] : grid)
    {
        node.mass = 1;
        node.velocity = {0,1,0};
    }
    std::vector<GridNode*> krylov_nodes = index_nodes(grid);
    std::vector<double> mass_scaling(krylov_nodes.size());
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        mass_scaling[i] = 1 / std::sqrt(krylov_nodes[i]->mass);
    }
    std::vector<ParticleLinearisation> linearisations;
    linearisations.push_back(build_lineratisation(snow[0], mu_0, lambda_0, 0));
    Conjugate_residual(grid, snow, stencils, linearisations, krylov_nodes, mass_scaling, 0.1, 0.1, 0.5, 1e-3, 50);
    for (auto& [index, node] : grid)
    {
        require((node.velocity - Eigen::Vector3d{0,1,0}).norm() < 1e-10, "CR initial residual zero error");
    }
}

void CR_non_trivial_test()
{
    double mu_0 = 2.0;
    double lambda_0 = 3.0;
    Grid grid;
    std::vector<Particle> snow;
    snow.push_back({{0.15,0.15,0.15}, {0,0,0}, 1, 0.01});
    snow[0].F_E << 
        1.05, 0, 0,
        0, 0.97, 0,
        0, 0, 1.02;
    std::vector<ParticleStencil> stencils;
    stencils.push_back(build_stencil(snow[0], 0.1));
    P2G(grid, snow[0], stencils[0], 0.1);
    std::vector<GridNode*> krylov_nodes = index_nodes(grid);
    std::vector<double> mass_scaling(krylov_nodes.size());
    KrylovVector b;
    for (auto& [index, node] : grid)
    {
        node.mass = 1;
        node.velocity = {0.3 * 0.1 * index.i, -0.2 * 0.1 * index.j, 0.1 * 0.1 * index.k};
    }
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        const GridNode& node = *krylov_nodes[i];
        b.push_back(node.velocity);
        mass_scaling[i] = 1 / std::sqrt(krylov_nodes[i]->mass);
    }
    std::vector<ParticleLinearisation> linearisations;
    linearisations.push_back(build_lineratisation(snow[0], mu_0, lambda_0, 0));
    double R_squared_zero{0.0};
    KrylovVector Ab = apply_A(snow, stencils, linearisations, krylov_nodes, b, 0.1, 0.5);
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        R_squared_zero += (b[i] - Ab[i]).dot(b[i] - Ab[i]);
    }
    require(R_squared_zero > 1e-10, "CR non trivial test error 1");

    Conjugate_residual(grid, snow, stencils, linearisations, krylov_nodes, mass_scaling, 0.1, 0.1, 0.5, 1e-12, 200);
    KrylovVector x;
    double R_squared{0.0};
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        const GridNode& node = *krylov_nodes[i];
        x.push_back(node.velocity);
    }
    KrylovVector Ax = apply_A(snow, stencils, linearisations, krylov_nodes, x, 0.1, 0.5);
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        R_squared += (b[i] - Ax[i]).dot(b[i] - Ax[i]);
    }
    require(R_squared < 1e-12, "CR non trivial error 2");
}

void CR_collision_test()
{
    double mu_0 = 2.0;
    double lambda_0 = 3.0;
    Grid grid;
    std::vector<Particle> snow;
    snow.push_back({{0.15,0.15,0.15}, {0,0,0}, 1, 0.01});
    snow[0].F_E << 
        1.05, 0, 0,
        0, 0.97, 0,
        0, 0, 1.02;
    std::vector<ParticleStencil> stencils;
    stencils.push_back(build_stencil(snow[0], 0.1));
    P2G(grid, snow[0], stencils[0], 0.1);
    std::vector<GridNode*> krylov_nodes = index_nodes(grid);
    std::vector<double> mass_scaling(krylov_nodes.size());
    KrylovVector b;
    for (auto& [index, node] : grid)
    {
        node.mass = 1;
        node.velocity = {0.3 * 0.1 * index.i, -0.2 * 0.1 * index.j, 0.1 * 0.1 * index.k};
    }
    grid[{0,0,0}].colision_constrained = true;
    grid[{0,0,0}].velocity = {0.4,-0.2,0.7};
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        const GridNode& node = *krylov_nodes[i];
        b.push_back(node.velocity);
    }
    std::vector<ParticleLinearisation> linearisations;
    linearisations.push_back(build_lineratisation(snow[0], mu_0, lambda_0, 0));
    double R_squared_zero{0.0};
    KrylovVector Ab = apply_A(snow, stencils, linearisations, krylov_nodes, b, 0.1, 0.5);
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        mass_scaling[i] = 1 / std::sqrt(krylov_nodes[i]->mass);
        const GridNode& node = *krylov_nodes[i];
        if (node.colision_constrained == false)
        {
            R_squared_zero += (b[i] - Ab[i]).dot(b[i] - Ab[i]);
        }
    }
    require(R_squared_zero > 1e-10, "CR collision test 1");

    Conjugate_residual(grid, snow, stencils, linearisations, krylov_nodes, mass_scaling, 0.1, 0.1, 0.5, 1e-12, 200);
    require((grid[{0,0,0}].velocity - Eigen::Vector3d{0.4,-0.2,0.7}).norm() < 1e-10, "CR collision test 2");

    KrylovVector x;
    double R_squared{0.0};
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        const GridNode& node = *krylov_nodes[i];
        x.push_back(node.velocity);
    }
    KrylovVector Ax = apply_A(snow, stencils, linearisations, krylov_nodes, x, 0.1, 0.5);
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        const GridNode& node = *krylov_nodes[i];
        if (node.colision_constrained == false)
        {
            R_squared += (b[i] - Ax[i]).dot(b[i] - Ax[i]);
        }
    }
    require(R_squared < 1e-12, "CR collision test 3");

}

void G2P_test()
{
    Grid grid;
    Particle particle{{0.15,0.15,0.15}, {2,3,4}, 0, 0};
    P2G(grid, particle, 0.1);
    for (auto& [index, node] : grid)
    {
        node.mass = 1;
        node.momentum = {2,3,4};
        node.velocity = {2,2,4};
    }
    G2P(grid, particle, 0.1, 0.95);
    Eigen::Vector3d v_difference_from_expected = particle.v_p - Eigen::Vector3d{2,2,4};
    require(v_difference_from_expected.norm() < 1e-10, "G2P velocity error");
}

void FLIP_PIC_test()
{
    Grid grid;
    Particle particle{{0.15,0.15,0.15}, {10,0,0}, 0, 0};
    P2G(grid, particle, 0.1);
    for (auto& [index, node] : grid)
    {
        node.mass = 1;
        node.momentum = {1,0,0};
        node.velocity = {3,0,0};
    }
    G2P(grid, particle, 0.1, 0.95);
    Eigen::Vector3d v_difference_from_expected = particle.v_p - Eigen::Vector3d{11.55,0,0};
    require(v_difference_from_expected.norm() < 1e-10, "FLIP PIC velocity error");
}

void collision_outside_particle_test()
{
    Particle particle{{0,1,0}, {2,-3,0}, 1, 1};
    std::vector<CollisionBody> colliders;
    colliders.push_back(make_plane({0,0,0}, {0,1,0}, 1));
    particle_collisions(particle, colliders);
    Eigen::Vector3d velocity_difference = particle.v_p - Eigen::Vector3d{2,-3,0};
    require(velocity_difference.norm() < 1e-10, "collision outside particle error");
}

void escaping_collision_particle_test()
{
    Particle particle{{0,-1,0}, {2,3,0}, 1, 1};
    std::vector<CollisionBody> colliders;
    colliders.push_back(make_plane({0,0,0}, {0,1,0}, 1));
    particle_collisions(particle, colliders);
    Eigen::Vector3d velocity_difference = particle.v_p - Eigen::Vector3d{2,3,0};
    require(velocity_difference.norm() < 1e-10, "escaping collision particle error");
}

void sticking_particle_test()
{
    Particle particle{{0,-1,0},{0.2,-3,0}, 1, 1};
    std::vector<CollisionBody> colliders;
    colliders.push_back(make_plane({0,0,0}, {0,1,0}, 0.6));
    particle_collisions(particle, colliders);
    Eigen::Vector3d velocity_difference = particle.v_p - Eigen::Vector3d{0,0,0};
    require(velocity_difference.norm() < 1e-10, "sticking particle test error");
}

void sliding_particle_test()
{
    Particle particle{{0,-1,0}, {2,-3,0}, 1, 1};
    std::vector<CollisionBody> colliders;
    colliders.push_back(make_plane({0,0,0}, {0,1,0}, 0.2));
    particle_collisions(particle, colliders);
    Eigen::Vector3d velocity_difference = particle.v_p - Eigen::Vector3d{1.4,0,0};
    require(velocity_difference.norm() < 1e-10, "sliding particle test error");
}

void position_update_test()
{
    Particle particle{{1,2,3}, {2,-1,0.5}, 1, 1};
    position_update(particle, 0.1);
    Eigen::Vector3d difference = particle.x_p - Eigen::Vector3d{1.2,1.9,3.05};
    require(difference.norm() < 1e-10, "position update error");
}

void particle_to_mesh_grid_test()
{
    MeshParticle particle{{1,1,1}, 1};
    ReconstructionGrid grid = construct_grid({0,0,0}, {2,2,2}, 0.1);
    std::unordered_set<GridIndex, GridIndexHash> active_cubes;
    particle_to_grid(particle, grid, active_cubes);
    double recovered_volume{0.0};
    for (std::size_t i{0}; i < grid.values.size(); ++i)
    {
        recovered_volume += grid.values[i];
    }
    recovered_volume *= grid.spacing * grid.spacing * grid.spacing;
    require(abs(recovered_volume - particle.volume) < 1e-10, "particle to mesh grid test error");
}

void multiple_particle_to_mesh_grid_test()
{
    MeshParticle particle1{{0.83,1.17,0.94}, 1};
    MeshParticle particle2{{1.26,0.72,1.31}, 0.4};
    ReconstructionGrid grid = construct_grid({0,0,0}, {2,2,2}, 0.1);
    std::unordered_set<GridIndex, GridIndexHash> active_cubes;
    particle_to_grid(particle1, grid, active_cubes);
    particle_to_grid(particle2, grid, active_cubes);
    double recovered_volume{0.0};
    for (std::size_t i{0}; i < grid.values.size(); ++i)
    {
        recovered_volume += grid.values[i];
    }
    recovered_volume *= grid.spacing * grid.spacing * grid.spacing;
    require(abs(recovered_volume - 1.4) < 1e-10, "multiple particle to mesh grid test error");
}

void cube_construction_test()
{
    ReconstructionGrid grid = construct_grid({0,0,0}, {1,1,1}, 1);
    grid.values = {0,1,2,3,4,5,6,7};
    auto cube_optional = construct_cube(grid, {0,0,0});
    require(cube_optional.has_value(), "cube construction failed 1");
    const auto& cube = *cube_optional;
    require(abs(cube[0].value - 0) < 1e-10, "cube construction error 1");
    require(abs(cube[1].value - 1) < 1e-10, "cube construction error 2");
    require(abs(cube[2].value - 3) < 1e-10, "cube construction error 3");
    require(abs(cube[3].value - 2) < 1e-10, "cube construction error 4");
    require(abs(cube[4].value - 4) < 1e-10, "cube construction error 5");
    require(abs(cube[5].value - 5) < 1e-10, "cube construction error 6");
    require(abs(cube[6].value - 7) < 1e-10, "cube construction error 7");
    require(abs(cube[7].value - 6) < 1e-10, "cube construction error 8");
}

void cube_interpolation_test()
{
    ReconstructionGrid grid = construct_grid( {0,0,0}, {1,1,1}, 1);
    grid.values = {0,1,1,1, 1,1,1,1};
    auto cube_optional = construct_cube( grid, {0,0,0});
    require(cube_optional.has_value(), "cube construction failed 2");
    const auto& cube = *cube_optional;
    Mesh mesh;
    std::unordered_map<EdgeKey, std::size_t, EdgeKeyHash> vertex_cache;
    triangulate_cube(mesh, vertex_cache, grid, cube, 0.5);
    require(mesh.faces.size() == 1, "Triangulation error 1");
    require(mesh.vertices.size() == 3, "Triangulation error 2");
    const auto& face = mesh.faces[0];
    const Eigen::Vector3d& v0 = mesh.vertices[face[0]];
    const Eigen::Vector3d& v1 = mesh.vertices[face[1]];
    const Eigen::Vector3d& v2 = mesh.vertices[face[2]];
    require(difference(v0[0], 0.5) < 1e-10, "Triangulation error 3");
    require(difference(v0[1], 0.0) < 1e-10, "Triangulation error 4");
    require(difference(v0[2], 0.0) < 1e-10, "Triangulation error 5");
    require(difference(v1[0], 0.0) < 1e-10, "Triangulation error 6");
    require(difference(v1[1], 0.0) < 1e-10, "Triangulation error 7");
    require(difference(v1[2], 0.5) < 1e-10, "Triangulation error 8");
    require(difference(v2[0], 0.0) < 1e-10, "Triangulation error 9");
    require(difference(v2[1], 0.5) < 1e-10, "Triangulation error 10");
    require(difference(v2[2], 0.0) < 1e-10, "Triangulation error 11");
}

int main()
{
    run_test("Spline test", spline_test);
    run_test("N' test", N_dash_test);
    run_test("constitutive differential test", constitutive_differential_test);
    run_test("constitutive differential stretch test", constitutive_differential_stretch_test);
    run_test("constitutive differential shear test", constitutive_differential_shear_test);
    run_test("constitutive differential rigid test", constitutive_differential_rigid_test);
    run_test("constitutive differential non-trivial test", constitutive_differential_non_trivial_test);
    run_test("P2G test", P2G_test);
    run_test("Overlap test", overlap_test);
    run_test("Translation test", translation_test);
    run_test("Volume test",volume_test);
    run_test("Zero force test", force_zero_test);
    run_test("Non-zero force test", force_non_zero_test);
    run_test("Velocity test", velocity_test);
    run_test("Outside collider grid test", collision_outside_grid_test);
    run_test("Escaping collision grid test", escaping_collision_grid_test);
    run_test("Sticking grid test", sticking_grid_test);
    run_test("Sliding grid test", sliding_grid_test);
    run_test("Rigid translation stress test", rigid_translation_test);
    run_test("Elastic stress test", elastic_test);
    run_test("Plastic stress test", plastic_test);
    run_test("Linear velocity field test", linear_velocity_field_test);
    run_test("deformation differential translation test", deformation_differential_translation_test);
    run_test("deformation differential linear_field test", deformation_differential_linear_field_test);
    run_test("Aq test", Aq_test);
    run_test("Aq linear test", Aq_linear_test);
    run_test("Aq symmetry test", Aq_symmetry_test);
    run_test("Mass scaling test", mass_scaling_test);
    run_test("Mass scaling symmetry test", mass_scaling_symmetry_test);
    run_test("CR initial residual zero test", CR_initial_residual_zero_test);
    run_test("CR non trivial test", CR_non_trivial_test);
    run_test("CR collision test", CR_collision_test);
    run_test("G2P test", G2P_test);
    run_test("FLIP and PIC blend test", FLIP_PIC_test);
    run_test("Outside collider particle test", collision_outside_particle_test);
    run_test("Escaping collision particle test", escaping_collision_particle_test);
    run_test("Sticking particle test", sticking_particle_test);
    run_test("Sliding particle test", sliding_particle_test);
    run_test("Position update test", position_update_test);
    run_test("Particle to mesh grid test", particle_to_mesh_grid_test);
    run_test("Multiple particles to mesh grid test", multiple_particle_to_mesh_grid_test);
    run_test("Cube construction test", cube_construction_test);
    run_test("Cube triangularisation test", cube_interpolation_test);
    std::cout << "all tests passed";
}