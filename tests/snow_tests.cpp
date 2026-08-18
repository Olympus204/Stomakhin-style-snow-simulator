#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <cassert>

#include "snow/grid.hpp"
#include "snow/particle.hpp"

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

    assert(difference(a, 2.0/3.0) < 1e-10);
    assert(difference(b, 1.0/6.0) < 1e-10);
    assert(c == 0);
    assert(d == 0);
    assert(difference(e,f) < 1e-10);
}

void N_dash_test()
{
    double a = N_dash(4,4.3);
    double b = N_dash(4.3,4);
    double c = N_dash(1,1);
    double d = N_dash(0,1);
    double e = N_dash(0,2);
    double f = N_dash(0,0.5);

    assert(difference(a, -1 * b) < 1e-10);
    assert(difference(c, 0) < 1e-10);
    assert(difference(d, -0.5) < 1e-10);
    assert(difference(e, 0) < 1e-10);
    assert(difference(f, -0.625) < 1e-10);
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
    double hardening_coefficient{0.0};

    double epsilon{1e-6};
    Eigen::Matrix3d analytical = constitutive_differential(particle, delta_F, mu_0, lambda_0, hardening_coefficient);

    Particle plus = particle;
    Particle minus = particle;

    plus.F_E = particle.F_E + epsilon * delta_F;
    minus.F_E = particle.F_E - epsilon * delta_F;

    Eigen::Matrix3d P_plus = force_constitutive(plus, mu_0, lambda_0, hardening_coefficient);
    Eigen::Matrix3d P_minus = force_constitutive(minus, mu_0, lambda_0, hardening_coefficient);
    Eigen::Matrix3d numerical = (P_plus - P_minus) / (2.0 * epsilon);
    double error = (analytical - numerical).norm();
    double relative_error = error / std::max(1.0, numerical.norm());
    assert(relative_error < 1e-5);
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
    double hardening_coefficient{0.0};

    double epsilon{1e-6};
    Eigen::Matrix3d analytical = constitutive_differential(particle, delta_F, mu_0, lambda_0, hardening_coefficient);

    Particle plus = particle;
    Particle minus = particle;

    plus.F_E = particle.F_E + epsilon * delta_F;
    minus.F_E = particle.F_E - epsilon * delta_F;

    Eigen::Matrix3d P_plus = force_constitutive(plus, mu_0, lambda_0, hardening_coefficient);
    Eigen::Matrix3d P_minus = force_constitutive(minus, mu_0, lambda_0, hardening_coefficient);
    Eigen::Matrix3d numerical = (P_plus - P_minus) / (2.0 * epsilon);
    double error = (analytical - numerical).norm();
    double relative_error = error / std::max(1.0, numerical.norm());
    assert(relative_error < 1e-5);
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
    double hardening_coefficient{0.0};

    double epsilon{1e-6};
    Eigen::Matrix3d analytical = constitutive_differential(particle, delta_F, mu_0, lambda_0, hardening_coefficient);

    Particle plus = particle;
    Particle minus = particle;

    plus.F_E = particle.F_E + epsilon * delta_F;
    minus.F_E = particle.F_E - epsilon * delta_F;

    Eigen::Matrix3d P_plus = force_constitutive(plus, mu_0, lambda_0, hardening_coefficient);
    Eigen::Matrix3d P_minus = force_constitutive(minus, mu_0, lambda_0, hardening_coefficient);
    Eigen::Matrix3d numerical = (P_plus - P_minus) / (2.0 * epsilon);
    double error = (analytical - numerical).norm();
    double relative_error = error / std::max(1.0, numerical.norm());
    assert(relative_error < 1e-5);
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
    double hardening_coefficient{0.0};

    double epsilon{1e-6};
    Eigen::Matrix3d analytical = constitutive_differential(particle, delta_F, mu_0, lambda_0, hardening_coefficient);

    Particle plus = particle;
    Particle minus = particle;

    plus.F_E = particle.F_E + epsilon * delta_F;
    minus.F_E = particle.F_E - epsilon * delta_F;

    Eigen::Matrix3d P_plus = force_constitutive(plus, mu_0, lambda_0, hardening_coefficient);
    Eigen::Matrix3d P_minus = force_constitutive(minus, mu_0, lambda_0, hardening_coefficient);
    Eigen::Matrix3d numerical = (P_plus - P_minus) / (2.0 * epsilon);
    double error = (analytical - numerical).norm();
    double relative_error = error / std::max(1.0, numerical.norm());
    assert(relative_error < 1e-5);
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
    double hardening_coefficient{0.0};

    double epsilon{1e-6};
    Eigen::Matrix3d analytical = constitutive_differential(particle, delta_F, mu_0, lambda_0, hardening_coefficient);

    Particle plus = particle;
    Particle minus = particle;

    plus.F_E = particle.F_E + epsilon * delta_F;
    minus.F_E = particle.F_E - epsilon * delta_F;

    Eigen::Matrix3d P_plus = force_constitutive(plus, mu_0, lambda_0, hardening_coefficient);
    Eigen::Matrix3d P_minus = force_constitutive(minus, mu_0, lambda_0, hardening_coefficient);
    Eigen::Matrix3d numerical = (P_plus - P_minus) / (2.0 * epsilon);
    double error = (analytical - numerical).norm();
    double relative_error = error / std::max(1.0, numerical.norm());
    assert(relative_error < 1e-5);
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
    assert(difference(total_mass, 2) < 1e-10);
    Eigen::Vector3d momentum_difference = total_momentum - Eigen::Vector3d{6, -4, 1};
    assert(momentum_difference.norm() < 1e-10);
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
    assert(difference(total_mass, 3.5) < 1e-10);
    Eigen::Vector3d momentum_difference = total_momentum - Eigen::Vector3d{4.5,2,4};
    assert(momentum_difference.norm() < 1e-10);
}

void translation_test()
{
    Grid grid_a;
    Grid grid_b;
    Particle particle_a{{2.15,3.35,4.65}, {1,2,3}, 1, 1};
    Particle particle_b{{2.65,3.35,4.65}, {1,2,3}, 1, 1};
    P2G(grid_a, particle_a, 0.5);
    P2G(grid_b, particle_b, 0.5);
    assert(grid_a.size() == grid_b.size());
    for (const auto& [index, node] : grid_a)
    {
        GridIndex translated_grid = {
            index.i + 1,
            index.j,
            index.k
        };
        if (grid_b.find(translated_grid) == grid_b.end())
        {
            assert(false);
        }
        assert(difference(node.mass, grid_b.find(translated_grid)->second.mass) < 1e-10);
        Eigen::Vector3d momentum_difference = node.momentum - grid_b.find(translated_grid)->second.momentum;
        assert(momentum_difference.norm() < 1e-10);
    }
}

void volume_test()
{
    Grid grid;
    Particle particle{{1,2,3}, {1,2,3}, 1, 7};
    P2G(grid, particle, 0.5);
    calculate_volume(grid, particle, 0.5);
    assert(difference(particle.V_p0, 1) < 1e-10);
}

void force_zero_test()
{
    Grid grid;
    Particle particle{{1,2,3}, {1,2,3}, 1, 7};
    P2G(grid, particle, 0.5);
    calculate_volume(grid, particle, 0.5);
    force_calculation(grid, particle, 0.5, 1, 1, 0);
    for (const auto& [index, node] : grid)
    {
        assert(node.force.norm() < 1e-10);
    }
}

void force_non_zero_test()
{
    Grid grid;
    Particle particle{{0.5,0.5,0.5}, {0,0,0}, 1, 1};
    particle.F_E.col(0) << 0.8, 0, 0;
    P2G(grid, particle, 1);
    force_calculation(grid, particle, 1, 1, 1, 0);
    assert(difference(grid.find({0,0,0})->second.force[0], -0.06888020833) < 1e-10);
    assert(difference(grid.find({0,0,0})->second.force[1], -0.02296006944) < 1e-10);
    assert(difference(grid.find({0,0,0})->second.force[2], -0.02296006944) < 1e-10);
    Eigen::Vector3d total_force = Eigen::Vector3d::Zero();
    for (const auto& [index, node] : grid)
    {
        total_force += node.force;
    }
    assert(difference(total_force.norm(), 0) < 1e-10);
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
    assert(difference(grid.find({0,0,0})->second.velocity[0], 3.01) < 1e-10);
    assert(difference(grid.find({0,0,0})->second.velocity[1], 1.8819) < 1e-10);
    assert(difference(grid.find({0,0,0})->second.velocity[2], 0.005) < 1e-10);
    assert(difference(grid.find({0,0,1})->second.velocity[0], 0) < 1e-10);
    assert(difference(grid.find({0,0,1})->second.velocity[1], 0) < 1e-10);
    assert(difference(grid.find({0,0,1})->second.velocity[2], 0) < 1e-10);
}

void collision_outside_grid_test()
{
    Grid grid;
    grid[{0,1,0}].velocity = {2,-3,0};
    std::vector<CollisionBody> colliders;
    colliders.push_back(make_plane({0,0,0}, {0,1,0}, 1));
    grid_collisions(grid, colliders, 0.1);
    Eigen::Vector3d velocity_difference = grid.find({0,1,0})->second.velocity - Eigen::Vector3d{2,-3,0};
    assert(velocity_difference.norm() < 1e-10);
}

void escaping_collision_grid_test()
{
    Grid grid;
    grid[{0,-1,0}].velocity = {2,3,0};
    std::vector<CollisionBody> colliders;
    colliders.push_back(make_plane({0,0,0}, {0,1,0}, 1));
    grid_collisions(grid, colliders, 1);
    Eigen::Vector3d velocity_difference = grid.find({0,-1,0})->second.velocity - Eigen::Vector3d{2,3,0};
    assert(velocity_difference.norm() < 1e-10);
}

void sticking_grid_test()
{
    Grid grid;
    grid[{0,-1,0}].velocity = {0.2,-3,0};
    std::vector<CollisionBody> colliders;
    colliders.push_back(make_plane({0,0,0}, {0,1,0}, 0.6));
    grid_collisions(grid, colliders, 0.1);
    Eigen::Vector3d velocity_difference = grid.find({0,-1,0})->second.velocity - Eigen::Vector3d{0,0,0};
    assert(velocity_difference.norm() < 1e-10);
}

void sliding_grid_test()
{
    Grid grid;
    grid[{0,-1,0}].velocity = {2,-3,0};
    std::vector<CollisionBody> colliders;
    colliders.push_back(make_plane({0,0,0}, {0,1,0}, 0.2));
    grid_collisions(grid, colliders, 0.1);
    Eigen::Vector3d velocity_difference = grid.find({0,-1,0})->second.velocity - Eigen::Vector3d{1.4,0,0};
    assert(velocity_difference.norm() < 1e-10);
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
    assert(elastic_change.norm() < 1e-10);
    assert(plastic_change.norm() < 1e-10);
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
    assert(elastic_change.norm() < 1e-10);
    assert(plastic_change.norm() < 1e-10);
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
    assert(elastic_change.norm() < 1e-10);
    assert(plastic_change.norm() < 1e-10);
    assert(Trial_f_change.norm() < 1e-10);
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
    assert(elastic_change.norm() < 1e-10);
    assert(plastic_change.norm() < 1e-10);
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

    Eigen::Matrix3d delta_F = deformation_differential(grid, particle, 0.1, 0.02);

    assert(delta_F.norm() < 1e-10);
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

    Eigen::Matrix3d delta_F = deformation_differential(grid, particle, 0.1, 0.1);
    Eigen::Matrix3d delta_F_expected = Eigen::Matrix3d::Zero();
    delta_F_expected(0,0) = -0.02;

    assert((delta_F - delta_F_expected).norm() < 1e-10);
}

void Aq_test()
{
    double mu_0 = 0;
    double lambda_0 = 0;
    Grid grid;
    std::vector<Particle> snow;
    snow.push_back({{0,0,0}, {0,0,0}, 1, 1});
    P2G(grid, snow[0], 0.1);
    std::vector<GridNode*> krylov_nodes = index_nodes(grid);
    KrylovVector q;
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        q.push_back({i + 1, 2 * i + 3, i + 2});
    }
    KrylovVector Aq = apply_A(grid, snow, krylov_nodes, q, 0.1, 0.1, mu_0, lambda_0, 0, 0.5);
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        assert(((krylov_nodes[i]->mass * q[i]) - Aq[i]).norm() < 1e-10);
    }
}

void Aq_linear_test()
{
    double mu_0 = 2;
    double lambda_0 = 3;
    Grid grid;
    std::vector<Particle> snow;
    snow.push_back({{0,0,0}, {0,0,0}, 1, 1});
    P2G(grid, snow[0], 0.1);
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
    KrylovVector Aq1 = apply_A(grid, snow, krylov_nodes, q1, 0.1, 0.1, mu_0, lambda_0, 0, 0.5);
    KrylovVector Aq2 = apply_A(grid, snow, krylov_nodes, q2, 0.1, 0.1, mu_0, lambda_0, 0, 0.5);
    KrylovVector Aq12 = apply_A(grid, snow, krylov_nodes, q12, 0.1, 0.1, mu_0, lambda_0, 0, 0.5);
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        assert((Aq12[i] - (Aq1[i] + Aq2[i])).norm() < 1e-10);
    }
}

void Aq_symmetry_test()
{
    double mu_0 = 2;
    double lambda_0 = 3;
    Grid grid;
    std::vector<Particle> snow;
    snow.push_back({{0,0,0}, {0,0,0}, 1, 1});
    P2G(grid, snow[0], 0.1);
    std::vector<GridNode*> krylov_nodes = index_nodes(grid);
    KrylovVector x;
    KrylovVector y;
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        x.push_back({i + 1, 2 * i + 3, i + 2});
        y.push_back({i + 2, 3 * i + 2, 4 * i + 2});
    }
    KrylovVector Ax = apply_A(grid, snow, krylov_nodes, x, 0.1, 0.1, mu_0, lambda_0, 0, 0.5);
    KrylovVector Ay = apply_A(grid, snow, krylov_nodes, y, 0.1, 0.1, mu_0, lambda_0, 0, 0.5);
    double L{0.0};
    double R{0.0};
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        L += x[i].dot(Ay[i]);
        R += Ax[i].dot(y[i]);
    }
    assert(abs(L - R) < 1e-10);
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
    assert(v_difference_from_expected.norm() < 1e-10);
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
    assert(v_difference_from_expected.norm() < 1e-10);
}

void collision_outside_particle_test()
{
    Particle particle{{0,1,0}, {2,-3,0}, 1, 1};
    std::vector<CollisionBody> colliders;
    colliders.push_back(make_plane({0,0,0}, {0,1,0}, 1));
    particle_collisions(particle, colliders);
    Eigen::Vector3d velocity_difference = particle.v_p - Eigen::Vector3d{2,-3,0};
    assert(velocity_difference.norm() < 1e-10);
}

void escaping_collision_particle_test()
{
    Particle particle{{0,-1,0}, {2,3,0}, 1, 1};
    std::vector<CollisionBody> colliders;
    colliders.push_back(make_plane({0,0,0}, {0,1,0}, 1));
    particle_collisions(particle, colliders);
    Eigen::Vector3d velocity_difference = particle.v_p - Eigen::Vector3d{2,3,0};
    assert(velocity_difference.norm() < 1e-10);
}

void sticking_particle_test()
{
    Particle particle{{0,-1,0},{0.2,-3,0}, 1, 1};
    std::vector<CollisionBody> colliders;
    colliders.push_back(make_plane({0,0,0}, {0,1,0}, 0.6));
    particle_collisions(particle, colliders);
    Eigen::Vector3d velocity_difference = particle.v_p - Eigen::Vector3d{0,0,0};
    assert(velocity_difference.norm() < 1e-10);
}

void sliding_particle_test()
{
    Particle particle{{0,-1,0}, {2,-3,0}, 1, 1};
    std::vector<CollisionBody> colliders;
    colliders.push_back(make_plane({0,0,0}, {0,1,0}, 0.2));
    particle_collisions(particle, colliders);
    Eigen::Vector3d velocity_difference = particle.v_p - Eigen::Vector3d{1.4,0,0};
    assert(velocity_difference.norm() < 1e-10);
}

void position_update_test()
{
    Particle particle{{1,2,3}, {2,-1,0.5}, 1, 1};
    position_update(particle, 0.1);
    Eigen::Vector3d difference = particle.x_p - Eigen::Vector3d{1.2,1.9,3.05};
    assert(difference.norm() < 1e-10);
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
    run_test("G2P test", G2P_test);
    run_test("FLIP and PIC blend test", FLIP_PIC_test);
    run_test("Outside collider particle test", collision_outside_particle_test);
    run_test("Escaping collision particle test", escaping_collision_particle_test);
    run_test("Sticking particle test", sticking_particle_test);
    run_test("Sliding particle test", sliding_particle_test);
    run_test("Position update test", position_update_test);
    std::cout << "all tests passed";
}