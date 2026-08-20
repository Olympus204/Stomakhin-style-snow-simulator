#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <cassert>

#include "snow/grid.hpp"
#include "snow/particle.hpp"

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
    calculate_volume(grid, particle, 0.5);
    force_calculation(grid, particle, 0.5, 1, 1, 0);
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
    force_calculation(grid, particle, 1, 1, 1, 0);
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
    assert(plastic_change.norm() < 1e-10, "plastic test plastic error");
    assert(Trial_f_change.norm() < 1e-10, "plastic test trial f error");
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
    assert(plastic_change.norm() < 1e-10, "linear velocity field plastic error");
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

int main()
{
    run_test("Spline test", spline_test);
    run_test("N' test", N_dash_test);
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
    run_test("G2P test", G2P_test);
    run_test("FLIP and PIC blend test", FLIP_PIC_test);
    run_test("Outside collider particle test", collision_outside_particle_test);
    run_test("Escaping collision particle test", escaping_collision_particle_test);
    run_test("Sticking particle test", sticking_particle_test);
    run_test("Sliding particle test", sliding_particle_test);
    run_test("Position update test", position_update_test);
    std::cout << "all tests passed";
}
