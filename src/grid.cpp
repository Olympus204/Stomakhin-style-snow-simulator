#include "snow/grid.hpp"
#include "snow/particle.hpp"
#include "snow/collision.hpp"
#include "snow/debug.hpp"

#include <cmath>
#include <unordered_map>
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <iostream>

namespace
{
    double signum(double x)
    {
        if (x > 0.0) return 1.0;
        if (x < 0.0) return -1.0;
        return 0.0;
    }
}


double N(const double& a, const double& b)
{
    double distance = std::abs(a - b);

    if (distance >= 0.0 && distance < 1.0)
    {
        double value = 0.5 * distance * distance * distance - distance * distance + 2.0 / 3.0;
        return value;
    }
    else if (distance >= 1.0 && distance < 2.0)
    {
        double value = -1.0 / 6.0 * distance * distance * distance + distance * distance - 2.0 * distance + 4.0 / 3.0;
        return value;
    }

    return 0.0;
}

double N_dash(const double& grid_node, const double& particle_coordinate)
{
    double distance = std::abs(grid_node - particle_coordinate);

    if (distance >= 0.0 && distance < 1.0)
    {
        double value = 1.5 * distance * distance - 2 * distance;
        return value * signum(particle_coordinate - grid_node);
    }
    else if (distance >= 1.0 && distance < 2.0)
    {
        double value = -1.0 / 2.0 * distance * distance + 2 * distance - 2.0;
        return value * signum(particle_coordinate - grid_node);
    }

    return 0.0;
}

ParticleLinearisation build_lineratisation(const Particle& particle, double mu_0, double lambda_0, double hardening_coefficient)
{
    ParticleLinearisation linearisation;
    Eigen::Matrix3d F = particle.F_E;
    linearisation.J = F.determinant();
    linearisation.F_transpose_inverse = F.transpose().inverse();
    linearisation.C = linearisation.J * F.inverse().transpose();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    linearisation.U = svd.matrixU();
    linearisation.V = svd.matrixV();
    linearisation.sigma = svd.singularValues();
    double h = exp(hardening_coefficient * (1 - particle.F_P.determinant()));
    linearisation.mu = mu_0 * h;
    linearisation.lambda = lambda_0 * h;
    return linearisation;
}

Eigen::Matrix3d constitutive_differential(const Eigen::Matrix3d& delta_f, ParticleLinearisation& linearisation)
{
    double delta_J = linearisation.C.cwiseProduct(delta_f).sum();
    Eigen::Matrix3d delta_C = delta_J * linearisation.F_transpose_inverse - linearisation.J * linearisation.F_transpose_inverse * delta_f.transpose() * linearisation.F_transpose_inverse;
    Eigen::Matrix3d B = linearisation.U.transpose() * delta_f * linearisation.V;
    Eigen::Matrix3d omega = Eigen::Matrix3d::Zero();
    double omega_01 = (B(0,1) - B(1,0)) / (linearisation.sigma(0) + linearisation.sigma(1));
    double omega_02 = (B(0,2) - B(2,0)) / (linearisation.sigma(0) + linearisation.sigma(2));
    double omega_12 = (B(1,2) - B(2,1)) / (linearisation.sigma(1) + linearisation.sigma(2));
    omega(0,1) = omega_01;
    omega(1,0) = -omega_01;
    omega(0,2) = omega_02;
    omega(2,0) = -omega_02;
    omega(1,2) = omega_12;
    omega(2,1) = -omega_12;
    Eigen::Matrix3d delta_R = linearisation.U * omega * linearisation.V.transpose();
    Eigen::Matrix3d delta_P = 2 * linearisation.mu * delta_f - 2 * linearisation.mu * delta_R + linearisation.lambda * linearisation.C * delta_J + linearisation.lambda * (linearisation.J - 1) * delta_C;
    return delta_P;
}

ParticleStencil build_stencil(const Particle& particle, double grid_spacing)
{
    ParticleStencil stencil;

    Eigen::Vector3d grid_position =
        particle.x_p / grid_spacing;

    Eigen::Vector3i base =grid_position.array().floor().cast<int>() - Eigen::Vector3i::Ones().array();

    std::array<double, 4> w_x;
    std::array<double, 4> w_y;
    std::array<double, 4> w_z;

    std::array<double, 4> d_x;
    std::array<double, 4> d_y;
    std::array<double, 4> d_z;
    for (int i{0}; i < 4; ++i)
    {
        w_x[i] = N(base[0] + i, grid_position[0]);

        d_x[i] = N_dash(base[0] + i, grid_position[0]) / grid_spacing;
    }
    for (int j{0}; j < 4; ++j)
    {
        w_y[j] = N(base[1] + j, grid_position[1]);

        d_y[j] = N_dash(base[1] + j, grid_position[1]) / grid_spacing;
    }
    for (int k{0}; k < 4; ++k)
    {
        w_z[k] = N(base[2] + k, grid_position[2]);

        d_z[k] = N_dash(base[2] + k, grid_position[2]) / grid_spacing;
    }

    std::size_t n{0};

    for (int i{0}; i < 4; ++i)
    {
        for (int j{0}; j < 4; ++j)
        {
            for (int k{0}; k < 4; ++k)
            {
                StencilEntry& entry = stencil[n];

                entry.index = GridIndex{base[0] + i, base[1] + j, base[2] + k};

                entry.weight = w_x[i] * w_y[j] * w_z[k];

                entry.gradient = Eigen::Vector3d{d_x[i] * w_y[j] * w_z[k], w_x[i] * d_y[j] * w_z[k], w_x[i] * w_y[j] * d_z[k]};

                ++n;
            }
        }
    }

    return stencil;
}

void P2G(Grid& grid, const Particle& particle, ParticleStencil& stencil, double grid_spacing)
{
    const Eigen::Vector3d particle_momentum = particle.m_p * particle.v_p;
    for (StencilEntry& entry : stencil)
    {
        GridNode & node = grid[entry.index];
        entry.node = &node;
        node.mass += entry.weight * particle.m_p;
        node.momentum += entry.weight * particle_momentum;
    }
}

void P2G(Grid& grid, const Particle& particle, double grid_spacing)
{
    ParticleStencil stencil = build_stencil( particle, grid_spacing);

    P2G(grid, particle, stencil, grid_spacing);
}

std::vector<GridNode*> index_nodes(Grid& grid)
{
    std::vector<GridIndex> indices;
    indices.reserve(grid.size());
    for (const auto& [index, node] : grid)
    {
        indices.push_back(index);
    }
    std::sort(indices.begin(), indices.end());
    std::vector<GridNode*> krylov_nodes;
    krylov_nodes.reserve(indices.size());
    for (const GridIndex& index : indices)
    {
        auto it = grid.find(index);
        if (it != grid.end())
        {
            krylov_nodes.push_back(&it->second);
        }
    }
    return krylov_nodes;
}

void calculate_volume (Grid& grid, Particle& particle, double grid_spacing)
{
    Eigen::Vector3d grid_position = particle.x_p / grid_spacing;
    Eigen::Vector3i base = grid_position.array().floor().cast<int>() - Eigen::Vector3i::Ones().array();
    double particle_density{0.0};
    for (int i{0}; i < 4; ++i)
    {
        for (int j{0}; j < 4; ++j)
        {
            for (int k{0}; k < 4; ++k)
            {
                GridIndex index{i + base[0], j + base[1], k + base[2]};
                const GridNode& node = grid[index];
                double w_x = N(index.i, grid_position[0]);
                double w_y = N(index.j, grid_position[1]);
                double w_z = N(index.k, grid_position[2]);
                double w_p = w_x * w_y * w_z;   
                particle_density += (node.mass * w_p) / pow(grid_spacing, 3.0);
            }
        }
    }
    particle.V_p0 = particle.m_p / particle_density;
}

Eigen::Matrix3d force_constitutive(const Particle& particle, ParticleLinearisation& linearistation)
{
    Eigen::Matrix3d R_E = linearistation.U * linearistation.V.transpose();
    Eigen::Matrix3d dphidF_E = 2 * linearistation.mu * (particle.F_E - R_E) + linearistation.lambda * (linearistation.J - 1) * linearistation.J * particle.F_E.transpose().inverse();
    return dphidF_E;
}

void force_grid_accumulation(Grid& grid, const Eigen::Matrix3d& dphidF_E, const Particle& particle, const ParticleStencil& stencil)
{
    Eigen::Matrix3d force_matrix = -particle.V_p0 * dphidF_E * particle.F_E.transpose();
    for (const StencilEntry& entry : stencil)
    {   
        assert(entry.node != nullptr);
        entry.node->force += force_matrix * entry.gradient;
    }
}

void force_calculation(Grid& grid,const Particle& particle, ParticleLinearisation& linearisation, double grid_spacing)
{
    Eigen::Matrix3d stress = force_constitutive(particle, linearisation);
    ParticleStencil stencil = build_stencil(particle, grid_spacing);
    for (StencilEntry& entry : stencil)
    {
        entry.node = &grid[entry.index];
    }
    force_grid_accumulation(grid,stress,particle, stencil);
}

void grid_velocity(Grid& grid, const Eigen::Vector3d gravity, const double time_step)
{
    for (auto& [index, node] : grid)
    {
        if (node.mass > 0)
        {
            node.velocity = (node.momentum / node.mass) + time_step * ((node.force / node.mass) + gravity);
        }       
    }
}

Eigen::Vector3d collision_helper(const Eigen::Vector3d& position, const Eigen::Vector3d& velocity, const CollisionBody& body, bool& constrained)
{
    double signed_distance = body.phi(position);
    if (signed_distance <= 0)
    {
        Eigen::Vector3d n = body.normal(position);
        Eigen::Vector3d v_c = body.velocity(position);
        Eigen::Vector3d v_rel = velocity - v_c;
        double v_n = v_rel.dot(n);
        if (v_n < 0)
        {
            constrained = true;
            Eigen::Vector3d v_t = v_rel - v_n * n;
            Eigen::Vector3d v_dash_rel{};
            if (v_t.norm() <= -1 * body.friction * v_n)
            {
                v_dash_rel = Eigen::Vector3d::Zero();
            }
            else
            {
                v_dash_rel = v_t + body.friction * v_n * (v_t / v_t.norm());
            }
            return v_dash_rel + v_c;
        }
    }
    return velocity;
}

void grid_collisions(Grid& grid, const std::vector<CollisionBody>& collisions, double grid_spacing)
{
    for (auto& [index, node] : grid)
    {
        node.colision_constrained = false;
        Eigen::Vector3d real_position = {grid_spacing * index.i, grid_spacing * index.j, grid_spacing * index.k};
        for (const CollisionBody& body : collisions)
        {   
            node.velocity = collision_helper(real_position, node.velocity, body, node.colision_constrained);
        }
    }
}

void stress_update(const Grid& grid, Particle& particle, double grid_spacing, double time_step, double max_compression, double max_stretch)
{
    Eigen::Vector3d grid_position = particle.x_p / grid_spacing;
    Eigen::Vector3i base = grid_position.array().floor().cast<int>() - Eigen::Vector3i::Ones().array();
    Eigen::Matrix3d velocity_gradient = Eigen::Matrix3d::Zero();
    for (int i{0}; i < 4; ++i)
    {
        for (int j{0}; j < 4; ++j)
        {
            for (int k{0}; k < 4; ++k)
            {
                GridIndex index{i + base[0], j + base[1], k + base[2]};
                auto it = grid.find(index);
                if (it == grid.end())
                {
                    continue;
                }
                const GridNode& node = it->second;
                Eigen::Vector3d basis_gradient = Eigen::Vector3d::Zero();
                double N_x = N(index.i, grid_position[0]);
                double N_y = N(index.j, grid_position[1]);
                double N_z = N(index.k, grid_position[2]);
                double N_dash_x = N_dash(index.i, grid_position[0]);
                double N_dash_y = N_dash(index.j, grid_position[1]);
                double N_dash_z = N_dash(index.k, grid_position[2]);
                basis_gradient(0) = N_dash_x * N_y * N_z * (1/grid_spacing);
                basis_gradient(1) = N_x * N_dash_y * N_z * (1/grid_spacing);
                basis_gradient(2) = N_x * N_y * N_dash_z * (1/grid_spacing);
                velocity_gradient += node.velocity * basis_gradient.transpose();
            }
        }
    }
    Eigen::Matrix3d trial_F_e = (Eigen::Matrix3d::Identity() + time_step * velocity_gradient) * particle.F_E;
    Eigen::Matrix3d F_trial = trial_F_e * particle.F_P;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(trial_F_e, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d sigma = svd.singularValues();
    sigma[0] = std::clamp(sigma[0], 1 - max_compression, 1 + max_stretch);
    sigma[1] = std::clamp(sigma[1], 1 - max_compression, 1 + max_stretch);
    sigma[2] = std::clamp(sigma[2], 1 - max_compression, 1 + max_stretch);
    particle.F_E = svd.matrixU() * sigma.asDiagonal() * svd.matrixV().transpose();
    particle.F_P = particle.F_E.inverse() * F_trial;
}

Eigen::Matrix3d deformation_differential(const Particle& particle, ParticleStencil& stencil, double time_step)
{
    Eigen::Matrix3d delta_velocity_gradient = Eigen::Matrix3d::Zero();
    for (StencilEntry& entry : stencil)
    {
        assert(entry.node != nullptr);
        delta_velocity_gradient += (entry.node->delta_v) * (entry.gradient.transpose());
    }
    return time_step * delta_velocity_gradient * particle.F_E;
}

void force_differential(const Particle& particle, ParticleStencil& stencil, ParticleLinearisation& linearisation, double time_step)
{
    Eigen::Matrix3d delta_F_p = deformation_differential(particle, stencil, time_step);
    Eigen::Matrix3d delta_P_p = constitutive_differential(delta_F_p, linearisation);
    Eigen::Matrix3d constant_term = -particle.V_p0 * delta_P_p * particle.F_E.transpose();
    for (StencilEntry& entry : stencil)
    {
        assert(entry.node != nullptr);
        entry.node->delta_force += constant_term * entry.gradient;
    }
}

KrylovVector apply_A(const std::vector<Particle>& snow, std::vector<ParticleStencil>& stencils, std::vector<ParticleLinearisation>& linearisations, const std::vector<GridNode*>& krylov_nodes, const KrylovVector& q, double time_step, double beta)
{
    assert(q.size() == krylov_nodes.size());
    assert(stencils.size() == snow.size());
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        krylov_nodes[i]->delta_v = q[i];
        krylov_nodes[i]->delta_force = Eigen::Vector3d::Zero();
    }
    for (std::size_t i{0}; i < snow.size(); ++i)
    {
        force_differential(snow[i], stencils[i], linearisations[i], time_step);
    }
    KrylovVector Aq(krylov_nodes.size(), Eigen::Vector3d::Zero());
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        const GridNode& node = *krylov_nodes[i];

        Aq[i] = node.mass * q[i] - beta * time_step * node.delta_force;
    }
    return Aq;
}

double Conjugate_residual(Grid& grid, const std::vector<Particle>& snow, std::vector<ParticleStencil>& stencils, std::vector<ParticleLinearisation>& linearisations, const std::vector<GridNode*>& krylov_nodes, double grid_spacing, double time_step, double beta, double squared_tolerance, int maximum_iterations)
{
    KrylovVector x;
    KrylovVector b;
    KrylovVector r;
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        const GridNode& node = *krylov_nodes[i];
        x.push_back(node.velocity);
        b.push_back(node.mass * node.velocity);
    }
    KrylovVector Ax = apply_A(snow, stencils, linearisations, krylov_nodes, x, time_step, beta);
    double r_squared_norm{0.0};
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        if (krylov_nodes[i]->colision_constrained == true)
        {
            r.push_back({0,0,0});
        }
        else
        {
            r.push_back(b[i] - Ax[i]);
        }
        r_squared_norm += r[i].dot(r[i]);
    }
    if (r_squared_norm <= squared_tolerance)
        {
            return 0;
        }
    KrylovVector p = r;
    KrylovVector Ap = apply_A(snow, stencils, linearisations, krylov_nodes, p, time_step, beta);
    for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
    {
        if (krylov_nodes[i]->colision_constrained == true)
        {
            Ap[i] = {0,0,0};
        }
    }
    int iterations{0};
    while (r_squared_norm > squared_tolerance and iterations < maximum_iterations)
    {
        double r_dot_Ap{0.0};
        double Ap_dot_Ap{0.0};
        for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
        {
            r_dot_Ap += r[i].dot(Ap[i]);
            Ap_dot_Ap += Ap[i].dot(Ap[i]);
        }
        assert(Ap_dot_Ap > 1e-12);
        double alpha = r_dot_Ap / Ap_dot_Ap;
        double new_r_squared_norm{0.0};
        for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
        {
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * Ap[i];
            new_r_squared_norm += r[i].dot(r[i]);
        }
        r_squared_norm = new_r_squared_norm;
        iterations += 1;
        if (r_squared_norm <= squared_tolerance or iterations >= maximum_iterations)
        {
            break;
        }
        KrylovVector Ar = apply_A(snow, stencils, linearisations, krylov_nodes, r, time_step, beta);
        for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
        {
            if (krylov_nodes[i]->colision_constrained == true)
            {
                Ar[i] = {0,0,0};
            }
        }
        double Ar_dot_Ap{0.0};
        double beta_CR{0.0};
        for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
        {
            Ar_dot_Ap += Ar[i].dot(Ap[i]);
        }
        beta_CR = -Ar_dot_Ap / Ap_dot_Ap;
        for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
        {
            p[i] = r[i] + beta_CR * p[i];
            Ap[i] = Ar[i] + beta_CR * Ap[i];
        }
    }
    if (r_squared_norm > squared_tolerance)
    {
        std::cout << "Maximum iterations reached, no convergaence achieved\n" << "achieved r squared norm: " << r_squared_norm << '\n';
    }
    else
    {
        for (std::size_t i{0}; i < krylov_nodes.size(); ++i)
        {
            krylov_nodes[i]->velocity = x[i];
        }
    }
    return iterations;
}

void G2P(const Grid& grid, Particle& particle, double grid_spacing, double alpha)
{
    Eigen::Vector3d v_pic = Eigen::Vector3d::Zero();
    Eigen::Vector3d v_flip{};
    Eigen::Vector3d v_temp = Eigen::Vector3d::Zero();
    Eigen::Vector3d grid_position = particle.x_p / grid_spacing;
    Eigen::Vector3i base = grid_position.array().floor().cast<int>() - Eigen::Vector3i::Ones().array();
    for (int i{0}; i < 4; ++i)
    {
        for (int j{0}; j < 4; ++j)
        {
            for (int k{0}; k < 4; ++k)
            {
                GridIndex index{i + base[0], j + base[1], k + base[2]};
                auto it = grid.find(index);
                if (it == grid.end())
                {
                    continue;
                }
                const GridNode& node = it->second;
                double w_x = N(index.i, grid_position[0]);
                double w_y = N(index.j, grid_position[1]);
                double w_z = N(index.k, grid_position[2]);
                double w_p = w_x * w_y * w_z;
                v_pic += w_p * node.velocity;
                if (node.mass > 0)
                {
                    v_temp += w_p * (node.velocity - (node.momentum / node.mass));
                }
            }
        }
    }

    v_flip = particle.v_p + v_temp;
    
    //update velocity
    particle.v_p = (1 - alpha) * v_pic + alpha * v_flip;
}

void particle_collisions(Particle& particle, const std::vector<CollisionBody> & colliders)
{
    bool required = true;
    for (const CollisionBody& body : colliders)
    {
        particle.v_p = collision_helper(particle.x_p, particle.v_p, body, required);
    }
}

void position_update(Particle& particle, double time_step)
{
    particle.x_p = particle.x_p + time_step * particle.v_p;
}

void setup(std::vector<Particle>& snow, double grid_spacing)
{
    Grid grid;
    std::vector<ParticleStencil> stencils(snow.size());
    for (std::size_t i{0}; i < snow.size(); ++i)
    {
        stencils[i] = build_stencil(snow[i], grid_spacing);
    }
    for (std::size_t j{0}; j < snow.size(); ++j)
    {
        P2G(grid, snow[j], stencils[j], grid_spacing);
    }
    for (Particle& particle : snow)
    {
        calculate_volume(grid, particle, grid_spacing);
    }

}

Grid step(std::vector<Particle>& snow,const std::vector<CollisionBody>& colliders, std::vector<ParticleStencil>& stencils, Eigen::Vector3d gravity, double time_step, double grid_spacing, double mu_0, double lambda_0, double hardening_coefficient, double max_compression, double max_stretch, double alpha, StepTimings& timings)
{
    Grid grid;

    auto start_stencil = std::chrono::steady_clock::now();
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < snow.size(); ++i)
    {
        stencils[i] = build_stencil(snow[i], grid_spacing);
    }
    auto end_stencil = std::chrono::steady_clock::now();

    timings.stencil += std::chrono::duration<double, std::milli>(end_stencil - start_stencil).count();

    auto start_P2G = std::chrono::steady_clock::now();
    for (std::size_t i{0}; i < snow.size(); ++i)
    {
        P2G(grid, snow[i], stencils[i], grid_spacing);
    }
    auto end_P2G = std::chrono::steady_clock::now();

    timings.p2g += std::chrono::duration<double, std::milli>(end_P2G - start_P2G).count();

    std::vector<Eigen::Matrix3d> stress_matrices(snow.size());
    auto start_forces_constitutive = std::chrono::steady_clock::now();
    std::vector<ParticleLinearisation> linearisation;
    for (const Particle& particle : snow)
    {
        linearisation.push_back(build_lineratisation(particle, mu_0, lambda_0, hardening_coefficient));
    }

    for (std::size_t i{0}; i < snow.size(); ++i)
    {
        stress_matrices[i] = force_constitutive(snow[i], linearisation[i]);
    }
    auto end_forces_constitutive = std::chrono::steady_clock::now();

    timings.forces_constitutive += std::chrono::duration<double, std::milli>(end_forces_constitutive - start_forces_constitutive).count();

    auto start_forces_accumulative = std::chrono::steady_clock::now();
    for (std::size_t i{0}; i < snow.size(); ++i)
    {
        force_grid_accumulation(grid, stress_matrices[i], snow[i], stencils[i]);
    }
    auto end_forces_accumulative = std::chrono::steady_clock::now();

    timings.forces_accumulative += std::chrono::duration<double, std::milli>(end_forces_accumulative - start_forces_accumulative).count();

    auto start_grid_velocity = std::chrono::steady_clock::now();
    grid_velocity(grid, gravity, time_step);
    auto end_grid_velocity = std::chrono::steady_clock::now();

    timings.grid_velocity += std::chrono::duration<double, std::milli>(end_grid_velocity - start_grid_velocity).count();

    auto start_grid_collisions = std::chrono::steady_clock::now();
    grid_collisions(grid, colliders, grid_spacing);
    auto end_grid_collisions = std::chrono::steady_clock::now();

    timings.grid_collisions += std::chrono::duration<double, std::milli>(end_grid_collisions - start_grid_collisions).count();

    auto start_stress = std::chrono::steady_clock::now();
    #pragma omp parallel for
    for (Particle& particle : snow)
    {
        stress_update(grid, particle, grid_spacing, time_step, max_compression, max_stretch);
    }
    auto end_stress = std::chrono::steady_clock::now();

    timings.stress_update += std::chrono::duration<double, std::milli>(end_stress - start_stress).count();

    auto start_G2P = std::chrono::steady_clock::now();
    #pragma omp parallel for
    for (Particle& particle : snow)
    {
        G2P(grid, particle, grid_spacing, alpha);
    }
    auto end_G2P = std::chrono::steady_clock::now();

    timings.g2p += std::chrono::duration<double, std::milli>(end_G2P - start_G2P).count();

    auto start_particle_collisions = std::chrono::steady_clock::now();
    for (Particle& particle : snow)
    {
        particle_collisions(particle, colliders);
    }
    auto end_particle_collisions = std::chrono::steady_clock::now();

    timings.particle_collisions += std::chrono::duration<double, std::milli>(end_particle_collisions - start_particle_collisions).count();

    auto start_position_update = std::chrono::steady_clock::now();
    for (Particle& particle : snow)
    {
        position_update(particle, time_step);
    }
    auto end_position_update = std::chrono::steady_clock::now();

    timings.position_update += std::chrono::duration<double, std::milli>(end_position_update - start_position_update).count();

    return grid;
}

Grid semi_implicit_step(std::vector<Particle>& snow,const std::vector<CollisionBody>& colliders, std::vector<ParticleStencil>& stencils, Eigen::Vector3d gravity, double time_step, double grid_spacing, double mu_0, double lambda_0, double hardening_coefficient, double max_compression, double max_stretch, double alpha, double squared_tolerance, int maximum_iterations, double beta, StepTimings& timings)
{
    Grid grid;

    auto start_stencil = std::chrono::steady_clock::now();
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < snow.size(); ++i)
    {
        stencils[i] = build_stencil(snow[i], grid_spacing);
    }
    auto end_stencil = std::chrono::steady_clock::now();

    timings.stencil += std::chrono::duration<double, std::milli>(end_stencil - start_stencil).count();

    auto start_P2G = std::chrono::steady_clock::now();
    for (std::size_t i{0}; i < snow.size(); ++i)
    {
        P2G(grid, snow[i], stencils[i], grid_spacing);
    }
    std::vector<GridNode*> krylov_nodes = index_nodes(grid);
    auto end_P2G = std::chrono::steady_clock::now();

    timings.p2g += std::chrono::duration<double, std::milli>(end_P2G - start_P2G).count();

    std::vector<Eigen::Matrix3d> stress_matrices(snow.size());
    auto start_forces_constitutive = std::chrono::steady_clock::now();
    std::vector<ParticleLinearisation> linearisations;
    for (const Particle& particle : snow)
    {
        linearisations.push_back(build_lineratisation(particle, mu_0, lambda_0, hardening_coefficient));
    }
    for (std::size_t i{0}; i < snow.size(); ++i)
    {
        stress_matrices[i] = force_constitutive(snow[i], linearisations[i]);
    }
    auto end_forces_constitutive = std::chrono::steady_clock::now();

    timings.forces_constitutive += std::chrono::duration<double, std::milli>(end_forces_constitutive - start_forces_constitutive).count();

    auto start_forces_accumulative = std::chrono::steady_clock::now();
    for (std::size_t i{0}; i < snow.size(); ++i)
    {
        force_grid_accumulation(grid, stress_matrices[i], snow[i], stencils[i]);
    }
    auto end_forces_accumulative = std::chrono::steady_clock::now();

    timings.forces_accumulative += std::chrono::duration<double, std::milli>(end_forces_accumulative - start_forces_accumulative).count();

    auto start_grid_velocity = std::chrono::steady_clock::now();
    grid_velocity(grid, gravity, time_step);
    auto end_grid_velocity = std::chrono::steady_clock::now();

    timings.grid_velocity += std::chrono::duration<double, std::milli>(end_grid_velocity - start_grid_velocity).count();

    auto start_grid_collisions = std::chrono::steady_clock::now();
    grid_collisions(grid, colliders, grid_spacing);
    auto end_grid_collisions = std::chrono::steady_clock::now();

    timings.grid_collisions += std::chrono::duration<double, std::milli>(end_grid_collisions - start_grid_collisions).count();

    double iterations;
    auto start_cr = std::chrono::steady_clock::now();
    iterations = Conjugate_residual(grid, snow, stencils, linearisations, krylov_nodes, grid_spacing, time_step, beta, squared_tolerance, maximum_iterations);
    auto end_cr = std::chrono::steady_clock::now();
    timings.iterations += iterations;
    timings.cr += std::chrono::duration<double, std::milli>(end_cr - start_cr).count();

    auto start_stress = std::chrono::steady_clock::now();
    #pragma omp parallel for
    for (Particle& particle : snow)
    {
        stress_update(grid, particle, grid_spacing, time_step, max_compression, max_stretch);
    }
    auto end_stress = std::chrono::steady_clock::now();

    timings.stress_update += std::chrono::duration<double, std::milli>(end_stress - start_stress).count();

    auto start_G2P = std::chrono::steady_clock::now();
    #pragma omp parallel for
    for (Particle& particle : snow)
    {
        G2P(grid, particle, grid_spacing, alpha);
    }
    auto end_G2P = std::chrono::steady_clock::now();

    timings.g2p += std::chrono::duration<double, std::milli>(end_G2P - start_G2P).count();

    auto start_particle_collisions = std::chrono::steady_clock::now();
    for (Particle& particle : snow)
    {
        particle_collisions(particle, colliders);
    }
    auto end_particle_collisions = std::chrono::steady_clock::now();

    timings.particle_collisions += std::chrono::duration<double, std::milli>(end_particle_collisions - start_particle_collisions).count();

    auto start_position_update = std::chrono::steady_clock::now();
    for (Particle& particle : snow)
    {
        position_update(particle, time_step);
    }
    auto end_position_update = std::chrono::steady_clock::now();

    timings.position_update += std::chrono::duration<double, std::milli>(end_position_update - start_position_update).count();

    return grid;
}

