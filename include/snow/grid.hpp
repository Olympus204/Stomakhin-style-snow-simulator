#pragma once

#include <Eigen/Dense>
#include <unordered_map>
#include <array>

#include "snow/particle.hpp"
#include "snow/collision.hpp"
#include "snow/debug.hpp"

struct GridNode
{
    double mass = 0.0;
    double density = 0.0;
    Eigen::Vector3d momentum = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d force = Eigen::Vector3d::Zero();
};

struct GridIndex
{
    int i;
    int j;
    int k;

    auto operator<=>(const GridIndex&) const = default;
};

struct GridIndexHash
{
    std::size_t operator()(const GridIndex& index) const noexcept
    {
        std::size_t seed{0};

        auto combine = [&seed](int value)
        {
            std::size_t hash = std::hash<int>{}(value);

            seed ^= hash
                + 0x9e3779b97f4a7c15ULL
                + (seed << 6)
                + (seed >> 2);
        };

        combine(index.i);
        combine(index.j);
        combine(index.k);

        return seed;
    }
};

struct StencilEntry
{
    GridIndex index;

    double weight{0.0};

    Eigen::Vector3d gradient{
        Eigen::Vector3d::Zero()
    };

    GridNode* node{nullptr};
};

using ParticleStencil =
    std::array<StencilEntry, 64>;

using Grid = std::unordered_map<GridIndex, GridNode, GridIndexHash>;

double N(const double& a, const double& b);
double N_dash(const double& grid_node, const double& particle_coordinate);
ParticleStencil build_stencil(const Particle& particle, double grid_spacing);
void P2G(Grid& grid, const Particle& particle, ParticleStencil& stencil, double grid_spacing);
void P2G(Grid& grid, const Particle& particle, double grid_spacing);
void calculate_volume (Grid& grid, Particle& particle, double grid_spacing);
Eigen::Matrix3d force_constitutive(const Particle& particle, double mu_0, double lambda_0, double hardening_coefficient);
void force_grid_accumulation(Grid& grid, const Eigen::Matrix3d& dphidF_E, const Particle& particle, const ParticleStencil& stencil);
void force_calculation(Grid& grid,const Particle& particle,double grid_spacing,double mu_0,double lambda_0,double hardening_coefficient);
void grid_velocity(Grid& grid, const Eigen::Vector3d gravity, const double time_step);
void grid_collisions(Grid& grid, const std::vector<CollisionBody>& collisions, double grid_spacing);
void stress_update(const Grid& grid, Particle& particle, double grid_spacing, double time_step, double max_compression, double max_stretch);
void G2P(const Grid& grid, Particle& particle, double grid_spacing, double alpha);
void particle_collisions(Particle& particle, const std::vector<CollisionBody> & colliders);
void position_update(Particle& particle, double time_step);
void setup(std::vector<Particle>& snow, double grid_spacing);
Grid step(std::vector<Particle>& snow,const std::vector<CollisionBody>& colliders, std::vector<ParticleStencil>& stencils, Eigen::Vector3d gravity, double time_step, double grid_spacing, double mu_0, double lambda_0, double hardening_coefficient, double max_compression, double max_stretch, double alpha, StepTimings& timings);