#pragma once

#include "snow/particle.hpp"

#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>

struct ParticleGroup
{
    Eigen::Vector3d velocity;
    double mass;
};

struct ParticleSeed
{
    int group_id;
    Eigen::Vector3d position;
};

std::map<int, ParticleGroup> import_groups(const std::string& filename);
std::vector<ParticleSeed> import_particle_seeds(const std::string& filename);
std::vector<Particle> assemble_particles(const std::vector<ParticleSeed>& seed, const std::map<int, ParticleGroup>& groups);
std::vector<Particle> import_particles(const std::string& filename_group, const std::string& filename_particle_seed);