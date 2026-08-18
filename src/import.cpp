#include "snow/import.hpp"
#include "snow/particle.hpp"

#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

std::string read_field(std::stringstream& row, const std::string& column_name)
{
    std::string field;

    if (!std::getline(row, field, ','))
    {
        throw std::runtime_error{
            "Missing CSV field: " + column_name
        };
    }

    if (!field.empty() && field.back() == '\r')
    {
        field.pop_back();
    }

    if (field.empty())
    {
        throw std::runtime_error{
            "Empty CSV field: " + column_name
        };
    }

    return field;
}

std::map<int, ParticleGroup> import_groups(const std::string& filename)
{
    std::map<int, ParticleGroup> groups;
    std::ifstream file{filename};
    if (!file.is_open())
    {
        throw std::runtime_error{"File not open: " + filename};
    }
    std::string line;
    if (!std::getline(file, line))
    {
        throw std::runtime_error{"Group data file is empty: " + filename};
    }
    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }
        std::stringstream row(line);
        
        const int group_id{std::stoi(read_field(row, "group_ID"))};
        const double velocity_x{std::stod(read_field(row, "v_x"))};
        const double velocity_y{std::stod(read_field(row, "v_y"))};
        const double velocity_z{std::stod(read_field(row, "v_z"))};
        const double mass{std::stod(read_field(row, "m"))};

        const Eigen::Vector3d velocity = {velocity_x, velocity_y, velocity_z};
        ParticleGroup new_group = {velocity, mass};
        auto [it, inserted] = groups.emplace(group_id, new_group);

        if (!inserted)
        {
            throw std::runtime_error{
                "Duplicate group id: " + std::to_string(group_id)
            };
        }
    }
    return groups;
}

std::vector<ParticleSeed> import_particle_seeds(const std::string& filename)
{
    std::vector<ParticleSeed> particles;
    std::ifstream file{filename};
    if (!file.is_open())
    {
        throw std::runtime_error{"File not open: " + filename};
    }
    std::string line;
    if (!std::getline(file, line))
    {
        throw std::runtime_error{"Particle data file is empty: " + filename};
    }
    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }
        std::stringstream row(line);
        
        const int group_id{std::stoi(read_field(row, "group_ID"))};
        const double x{std::stod(read_field(row, "x"))};
        const double y{std::stod(read_field(row, "y"))};
        const double z{std::stod(read_field(row, "z"))};

        const Eigen::Vector3d position = {x, y, z};
        particles.push_back({group_id, position});
    }
    return particles;
}

std::vector<Particle> assemble_particles(const std::vector<ParticleSeed>& seeds, const std::map<int, ParticleGroup>& groups)
{   
    std::vector<Particle> particles;
    for (const ParticleSeed& particle_data : seeds)
    {
        auto it = groups.find(particle_data.group_id);

        if (it == groups.end())
        {
            throw std::runtime_error{
                "Particle references undefined group id: "
                + std::to_string(particle_data.group_id)
            };
        }

        const ParticleGroup& group = it->second;    
        particles.emplace_back(particle_data.position, group.velocity, group.mass, 0.0);
    }
    return particles;
}

std::vector<Particle> import_particles(const std::string& filename_group, const std::string& filename_particle_seed)
{
    std::map<int, ParticleGroup> groups{import_groups(filename_group)};
    std::vector<ParticleSeed> particle_seed{import_particle_seeds(filename_particle_seed)};
    std::vector<Particle> particles{assemble_particles(particle_seed, groups)};
    return particles;
}