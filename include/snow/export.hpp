#pragma once

#include "snow/particle.hpp"
#include "snow/grid.hpp"

#include <vector>
#include <optional>
#include <unordered_set>

struct ReconstructionGrid
{
    Eigen::Vector3d origin;
    double spacing;

    std::size_t Nx;
    std::size_t Ny;
    std::size_t Nz;

    std::vector<double> values;
};

struct MeshParticle
{
    Eigen::Vector3d position;
    double volume;
};

struct CubeCorners
{
    Eigen::Vector3d grid_position;
    double value;
};

struct Mesh
{
    std::vector<Eigen::Vector3d> vertices;
    std::vector<std::array<std::size_t, 3>> faces;
};

struct EdgeKey
{
    GridIndex first;
    GridIndex second;

    bool operator==(const EdgeKey&) const = default;
};

struct EdgeKeyHash
{
    std::size_t operator()(const EdgeKey& edge) const noexcept
    {
        GridIndexHash hash;

        std::size_t h1 = hash(edge.first);
        std::size_t h2 = hash(edge.second);

        return h1 ^ (
            h2
            + 0x9e3779b9
            + (h1 << 6)
            + (h1 >> 2)
        );
    }
};


void export_particles(const std::vector<Particle>& particles, int frame_no);
std::vector<MeshParticle> import_particles(const std::string& filename);
ReconstructionGrid construct_grid(Eigen::Vector3d corner1, Eigen::Vector3d corner2, double spacing);
std::optional<std::size_t> grid_index(const ReconstructionGrid& grid, double i, double j, double k);
void particle_to_grid(const MeshParticle& particle, ReconstructionGrid& grid, std::unordered_set<GridIndex, GridIndexHash>& active_cubes);
std::optional<std::array<CubeCorners, 8>> construct_cube(const ReconstructionGrid& grid, Eigen::Vector3i starting_corner);
EdgeKey make_edge_key(const std::array<CubeCorners, 8>& cube, int edge_number);
int case_index(const std::array<CubeCorners, 8>& cube, double isovalue);
Eigen::Vector3d interpolate_edge(const std::array<CubeCorners, 8>& cube, int edge_number, double isovalue);
std::size_t get_vertex(Mesh& mesh, std::unordered_map<EdgeKey, std::size_t, EdgeKeyHash>& vertex_cache, const ReconstructionGrid& grid, const std::array<CubeCorners, 8>& cube, int edge_number, double isovalue);
void triangulate_cube(Mesh& mesh, std::unordered_map< EdgeKey, std::size_t, EdgeKeyHash>& vertex_cache, const ReconstructionGrid& grid, const std::array<CubeCorners, 8>& cube, double isovalue);
void grid_to_mesh(Mesh& mesh, const ReconstructionGrid& grid, const std::unordered_set<GridIndex, GridIndexHash>& active_cubes, double isovalue);
void triangle_grid_to_world(std::vector<std::array<Eigen::Vector3d, 3>>& triangles, const ReconstructionGrid& grid);
void export_mesh(const Mesh& mesh, int frame_no);
void build_mesh_from_particles(Eigen::Vector3d corner1, Eigen::Vector3d corner2, double spacing, double isovalue);