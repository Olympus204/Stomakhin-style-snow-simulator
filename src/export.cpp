#include "snow/export.hpp"
#include "snow/particle.hpp"
#include "snow/grid.hpp"
#include "snow/marching_cubes_tables.hpp"

#include <vector>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <optional>
#include <array>
#include <utility>
#include <filesystem>
#include <algorithm>
#include <cassert>
#include <unordered_set>
#include <unordered_map>
#include <iostream>

namespace
{
    constexpr std::array<std::pair<int, int>, 12> edge_corners = {{
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 0},
        {4, 5},
        {5, 6},
        {6, 7},
        {7, 4},
        {0, 4},
        {1, 5},
        {2, 6},
        {3, 7}
    }};

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
}

void export_particles(const std::vector<Particle>& particles, int frame_no)
{
    std::ostringstream filename;
    filename << "output/particle/frame_" << std::setw(6) << std::setfill('0') << frame_no << ".csv";
    std::ofstream file{filename.str()};
    file << "x,y,z,Vp\n";
    for (const Particle& particle : particles)
    {
        double Vp = particle.V_p0 * (particle.F_E * particle.F_P).determinant();
        assert(Vp > 0);
        file << particle.x_p.x() << ","
             << particle.x_p.y() << ","
             << particle.x_p.z() << ","
             << Vp << "\n";
    }

}



std::vector<MeshParticle> import_particles(const std::string& filename)
{
    std::vector<MeshParticle> particles;
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
        
        const double x{std::stod(read_field(row, "x"))};
        const double y{std::stod(read_field(row, "y"))};
        const double z{std::stod(read_field(row, "z"))};
        const double volume{std::stod(read_field(row, "Vp"))};

        const Eigen::Vector3d position = {x, y, z};
        particles.push_back({position, volume});
    }
    return particles;
}

ReconstructionGrid construct_grid(Eigen::Vector3d corner1, Eigen::Vector3d corner2, double spacing)
{
    ReconstructionGrid grid;
    grid.origin = {std::min(corner1[0], corner2[0]), std::min(corner1[1], corner2[1]), std::min(corner1[2], corner2[2])};
    grid.spacing = spacing;
    grid.Nx = ceil(abs(corner1[0] - corner2[0]) / spacing) + 1;
    grid.Ny = ceil(abs(corner1[1] - corner2[1]) / spacing) + 1;
    grid.Nz = ceil(abs(corner1[2] - corner2[2]) / spacing) + 1;
    grid.values.resize(grid.Nx * grid.Ny * grid.Nz, 0.0);
    return grid;
}

std::optional<std::size_t> grid_index(const ReconstructionGrid& grid, double i, double j, double k)
{
    if( i < grid.Nx and j < grid.Ny and k < grid.Nz and i >= 0 and j >= 0 and k >= 0)
    {
        return static_cast<std::size_t>(i + grid.Nx * (j + grid.Ny * k));
    }
    return std::nullopt;
}

void particle_to_grid(const MeshParticle& particle, ReconstructionGrid& grid, std::unordered_set<GridIndex, GridIndexHash>& active_cubes)
{
    Eigen::Vector3d grid_position = (particle.position - grid.origin) / grid.spacing;
    Eigen::Vector3d base = grid_position.array().floor().matrix() - Eigen::Vector3d{1,1,1};
    for(int i{0}; i < 4; ++i)
    {
        for(int j{0}; j < 4; ++j)
        {
            for(int k{0}; k < 4; ++k)
            {
                Eigen::Vector3d grid_node = {i + base[0], j + base[1], k + base[2]};
                auto index = grid_index(grid, static_cast<int>(grid_node[0]),
                                              static_cast<int>(grid_node[1]),
                                              static_cast<int>(grid_node[2]));
                double w_x = N(grid_node[0], grid_position[0]);
                double w_y = N(grid_node[1], grid_position[1]);
                double w_z = N(grid_node[2], grid_position[2]);
                double w_p = w_x * w_y * w_z; 
                if (index)
                {
                    grid.values[*index] += (particle.volume / (grid.spacing * grid.spacing * grid.spacing) * w_p);
                    for (int l{0}; l < 2; ++l)
                    {
                        for (int m{0}; m < 2; ++m)
                        {
                            for (int n{0}; n < 2; ++n)
                            {
                                active_cubes.insert({static_cast<int>(grid_node[0]) + l - 1, static_cast<int>(grid_node[1]) + m - 1, static_cast<int>(grid_node[2]) + n - 1});
                            }
                        }
                    }
                }
            }
        }
    }
}

std::optional<std::array<CubeCorners, 8>> construct_cube(const ReconstructionGrid& grid, Eigen::Vector3i starting_corner)
{
    if (grid.Nx < 2 || grid.Ny < 2 || grid.Nz < 2)
    {
        return std::nullopt;
    }
    if (starting_corner[0] < 0 || starting_corner[0] >= grid.Nx -1 || starting_corner[1] < 0 || starting_corner[1] >= grid.Ny -1 || starting_corner[2] < 0 || starting_corner[2] >= grid.Nz -1)
    {
        return std::nullopt;
    }
    Eigen::Vector3d grid_corner = starting_corner.cast<double>();
    std::array<CubeCorners, 8> cube;
    cube[0].grid_position = grid_corner;
    auto index = grid_index(grid, grid_corner[0], grid_corner[1], grid_corner[2]);
    cube[0].value = grid.values[*index];
    cube[1].grid_position = {grid_corner[0] + 1, grid_corner[1], grid_corner[2]};
    index = grid_index(grid, grid_corner[0] + 1, grid_corner[1], grid_corner[2]);
    cube[1].value = grid.values[*index];
    cube[2].grid_position = {grid_corner[0] + 1, grid_corner[1] + 1, grid_corner[2]};
    index = grid_index(grid, grid_corner[0] + 1, grid_corner[1] + 1, grid_corner[2]);
    cube[2].value = grid.values[*index];
    cube[3].grid_position = {grid_corner[0], grid_corner[1] + 1, grid_corner[2]};
    index = grid_index(grid, grid_corner[0], grid_corner[1] + 1, grid_corner[2]);
    cube[3].value = grid.values[*index];
    cube[4].grid_position = {grid_corner[0], grid_corner[1], grid_corner[2] + 1};
    index = grid_index(grid, grid_corner[0], grid_corner[1], grid_corner[2] + 1);
    cube[4].value = grid.values[*index];
    cube[5].grid_position = {grid_corner[0] + 1, grid_corner[1], grid_corner[2] + 1};
    index = grid_index(grid, grid_corner[0] + 1, grid_corner[1], grid_corner[2] + 1);
    cube[5].value = grid.values[*index];
    cube[6].grid_position = {grid_corner[0] + 1, grid_corner[1] + 1, grid_corner[2] + 1};
    index = grid_index(grid, grid_corner[0] + 1, grid_corner[1] + 1, grid_corner[2] + 1);
    cube[6].value = grid.values[*index];
    cube[7].grid_position = {grid_corner[0], grid_corner[1] + 1, grid_corner[2] + 1};
    index = grid_index(grid, grid_corner[0], grid_corner[1] + 1, grid_corner[2] + 1);
    cube[7].value = grid.values[*index];
    
    return cube;
}

EdgeKey make_edge_key(const std::array<CubeCorners, 8>& cube, int edge_number)
{
    auto [a, b] = edge_corners[edge_number];

    GridIndex first{
        static_cast<int>(cube[a].grid_position.x()),
        static_cast<int>(cube[a].grid_position.y()),
        static_cast<int>(cube[a].grid_position.z())
    };

    GridIndex second{
        static_cast<int>(cube[b].grid_position.x()),
        static_cast<int>(cube[b].grid_position.y()),
        static_cast<int>(cube[b].grid_position.z())
    };

    if (second < first)
    {
        std::swap(first, second);
    }

    return {first, second};
}

int case_index(const std::array<CubeCorners, 8>& cube, double isovalue)
{
    int case_index{0};
    for (int i{0}; i < 8; ++i)
    {
        if (cube[i].value < isovalue)
        {
            case_index |= 1 << i;
        }
    }
    return case_index;
}

Eigen::Vector3d interpolate_edge(const std::array<CubeCorners, 8>& cube, int edge_number, double isovalue)
{
    assert(edge_number >= 0 and edge_number < 12);
    auto [a, b] = edge_corners[edge_number];
    double value_a = cube[a].value;
    double value_b = cube[b].value;
    assert(value_a != value_b);
    double t = (isovalue - value_a) / (value_b - value_a);
    return cube[a].grid_position + t * (cube[b].grid_position - cube[a].grid_position);
}

std::size_t get_vertex(Mesh& mesh, std::unordered_map<EdgeKey, std::size_t, EdgeKeyHash>& vertex_cache, const ReconstructionGrid& grid, const std::array<CubeCorners, 8>& cube, int edge_number, double isovalue)
{
    EdgeKey key = make_edge_key(cube, edge_number);

    auto it = vertex_cache.find(key);

    if (it != vertex_cache.end())
    {
        return it->second;
    }

    Eigen::Vector3d grid_position = interpolate_edge(cube, edge_number, isovalue);

    Eigen::Vector3d world_position = grid.origin + grid.spacing * grid_position;

    std::size_t vertex_index = mesh.vertices.size();

    mesh.vertices.push_back(world_position);

    vertex_cache.emplace(key, vertex_index
    );
    return vertex_index;
}

void triangulate_cube(Mesh& mesh, std::unordered_map< EdgeKey, std::size_t, EdgeKeyHash>& vertex_cache, const ReconstructionGrid& grid, const std::array<CubeCorners, 8>& cube, double isovalue)
{
    int index = case_index(cube, isovalue);

    const auto& row = marching_cubes::tri_table[index];

    for (int i{0}; i < 5; ++i)
    {
        if (row[3 * i] == -1)
        {
            break;
        }

        std::size_t a = get_vertex(mesh, vertex_cache, grid, cube, row[3 * i],  isovalue);

        std::size_t b = get_vertex(mesh, vertex_cache, grid, cube, row[3 * i + 1], isovalue);

        std::size_t c = get_vertex(mesh, vertex_cache, grid, cube, row[3 * i + 2], isovalue);

        mesh.faces.push_back({a, b, c});
    }
}

void grid_to_mesh(Mesh& mesh, const ReconstructionGrid& grid, const std::unordered_set<GridIndex, GridIndexHash>& active_cubes, double isovalue)
{
    assert(grid.Nx >= 2 && grid.Ny >= 2 && grid.Nz >= 2);

    std::unordered_map<EdgeKey, std::size_t, EdgeKeyHash> vertex_cache;

    vertex_cache.reserve(active_cubes.size() * 3);

    for (const GridIndex& index : active_cubes)
    {
        auto cube = construct_cube(grid, {index.i, index.j, index.k});

        if (!cube)
        {
            continue;
        }

        triangulate_cube(mesh, vertex_cache, grid, *cube, isovalue);
    }
}

void export_mesh(const Mesh& mesh, int frame_no)
{
    std::ostringstream filename;

    filename
        << "output/mesh/frame_"
        << std::setw(6)
        << std::setfill('0')
        << frame_no
        << ".obj";

    std::ofstream file{filename.str()};

    if (!file.is_open())
    {
        throw std::runtime_error{"Could not open mesh output file: " + filename.str()};
    }

    for (const Eigen::Vector3d& vertex : mesh.vertices)
    {
        file
            << "v "
            << vertex.x() << " "
            << vertex.y() << " "
            << vertex.z() << "\n";
    }

    for (const auto& face : mesh.faces)
    {
        file
            << "f "
            << face[0] + 1 << " "
            << face[1] + 1 << " "
            << face[2] + 1 << "\n";
    }
}

void build_mesh_from_particles(Eigen::Vector3d corner1, Eigen::Vector3d corner2, double spacing, double isovalue)
{
    namespace fs = std::filesystem;

    const fs::path particle_directory{"output/particle"};
    const fs::path mesh_directory{"output/mesh"};

    if (!fs::exists(particle_directory))
    {
        throw std::runtime_error{"Particle output directory does not exist"};
    }

    fs::create_directories(mesh_directory);

    std::vector<std::pair<int, fs::path>> frames;

    for (const fs::directory_entry& entry :
         fs::directory_iterator(particle_directory))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        const fs::path& path = entry.path();

        if (path.extension() != ".csv")
        {
            continue;
        }

        std::string stem = path.stem().string();

        if (stem.rfind("frame_", 0) != 0)
        {
            continue;
        }

        std::string frame_text = stem.substr(6);

        try
        {
            std::size_t characters_read{0};

            int frame_no =
                std::stoi(frame_text, &characters_read);

            if (characters_read != frame_text.size())
            {
                continue;
            }

            frames.push_back({frame_no, path});
        }
        catch (const std::exception&)
        {
            continue;
        }
    }

    std::sort(
        frames.begin(),
        frames.end(),
        [](const auto& a, const auto& b)
        {
            return a.first < b.first;
        });

    for (const auto& [frame_no, path] : frames)
    {

        std::vector<MeshParticle> particles = import_particles(path.string());

        ReconstructionGrid grid = construct_grid(corner1, corner2, spacing);
        std::unordered_set<GridIndex, GridIndexHash> active_cubes;
        for (const MeshParticle& particle : particles)
        {
            particle_to_grid(particle, grid, active_cubes);
        }
        Mesh mesh;

        grid_to_mesh(mesh, grid, active_cubes, isovalue);

        export_mesh(mesh, frame_no);
        std::cout << frame_no << '\n';
    }
}