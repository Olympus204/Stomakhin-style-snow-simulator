#include "snow/export.hpp"

#include <vector>
#include <array>

int main()
{
    Eigen::Vector3d corner1 = {-10,-10,-10};
    Eigen::Vector3d corner2 = {10,10,10};
    double spacing = 0.05;
    double isovalue = 0.55;
    build_mesh_from_particles(corner1, corner2, spacing, isovalue);
}
