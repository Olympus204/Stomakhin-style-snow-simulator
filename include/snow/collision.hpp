#pragma once

#include <functional>
#include <Eigen/Dense>

struct CollisionBody
{
    std::function<double(const Eigen::Vector3d&)> phi;
    std::function<Eigen::Vector3d(const Eigen::Vector3d&)> normal;
    std::function<Eigen::Vector3d(const Eigen::Vector3d&)> velocity;
    double friction;
};

CollisionBody make_plane(const Eigen::Vector3d& point, const Eigen::Vector3d& normal, double friction);