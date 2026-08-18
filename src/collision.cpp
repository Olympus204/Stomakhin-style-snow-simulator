#include "snow/collision.hpp"

#include <Eigen/Dense>

CollisionBody make_plane(const Eigen::Vector3d& point, const Eigen::Vector3d& normal, double friction)
{
    Eigen::Vector3d unit_normal = normal.normalized();

    return CollisionBody{
        [point, unit_normal](const Eigen::Vector3d& x)
        {
            return unit_normal.dot(x - point);
        },

        [unit_normal](const Eigen::Vector3d&)
        {
            return unit_normal;
        },

        [](const Eigen::Vector3d&)
        {
            return Eigen::Vector3d::Zero();
        },

        friction
    };
}

