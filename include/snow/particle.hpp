#pragma once

#include <Eigen/Dense>

struct Particle
{
    Eigen::Vector3d x_p;
    Eigen::Vector3d v_p;
    double m_p;
    double V_p0;
    Eigen::Matrix3d F_P;
    Eigen::Matrix3d F_E;

    Particle(
        const Eigen::Vector3d& initial_position,
        const Eigen::Vector3d& initial_velocity,
        double initial_mass,
        double initial_volume
    )
        : x_p(initial_position),
          v_p(initial_velocity),
          m_p(initial_mass),
          V_p0(initial_volume),
          F_P(Eigen::Matrix3d::Identity()),
          F_E(Eigen::Matrix3d::Identity())
    {
    }
};