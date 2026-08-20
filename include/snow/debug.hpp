#pragma once

struct StepTimings
{
    double stencil{0.0};
    double p2g{0.0};
    double forces_constitutive{0.0};
    double forces_accumulative{0.0};
    double grid_velocity{0.0};
    double grid_collisions{0.0};
    double stress_update{0.0};
    double cr{0.0};
    double g2p{0.0};
    double particle_collisions{0.0};
    double position_update{0.0};
};

double print_timings(StepTimings timings);