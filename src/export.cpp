#include "snow/export.hpp"
#include "snow/particle.hpp"

#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <fstream>

void export_particles(const std::vector<Particle>& particles, int frame_no)
{
    std::ostringstream filename;
    filename << "output/frame_" << std::setw(6) << std::setfill('0') << frame_no << ".csv";
    std::ofstream file{filename.str()};
    file << "x,y,z\n";
    for (const Particle& particle : particles)
    {
        file << particle.x_p.x() << ","
             << particle.x_p.z() * -1 << ","
             << particle.x_p.y() << "\n";
    }

}