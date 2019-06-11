#include <pybind11/pybind11.h>
#include <cmath>

namespace py = pybind11;

double ricker_frequency_domain(double omega, double omega_central){
    return 2 * std::pow(omega, 2.) / (std::sqrt(M_PI) * std::pow(omega_central, 3.))
    * std::exp(-std::pow(omega, 2.) / std::pow(omega_central, 2.));
    }
    
double scattering_potential(double v, double v0){
    return 1/std::pow(v0, 2) - 1/std::pow(v, 2);
    }


PYBIND11_MODULE(fastfunctions, m){
    m.doc() = "Fast functions for Born modeling";
    m.def("ricker_frequency_domain", &ricker_frequency_domain, "Ricker wavelet in frequency domain",
          py::arg("omega"), py::arg("omega_central"));
    m.def("scattering_potential", &scattering_potential, "Scattering potential with respect to the homogeneous medium",
          py::arg("v"), py::arg("v0"));
}

