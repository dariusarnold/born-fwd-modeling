#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
//#include <string>
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
//#include <Eigen/Dense>
//#include <pybind11/eigen.h>

namespace py = pybind11;

double ricker_frequency_domain(double omega, double omega_central){
    return 2 * std::pow(omega, 2.) / (std::sqrt(M_PI) * std::pow(omega_central, 3.)) * std::exp(-std::pow(omega, 2.) / std::pow(omega_central, 2.));
    }

// Equivalent to from VelocityModel import Vector3D
//py::object Vector3D = py::module::import("VelocityModel").attr("Vector3D");

double length(const std::vector<double>& vec){
    return std::sqrt(std::pow(vec[0], 2.) + std::pow(vec[1], 2.) + std::pow(vec[2], 2.));
    }
    
std::vector<double> operator-(const std::vector<double>& vector1, const std::vector<double>& vector2){
    std::vector<double> result;
    std::transform(vector1.begin(), vector1.end(), vector2.begin(), std::back_inserter(result), std::minus<double>());
    return result;
    }

std::complex<double> greens_function(double density, double v0, const std::vector<double>& x, const std::vector<double>& x_prime, double omega){
    double l = length(x - x_prime);
    double a = 1. / (4. * M_PI * density * std::pow(v0, 2.) * l);
    std::complex<double> i(0, -1);
    std::complex<double> b = std::exp(i * omega * l / v0);
    return a * b;
}
    
double scattering_potential(double v, double v0){
    return 1/std::pow(v0, 2) - 1/std::pow(v, 2);
    }
    
double scattering_potential_one_div(double v, double v0){
    /* 
    Eine Multiplikation kostet weniger cycles als eine Division für floats 
    https://godbolt.org/z/sSxHeH
    */    
    return (std::pow(v, 2) - std::pow(v0, 2)) / (std::pow(v, 2) * std::pow(v0, 2));
}


std::complex<double> integral(double x, double y, double z, py::tuple additional_params){
    std::vector<double> x_prime{x, y, z};
    auto velocity_model = additional_params[3];
    auto result_obj = velocity_model.attr("eval_at")(x_prime);
    double v0 = additional_params[1].cast<double>();
    double v = result_obj.cast<double>();
    if (v == v0){
        return 0.;
        }
        
    
    double density = additional_params[0].cast<double>();
    double omega = additional_params[2].cast<double>();
    auto xs = additional_params[4].cast<std::vector<double>>();
    auto xr = additional_params[5].cast<std::vector<double>>();
 
    double epsilon = scattering_potential(v, v0);
    std::complex<double> G0_left = greens_function(density, v0, xs, x_prime, omega);
    std::complex<double> G0_right = greens_function(density, v0, x_prime, xr, omega);
    return G0_left * epsilon * G0_right;
    }


py::object sub(py::object i, py::object j){
    return i.attr("__sub__")(j);
}

/*
std::vector<double> getVector(int len=0){
    std::vector<double> v(len);
    for (int i = 0; i != len; i++){
        v[i] = i;
        }
    return v;
}

py::array_t<double> getArray(const py::ssize_t size){
    std::vector<double> v(len);
    for (int i = 0; i != len; i++){
        v[i] = i;
        }
    return py::array_t<double>(v);
}

void throw_exception(){
    // this actually works and is translated to a Python ValueError
    throw std::invalid_argument("FEAR THE EXCEPTION");
}

double answer = 42.0;

Eigen::MatrixXd rand_matrix(){
    Eigen::MatrixXd m = Eigen::MatrixXd::Random(3,3);
    return m;
}
*/

PYBIND11_MODULE(fastfunctions, m){
    m.doc() = "Fast functions for Born modeling";
    m.def("ricker_frequency_domain", &ricker_frequency_domain, "Ricker wavelet in frequency domain");
    m.def("greens_function", &greens_function, "Response of medium at position x to a source at position x_prime");
    m.def("scattering_potential", &scattering_potential, "Scattering potential with respect to the homogeneous medium");
    m.def("scattering_potential_one_div", &scattering_potential_one_div, "Scattering potential with one division operation less");
    m.def("length", &length);
    m.def("integral", &integral);
    /*
    //name arguments with py::arg
    m.def("greet", &greet, "Greet the user", py::arg("user_name"));
    m.def("throw_exception", &throw_exception, "Throw an exception");
    // py::arg supports default arguments
    m.def("get_vector", &getVector, "Arange like vector init", py::arg("len") = 0);
    //m.def("get_array", &getArray, "Return ndarray");
    // export a variable
    m.attr("the_answer") = answer;
    m.def("random_matrix", &rand_matrix, "Create random matrix");
    m.def("subtract", &sub, "Generalized subtraction");
    */
}

