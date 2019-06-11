## Contents

Implement Born modeling to calculate the scattered P wave due to fractures in the
frequency domain. The method is described in Hu2018a (3D seismic characterization
of fractures in a dipping layer using the double-beam method, Hao Hu and Yingcai Zheng).

## Options

Options for main.py can either be specified on the command line or in a text file.
If given as a file, every space on the command line needs to be replaced with a newline
in the file. An example options.txt is located in the project. The name of the option 
file is given to main.py prefixed with a @ character: python main.py @options.txt.
The file name of the options file is arbitrary.


# C++ fastfunctions

Extension written in C++ that provides two functions used during the main program.
For a (probably unnoticeable) speed improvement, follow the instructions for compiling
the module. A python fallback version is provided.

### How to compile fastfunctions

Change to cpp_fastfunctions directory and run
```
mkdir cmake-build-dir && cmake .. && make
```
An extension module named `fastfunctions.cpython-<version specific>`.so will be created.
Copy this module to the main born-fwd-modeling folder.

### Requirements to compile fastfunctions

 - CMake install
 - A C++ compiler
 - pybind11 (Specify the version of python that CMake searches in the CMakeLists.txt)