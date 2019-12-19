## Contents

Use Born modeling to generate seismic recordings from fracture scattered waves.
The method is described in Hu2018a (3D seismic characterization of fractures in
a dipping layer using the double-beam method, Hao Hu and Yingcai Zheng).

The program oneshot will generate a seismogram from a single source, recorded at a single
receiver location.

The program fullmodel will generate seismograms for one or more sources recorded at one
or more receivers.

## Requirements

 - Python 3.6+
 - numpy
 - quadpy
 - numexpr
 - tqdm
 - matplotlib

## How to start a simulation

To generate a full data set for a number of sources and receivers run `main.py fullmodel`:

 - Create a source file in which all source positions are specified.
 - Create a stations file in which all receiver positions are specified.
 - Create a velocity model file. This file should contain a function called
   create_velocity_model which returns a VelocityModel object. The velocity model
   provides the scatterer positions in a (N, 3) shape array. Scatterers are
   discrete points used to model fractures. A vertical fracture plane is
   represented by a dense row of scatterers along its top.
   An example script that does all of the above is located in `simple_model.py`. When run, this file will create a source file named `simple_sources.txt` and a stations file called `simple_receivers.txt`. It also contains a function to create the velocity model, so it can be passed to `main.py` as the model argument.
 - Decide on good values for the general required parameters: source wavelet frequency, (`--omega_central`), source amplitude (`--amplitude`),
   seismogram time (`--time`). An explanation of the parameters is provided when specifying the `-h/--help` flag.
 - Call `main.py fullmodel` with specifying the path to the model file, the source and receiver file.
   Specify the general parameters after main and the fullmodel parameters (source and receiver file) after fullmodel.
 - The output of the simulation will be saved in a directory called `output`, which is automatically created.

## Options

Available options and their explanation can be seen by calling the script main.py
with the -h flag:
```
python main.py -h
```
These options are valid for both subcommands.

Options for the subcommands oneshot and fullmodel can be viewed as such:
```
python main.py oneshot -h
```
or
```
python main.py fullmodel -h
```
These options are specific to the subcommands and need to be listed behind the
subcommand.

### Specifying options in a text file

Options for main.py can either be specified on the command line or in a text file.
If given as a file, every space on the command line needs to be replaced with a newline
in the file. An example options.txt is located in the project. The name of the option 
file is given to main.py prefixed with a `@` character: `python main.py @options.txt`.
The file name of the options file is arbitrary.

## Source file

Source files contain the position for all sources in the model.

### Source file formatting

The number of sources needs to be specified as `nsrc = N`, where `N` is an
integer. `nsrc` specifies how many sources will be defined in the file.
Afterwards, sources are listed sequentially:
```
xsource = <float x coordinate of source 1>
ysource = <float y coordinate of source 1>
zsource = <float z coordinate of source 1>
xsource = <float x coordinate of source 2>
ysource = <float y coordinate of source 2>
zsource = <float z coordinate of source 2>
.
.
.
```
Coordinates can either be specified as X.Y, X., or X, where X and Y are strings
of digits [0-9]. Scientific notation is not allowed.
The amount of whitespace around the equal sign doesnt matter, as well as the
amount of whitespace before and after the line.

## Stations file

The stations file contains the positions of all the receivers in the model. For
a single shot, a seismogram will be generated for every receiver.

### Stations file formatting

Example:
```
5
001 0.000000 0.000000 0.000000
002 0.000000 50.000000 0.000000
003 0.000000 100.000000 0.000000
004 0.000000 150.000000 0.000000
005 0.000000 200.000000 0.000000
```

The first line contains the number of receivers defined in the file. Every
following is preceded by a running index. The three numbers following it define
one receiver position. Their order is x, y, z coordinates of the receiver
position in m. The separator for values is whitespace.
