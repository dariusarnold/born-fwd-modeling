## Contents

Implement Born modeling to calculate the scattered P wave due to fractures in the
frequency domain. The method is described in Hu2018a (3D seismic characterization
of fractures in a dipping layer using the double-beam method, Hao Hu and Yingcai Zheng).

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

### Formatting

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

### Formatting

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
position in m. The seperator for values is whitespace.
