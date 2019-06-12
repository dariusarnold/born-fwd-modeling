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

Options for main.py can either be specified on the command line or in a text file.
If given as a file, every space on the command line needs to be replaced with a newline
in the file. An example options.txt is located in the project. The name of the option 
file is given to main.py prefixed with a @ character: python main.py @options.txt.
The file name of the options file is arbitrary.
