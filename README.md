# Neural Networks for Finite Difference Approximations (NNFDA)

## Authors: Hunter Johnson and Clayton Kramp

### Running the Project
Running the MATLAB scripts `mit*.m` and `heatEquation.m` to create the data files.  Type `python3 HEDriver.py` or `python3 NSDriver.py` to launch NNFDA.  Use `openAndCompare*.m` to visualize the results.

### Description of files

#### \*Driver.py
The drivers used in this program.  They each call `solver.py` with different hyperparameters to create and deploy a Neural Network.

#### solver.py
File that contains the solver.  Creates and runs the neural network

#### helpers/
Helpers that we used for the project

#### differential\_equations/\*DE.py
The classes that hold informaiton for each DE.  Analytical and FinDiff each hold ODEs, and PDE holds information for the heat equation and N-S model.

#### differential\_equations/OpenAndCompare\*.py
Opens and visualizes the results in MATLAB
