# BioGraph

This is an Open Source Python library to segment tissues into different biologically relevant regions based on Hidden Markov Random Fields.

The library currently works from networkX graph objects, the user can refer to the `SACpy` package to transform segmented microscopy images to networkX graph objects.

## Installation

1. Install Anaconda
2. Create a new conda environment:

Navigate to the future project directory, and enter:

`conda create -n grph python=3.8`

`conda activate grph`

3. Download the package

`git clone https://gitlab.pasteur.fr/gronteix1/biograph.git`

This should download the project.

4. Install the required packages

In the terminal, navigate to the package directory and type:

`pip install .`

This should install all the relevant libraries.

Congratulations! You're ready to rock'n'roll. The library is now installed on your computer and is ready to be used from `jupyter`. 

## Example

A working example of the library can be accessed [here](Examples/test_hmrf_single_cell.ipynb)