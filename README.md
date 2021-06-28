# 🥗 Salad 🥗

🥗  is an Open Source Python library to segment tissues into different biologically relevant regions based on [Hidden Markov Random Fields](https://en.wikipedia.org/wiki/Hidden_Markov_random_field#:~:text=In%20statistics%2C%20a%20hidden%20Markov,an%20underlying%20Markov%20random%20field.&text=are%20independent%20(conditional%20independence%20of,given%20the%20Markov%20random%20field).). The algorithm is loosely inspired from Zhang et al.'s seminal [work](https://www.csd.uwo.ca/~oveksler/Courses/Fall2012/CS9840/PossibleStudentPapers/Zhang2001.pdf).

Starting from a network where each cell possesses a categorial feature `cell_type`, 🥗  creates a network with an identical structure with homogeneous regions corresponding to specific mixes of cells. 

The HMRF algorith infers the tissue structure from the observed cell distribution. The homogeneous regions in the inferred tissue are of a given **class**. Any **class** is defined by a specific mix of **cell types**. The **cell type** of a given cell is the mix of cell properties that have been recorded using the cell properties on other channels.

For the creation of the network from segmented microscopy images, we encourage you to look at **🍒 griottes 🍒** (and give us your feedback!).

## Installation

If it's not already done, install [Anaconda](https://www.anaconda.com/). Then you should install the 🥗 in a local environment. For the moment, only installation via the repository is possible, so you'll have to download it from the command line. In the command prompt enter:

`git clone https://gitlab.pasteur.fr/gronteix1/biograph.git`

This should download the project. Then install the required packages. In the terminal, navigate to the package directory and type:

`pip install .`

Congratulations! You're ready to play. The library is now installed on your computer and is ready to be used from `jupyter`. 

## Example

A working example of the library can be accessed [here](Examples/example_final.ipynb)