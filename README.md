# **`Raisin`**

**`Raisin`**  is an Open Source Python library to segment tissues into different biologically relevant regions based on [Hidden Markov Random Fields](https://en.wikipedia.org/wiki/Hidden_Markov_random_field#:~:text=In%20statistics%2C%20a%20hidden%20Markov,an%20underlying%20Markov%20random%20field.&text=are%20independent%20(conditional%20independence%20of,given%20the%20Markov%20random%20field).). The algorithm is loosely inspired from [Zhang et al.](https://www.csd.uwo.ca/~oveksler/Courses/Fall2012/CS9840/PossibleStudentPapers/Zhang2001.pdf).

Starting from a networkX graph where each cell possesses a categorial feature `cell_type`, `Raisin`  creates a network with an identical structure with homogeneous regions corresponding to specific mixes of cells. This allows for **unsupervised segmentation** of the tissue.

## Installation

If it's not already done, install [Anaconda](https://www.anaconda.com/). Then you should install `Raisin` in a local environment. For the moment, only installation via the repository is possible. In the command prompt enter:

`git clone https://github.com/microfluidix/HMRF`

This should download the project. Then install the required packages. In the terminal, navigate to the package directory and type:

`pip install .`

Congratulations! You're ready to play. The library is now installed on your computer and is ready to be used from `jupyter`. 

## Example

A working example of the library can be accessed [here](Examples/example_final.ipynb)
