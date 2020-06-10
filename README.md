# Multitask Graph Convolutional Networks for Molecular Property Prediction

**Overview**
Our project seeks to combine multitask learning with graph convolutional networks to build a network capable of predicting molecular properties with high speed and accuracy. It builds upon the Spektral library and uses Psi4 for as a baseline for speed comparisons.

**Required Packages and Installation Instructions**
1. We recommend the use of the `conda` package manager, , available from [Anaconda](https://www.anaconda.com/distribution/), for easy installation of packages. In this project, we used Python 3.7 and machine learning and scientific computing libraries Tensorflow, Scikit-Learn, and matplotlib. These maybe installed with `conda install tensorflow scikit-learn matplotlib`.
2. Spektral - Spektral is a library built on top of Tensorflow and Keras including packages for graph learning and molecular datasets. The documetation is found at https://danielegrattarola.github.io/spektral/. To install on Ubuntu, install the required dependencies using 
`sudo apt install python3-dev graphviz libgraphviz-dev libcgraph6 pkg-config` and then run `pip install spektral`. The complete installation instructions are found at https://pypi.org/project/spektral/
3. Psi4 - Psi4 is a suite of program for chemical and molecular simulation and property prediction. Psi4 can be installed into Anaconda by running `conda install psi4 psi4-rt -c psi4`. The complete installation instructions are found at http://www.psicode.org/psi4manual/master/build_obtaining.html
4. RDKit - RDKit is cheminformatics library that provides various chemical visualization and conversion tools. To install in Anaconda, run `conda install rdkit -c rdkit`. The complete installation instructions can be found at https://rdkit.org/docs/Install.html

**Dataset**
We are learning from the QM9 dataset, which has chemical properties for 134k organic molecules with up to 9 heavy atoms (CONF). More information about this dataset and download instructions can be found at http://www.quantum-machine.org/datasets/. Note that graphical representations of the molecules come bundled with Spektral, but the true geometries are necessary for density-functional computation with Psi4.

**Build and Run Instructions**
Unfortunately, we do not yet have a complete shipped product. However, all of our files are interactive Python notetbooks and may be downloaded and run in Jupyter Notebook or Google Colab.
