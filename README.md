# Multitask Graph Convolutional Networks for Molecular Property Prediction

**Overview**
Our project seeks to combine multitask learning with graph convolutional networks to build a network capable of predicting molecular properties with high speed and accuracy. It builds upon the Spektral library and uses Psi4 for as a baseline for speed comparisons.

**Required Packages and Installation Instructions**
1. Spektral - Spektral is a library, built on top of Keras with a Tensorflow backend, including packages for graph learning and molecular datasets. The documetation is found at [https://danielegrattarola.github.io/spektral/] and it may be installed by running `pip install spektral`. The complete installation instructions are found at [https://pypi.org/project/spektral/]
2. Psi4 - Psi4 is a suite of program for chemical and molecular simulation and property prediction. Psi4 can be installed into Anaconda by running `conda update psi4 -c psi4`. The complete installation instructions are foud at [http://www.psicode.org/psi4manual/master/build_obtaining.html]

**Dataset**
We are learning from the QM9 dataset, which has chemical properties for 134k organic molecules with up to 9 heavy atoms (CONF). More information about this dataset and download instructions can be found at [http://www.quantum-machine.org/datasets/]

**Build and Run Instructions**
Unfortunately, we do not yet have a complete shipped product. However, all of our files are interactive Python notetbooks and may be downloaded and run in Jupyter Notebook or Google Colab.
