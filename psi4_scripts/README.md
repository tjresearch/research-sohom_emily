## Psi4 Computation

The `xyzparser.py` script allows you to run speed benchmarks of DFT calculations using the Psi4 chemical computation software. In order to use this script, you should have already installed Psi4 into your conda environment. For further installation instructions, please go to the [source documentation](http://www.psicode.org/psi4manual/master/conda.html)

This script assumes you already have the QM9 dataset, available on  [FigShare](https://springernature.figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904), downloaded and unzipped on your machine. You can specify the path to the dataset using the `--data` command line argument, such as below:
```
python xyzparser.py --data <QM9_DIR_PATH>
```

To enable/disable the calculation of thermochemical properties, change the function or basis set used, or batch process a different set of molecules, please edit the source code directly. The supported list of properties are included below:
- rotational constants (A,B,C)
- dipole moment (mu)
- HOMO and LUMO energies
- zero-point vibrational energy (zpve)
- standard enthalpy and Gibbs free energy (H 298.15,G 298.15)