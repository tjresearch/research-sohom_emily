import os
import time
import argparse

import psi4

# QM9 data available for download at 
# https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
# Each of the xyz files are labeled with a number from 1 to 133885

# By default, we assume that the data folder is in a sibling directory to this repository
QM9_DATA_DIR = os.path.join(os.getcwd(), '..', '..', 'qm9_data')
FUNCTIONAL = 'b3lyp'
BASIS_SET = 'cc-pvqz'


def get_molecule_from_file(filenum):
    f = open(os.path.join(QM9_DATA_DIR,
                          "dsgdb9nsd_" + str(filenum).zfill(6) + ".xyz"), "r")
    lines = f.readlines()
    f.close()
    num_atoms = int(lines[0])
    atom_list = lines[2:2+num_atoms]
    for i in range(len(atom_list)):
        atom_list[i] = atom_list[i][:atom_list[i].rfind("\t")] + "\n"
    return psi4.geometry("".join(atom_list))


def generate_output_file_path(filenum):
    return os.path.join(os.getcwd(), "output", 'output_' + str(filenum) + '.dat')


def get_output_file_lines(filenum):
    f = open(generate_output_file_path(filenum), 'r')
    lines = f.readlines()
    f.close()
    return lines


def tokenize_output_file_line_with_targets(filenum, *target_list):
    lines = get_output_file_lines(filenum)
    for i in range(len(lines)):
        if all([lines[i].find(target) > -1 for target in target_list]):
            return lines[i].split()


def process_molecule(filenum, thermochemical=False):
    psi4.core.set_output_file(generate_output_file_path(filenum), False)
    psi4.set_memory("2 GB")
    molecule = get_molecule_from_file(filenum)
    computational_method = FUNCTIONAL + '/' + BASIS_SET
    if thermochemical:
        e, wfn = psi4.freq(computational_method,
                           molecule=molecule, return_wfn=True)
    else:
        e, wfn = psi4.energy(
            computational_method, molecule=molecule, return_wfn=True)
    return wfn


def extract_rotational_constants(filenum, wfn):
    tokens = tokenize_output_file_line_with_targets(
        filenum, "Rotational constants:", "[MHz]")
    return float(tokens[4])/1000, float(tokens[7])/1000, float(tokens[10])/1000


def extract_dipole_moment(filenum, wfn):
    lines = get_output_file_lines(filenum)
    for i in range(len(lines)):
        if lines[i].find("Dipole Moment: [D]") > -1:
            return float(lines[i+1][lines[i+1].find("Total:") + 6:])


def extract_homo_lumo(filenum, wfn):
    homo = wfn.epsilon_a_subset("AO", "ALL").get(wfn.nalpha())
    lumo = wfn.epsilon_a_subset("AO", "ALL").get(wfn.nalpha() + 1)
    return homo, lumo


def extract_zpve(filenum, wfn):
    tokens = tokenize_output_file_line_with_targets(
        filenum, "Total ZPE, Electronic energy at 0 [K]")
    return float(tokens[-2])


def extract_enthalpy(filenum, wfn):
    tokens = tokenize_output_file_line_with_targets(
        filenum, "Total H, Enthalpy at  298.15 [K]")
    return float(tokens[-2])


def extract_gibbs_free_energy(filenum, wfn):
    tokens = tokenize_output_file_line_with_targets(
        filenum, "Total G, Free enthalpy at  298.15 [K]")
    return float(tokens[-2])


def batch_process(start_num, end_num, thermochemical=False):
    f = open("output.csv", "w")
    output_header = "Index,A,B,C,Dipole,HOMO,LUMO"
    if thermochemical:
        output_header += ",zpve,H 298.15,G 298.15"
    output_header += "\n"
    f.write(output_header)
    for filenum in range(start_num, end_num+1):
        wfn = process_molecule(filenum, thermochemical=thermochemical)
        a, b, c = extract_rotational_constants(filenum, wfn)
        dipole = extract_dipole_moment(filenum, wfn)
        homo, lumo = extract_homo_lumo(filenum, wfn)
        output = str(filenum) + "," + str(a) + "," + str(b) + "," + \
            str(c) + "," + str(dipole) + "," + str(homo) + "," + str(lumo)
        if thermochemical:
            zpve = extract_zpve(filenum, wfn)
            enthalpy = extract_enthalpy(filenum, wfn)
            gibbs_free_energy = extract_gibbs_free_energy(filenum, wfn)
            output += "," + str(zpve) + "," + str(enthalpy) + \
                "," + str(gibbs_free_energy)
        output += "\n"
        f.write(output)
    f.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Computes quantum chemical properties given xyz data")
    parser.add_argument('--data', type=str, dest='data_path')
    return parser.parse_args()

start = time.time()
args = parse_arguments()
if args.data_path:
    QM9_DATA_DIR = args.data_path
batch_process(1, 3, thermochemical=True)
end = time.time()
print("Time elapsed (s): ", end-start)
