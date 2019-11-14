import psi4
import os
import time

# Each of the xyz files are labeled with a number from 1 to 133885

def get_molecule_from_file(filenum):
    # Assumes dataset is in sibling directory
    f = open(os.path.join(os.getcwd(), "..", "dsgdb9nsd", "dsgdb9nsd_" + str(filenum).zfill(6) + ".xyz"), "r")
    lines = f.readlines()
    f.close()
    num_atoms = int(lines[0])
    atom_list = lines[2:2+num_atoms]
    for i in range(len(atom_list)):
        atom_list[i] = atom_list[i][:atom_list[i].rfind("\t")] + "\n"
    return psi4.geometry("".join(atom_list))

def generate_output_file_path(filenum):
    return os.path.join(os.getcwd(), "output", 'output_' + str(filenum) + '.dat')

def process_molecule(filenum, thermochemical=False):
    psi4.core.set_output_file(generate_output_file_path(filenum), False)
    psi4.set_memory("2 GB")
    molecule = get_molecule_from_file(filenum)
    if thermochemical:
        e, wfn = psi4.freq('b3lyp/cc-pvqz', molecule=molecule, return_wfn=True)
    else:
        e, wfn = psi4.energy('b3lyp/cc-pvqz', molecule=molecule, return_wfn=True)
    return wfn

def extract_rotational_constants(filenum, wfn):
    f = open(generate_output_file_path(filenum), 'r')
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        if lines[i].find("Rotational constants:") > -1 and lines[i].find("[MHz]") > -1:
            words = lines[i].split()
            return float(words[4])/1000, float(words[7])/1000, float(words[10])/1000

def extract_dipole_moment(filenum, wfn):
    f = open(generate_output_file_path(filenum), 'r')
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        if lines[i].find("Dipole Moment: [D]") > -1:
            return float(lines[i+1][lines[i+1].find("Total:") + 6:])
    
def extract_homo_lumo(filenum, wfn):
    homo = wfn.epsilon_a_subset("AO", "ALL").get(wfn.nalpha())
    lumo = wfn.epsilon_a_subset("AO", "ALL").get(wfn.nalpha() + 1)
    return homo, lumo

def extract_zpve(filenum, wfn):
    f = open(generate_output_file_path(filenum), 'r')
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        if lines[i].find("Total ZPE, Electronic energy at 0 [K]") > -1:
            words = lines[i].split()
            return float(words[-2])

def extract_enthalpy(filenum, wfn):
    f = open(generate_output_file_path(filenum), 'r')
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        if lines[i].find("Total H, Enthalpy at  298.15 [K]") > -1:
            words = lines[i].split()
            return float(words[-2])

def extract_gibbs_free_energy(filenum, wfn):
    f = open(generate_output_file_path(filenum), 'r')
    lines = f.readline()
    f.close()
    for i in range(len(lines)):
        if lines[i].find("Total G, Free enthalpy at  298.15 [K]") > -1:
            print(lines[i])
            words = lines[i].split()
            return float(words[-2])

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
        output = str(filenum) + "," + str(a) + "," + str(b) + "," + str(c) + "," + str(dipole) + "," + str(homo) + "," + str(lumo)
        if thermochemical:
            zpve = extract_zpve(filenum, wfn)
            enthalpy = extract_enthalpy(filenum, wfn)
            gibbs_free_energy = extract_gibbs_free_energy(filenum, wfn)
            output += "," + str(zpve) + "," + str(enthalpy) + "," + str(gibbs_free_energy)
        output += "\n" 
        f.write(output)
    f.close()

start = time.time()
batch_process(1, 3, thermochemical=True)
end = time.time()
print(end-start)
