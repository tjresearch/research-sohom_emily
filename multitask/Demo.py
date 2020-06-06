#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().run_line_magic('run', 'QM9GNN2_Multitask.ipynb')
import QM9GNN2_Multitask
print("finished importing QM9GNN2_Multitask")

# In[ ]:


import psi4
import time
from os import path, getcwd


# In[ ]:


def predict_property_neural(prop=None, mol_id=-1):
    if mol_id == -1:
        raise ValueError("ID must be between 1 and 133885")
    if prop == 'gap':
        lumo = predict_property_neural(prop='lumo', mol_id=mol_id)
        homo = predict_property_neural(prop='homo', mol_id=mol_id)
        return lumo - homo
    
    if not any([prop in cluster for cluster in clusters]):
        raise ValueError("Property was not found in clusters list")
    
    return predict_property(prop, mol_id, clusters, N=N, F=F, S=S)
    


# In[ ]:


predict_property_neural('A', 1)


# In[ ]:


def get_data_folder_path():
    return path.join(getcwd(), '..', '..', 'qm9_data')


# In[ ]:


def get_molecule_from_file(filenum):
    filepath = path.join(get_data_folder_path(), 
                           'dsgdb9nsd_' + str(filenum).zfill(6) + '.xyz')
    f = open(filepath, 'r')
    lines = f.readlines()
    f.close()
    num_atoms = int(lines[0])
    atom_list = lines[2:2+num_atoms]
    for i in range(len(atom_list)):
        atom_list[i] = atom_list[i][:atom_list[i].rfind("\t")] + "\n"
    return psi4.geometry("".join(atom_list))


# In[ ]:


def generate_dft_output_file_path(filenum):
    return path.join('psi4_output', 'output_'+str(filenum)+'.dat')


# In[ ]:


def process_molecule(filenum, thermochemical=False):
    psi4.core.set_output_file(generate_dft_output_file_path(filenum), False)
    psi4.set_memory('2 GB')
    molecule = get_molecule_from_file(filenum)
    if thermochemical:
        e, wfn = psi4.freq('b3lyp/cc-pvqz', molecule=molecule, return_wfn=True)
    else:
        e, wfn = psi4.energy('b3lyp/cc-pvqz', molecule=molecule, return_wfn=True)
    return wfn


# In[ ]:


def extract_rotational_constants(filenum, wfn):
    f = open(generate_dft_output_file_path(filenum), 'r')
    lines = f.readlines()
    f.close()
    for i in range(len(lines)-1, -1, -1):
        if lines[i].find('Rotational constants:') > -1 and lines[i].find('[MHz]') > -1:
            words = lines[i].split()
            rot_constants = []
            for const in [words[4], words[7], words[10]]:     
                if const.isnumeric():
                    rot_constants.append(float(const)/1000)
                else:
                    rot_constants.append(const)
                    # rot_constants.append(float(const)/1000)
            return rot_constants
    return None, None, None


# In[ ]:


def extract_dipole_moment(filenum, wfn):
    f = open(generate_dft_output_file_path(filenum), 'r')
    lines = f.readlines()
    f.close()
    for i in range(len(lines)-1, -1, -1):
        if lines[i].find("Dipole Moment: [D]") > -1:
            return lines[i+1][lines[i+1].find("Total:") + 6:]


# In[ ]:


def extract_homo_lumo(filenum, wfn):
    homo = wfn.epsilon_a_subset("AO", "ALL").get(wfn.nalpha())
    lumo = wfn.epsilon_a_subset("AO", "ALL").get(wfn.nalpha() + 1)
    return homo, lumo


# In[ ]:


def extract_gap(filenum, wfn):
    homo, lumo = extract_homo_lumo_gap(fileum, wfn)
    return lumo - homo


# In[ ]:


def extract_zpve(filenum, wfn):
    f = open(generate_dft_output_file_path(filenum), 'r')
    lines = f.readlines()
    f.close()
    for i in range(len(lines)-1, -1, -1):
        if lines[i].find("Total ZPE, Electronic energy at 0 [K]") > -1:
            words = lines[i].split()
            return words[-2]


# In[ ]:


def extract_zero_point_internal_energy(filenum, wfn):
    f = open(generate_dft_output_file_path(filenum), 'r')
    lines = f.readlines()
    f.close()
    for i in range(len(lines)-1, -1, -1):
        if lines[i].find("Total E0, Electronic energy") > -1:
            words = lines[i].split()
            return words[-2]


# In[ ]:


def extract_internal_energy(filenum, wfn):
    f = open(generate_dft_output_file_path(filenum), 'r')
    lines = f.readlines()
    f.close()
    for i in range(len(lines)-1, -1, -1):
        if lines[i].find("Total E, Electronic energy at  298.15 [K]") > -1:
            words = lines[i].split()
            return words[-2]


# In[ ]:


def extract_enthalpy(filenum, wfn):
    f = open(generate_dft_output_file_path(filenum), 'r')
    lines = f.readlines()
    f.close()
    for i in range(len(lines)-1, -1, -1):
        if lines[i].find("Total H, Enthalpy at  298.15 [K]") > -1:
            words = lines[i].split()
            return words[-2]


# In[ ]:


def extract_gibbs_free_energy(filenum, wfn):
    f = open(generate_dft_output_file_path(filenum), 'r')
    lines = f.readlines()
    f.close()
    for i in range(len(lines)-1, -1, -1):
        if lines[i].find("Total G,") > -1:
            words = lines[i].split()
            return words[-2]


# In[ ]:


def extract_cv(filenum, wfn):
    f = open(generate_dft_output_file_path(filenum), 'r')
    lines = f.readlines()
    f.close()
    for i in range(len(lines)-1, -1, -1):
        if lines[i].find('Total Cv') > -1:
            words = lines[i].split()
            return words[2]


# In[ ]:


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


# In[ ]:


def predict_all_properties_dft(filenum=-1, thermochemical=False):
    if filenum == -1:
        raise ValueError("Filenum must be between 1 and 133885")
    wfn = process_molecule(filenum, thermochemical=thermochemical)
    a, b, c = extract_rotational_constants(filenum, wfn)
    dipole = extract_dipole_moment(filenum, wfn)
    homo, lumo = extract_homo_lumo(filenum, wfn)
    ret_dict = {'A': a, 'B': b, 'C': c, 'mu': dipole, 'homo': homo, 
                'lumo': lumo, 'gap': lumo-homo}
    if thermochemical:
        zpve = extract_zpve(filenum, wfn)
        internal_energy = extract_internal_energy(filenum, wfn)
        u0 = extract_zero_point_internal_energy(filenum, wfn)
        enthalpy = extract_enthalpy(filenum, wfn)
        gibbs_free_energy = extract_gibbs_free_energy(filenum, wfn)
        cv = extract_cv(filenum, wfn)
        ret_dict.update({'zpve': zpve, 'u0': u0, 'u298': internal_energy, 
                           'h298': enthalpy, 'g298': gibbs_free_energy, 'cv': cv})
    return ret_dict


# In[ ]:


def lookup_property(prop=None, mol_id=-1):
    return y_all.loc[mol_id-1, prop]


# In[ ]:


def prompt_user_for_calculation():
    while True:
        num = -1
        while num < 1 or num > 133885:
            try:
                num = int(input('Choose a molecule index (1-133885): '))
                if num == -1:
                    return
            except ValueError:
                print("Please provide a valid number")
                num = -1
            
        properties = ['A', 'B', 'C', 'mu', 'homo', 'lumo', 'gap', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
        prop = None
        while prop not in properties:
            print("Choose an available property from the following:")
            print("A, B, C, mu, homo, lumo, gap, zpve, u0, u298, h298, g298, cv")
            prop = input('Choose a property: ')
        
        dft = None
        while dft not in ['0', '1']:
            print("Choose whether to use DFT or neural methods")
            print("0 for neural methods, 1 for DFT")
            dft = input('Calculation type: ')
        
        if dft == '1':
            print('Beginning DFT calculation')
            start = time.time()
            thermochemical = prop in properties[7:]
            ret_dict = predict_all_properties_dft(num, thermochemical=thermochemical)
            print(ret_dict[prop])
            end = time.time()
            print('DFT calculation took', end-start, 's')
        else:
            print('Beginning neural calculation:')
            start = time.time()
            print(predict_property_neural(prop=prop, mol_id=num))
            end = time.time()
            print('Neural method took', end-start, 's')
        print('Actual data:')
        print(lookup_property(prop=prop, mol_id=num))


# In[ ]:


prompt_user_for_calculation()


# In[ ]:


properties = ['A', 'B', 'C', 'mu', 'homo', 'lumo', 'gap', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']

for prop in properties:
    print(prop)
    errors = list()
    for index in np.random.choice(133885, 10):
        pred = predict_property_neural(prop=prop, mol_id=index+1)
        actual = lookup_property(prop=prop, mol_id=index+1)
        err = abs((pred-actual)/actual*100)
        print(err)
        errors.append(err)
    print('total err', sum(errors)/len(errors))


# In[ ]:




