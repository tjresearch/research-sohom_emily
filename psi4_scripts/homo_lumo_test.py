import psi4

psi4.set_memory("2 GB")

h2o = psi4.geometry("""
O
H   1   0.96
H   1   0.96    2   104.5
""")

scf_e, scf_wfn = psi4.energy("scf/cc-pvdz", return_wfn=True)
print(scf_wfn)