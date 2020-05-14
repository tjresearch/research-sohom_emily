import psi4

psi4.core.set_output_file('polarizability_test.out', False)
psi4.set_memory("2 GB")

h2o = psi4.geometry("""
O
H   1   0.96
H   1   0.96    2   104.5
""")

psi4.properties("ccsd/cc-pvdz", properties=['polarizability'])

