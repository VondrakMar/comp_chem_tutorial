#!/bin/bash

# The first argument passed to the script is the filename
input_file=$1
shift

# Initialize parsing flags to false
parse_energy=false
parse_hirshfeld=false
parse_aims_charges=false
parse_forces=false

# Loop through the remaining arguments to set flags
for arg in "$@"
do
  case $arg in
    energy)
      parse_energy=true
      ;;
    hirshfeld)
      parse_hirshfeld=true
      ;;
    aims_charges)
      parse_aims_charges=true
      ;;
    forces)
      parse_forces=true
      ;;
    *)
      echo "Unknown argument: $arg"
      ;;
  esac
done

# Calculate the number of atoms
nAt=$(( $(cat $input_file | wc -l) - 2 ))

# Extract total energy from aims.out and append to the energy file if flag is set
if [ "$parse_energy" = true ]; then
    grep "| Total energy of the DFT / Hartree-Fock s.c.f. calculation      : " aims.out | awk '{print $(NF-1)}' > energy
fi

# Extract Hirshfeld charges from aims.out and append to the hirshfeld file if flag is set
if [ "$parse_hirshfeld" = true ]; then
    grep "Hirshfeld charge   " aims.out | awk '{print $5}' > aims_hirshfeld
fi

# Extract AIMs charges from aims.out and append to the aims_charges file if flag is set
if [ "$parse_aims_charges" = true ]; then
    grep "|                   0   0 " aims.out | tail -${nAt} | awk '{print $4}' > aims_charges
fi

# Extract forces from aims.out and append to the forces file if flag is set
if [ "$parse_forces" = true ]; then
    sed -n '/Total atomic forces/,/------------------------------------/p' aims.out | grep '^  |' | awk '{print $3, $4, $5}' > forces
fi

# Load data into Python and write to an xyz file
python -c "
from ase.io import read, write
import numpy as np

# Read molecular structure from xyz file
mols = read('$input_file', format='xyz')
parse_hirshfeld = False
parse_aims_charges = False
parse_forces=False
parse_energy=False

if '$parse_hirshfeld' == 'true':
    parse_hirshfeld = True
if '$parse_aims_charges' == 'true':
    parse_aims_charges = True
if '$parse_forces' == 'true':
    parse_forces = True
if '$parse_energy' == 'true':
    parse_energy = True

# Load data from the respective files if they exist
if parse_hirshfeld:
    hirshfeld_data = np.loadtxt('aims_hirshfeld')
    mols.arrays['aims_hirshfeld'] = hirshfeld_data

if parse_aims_charges:
    aims_charges_data = np.loadtxt('aims_charges')
    mols.arrays['aims_charges'] = aims_charges_data

if parse_forces:
    forces_data = np.loadtxt('forces')
    mols.arrays['aims_forces'] = forces_data

if parse_energy:
    energy_data = np.loadtxt('energy')
    mols.info['aims_energy'] = energy_data

# Write the updated molecule to a new xyz file
if len(hirshfeld_data) > 0:
    write('loaded_file.xyz', mols)
"
