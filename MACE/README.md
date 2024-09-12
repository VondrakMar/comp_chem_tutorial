# MACE
> "MACE is a machine learning software for predicting many-body atomic interactions and generating force fields. It utilizes higher order equivariant message passing for fast and accurate predictions."

## Documentation
The official mace documentation can be found here:
[MACE docu](https://mace-docs.readthedocs.io/en/latest/)

## Workflow - Training and Evaluating the MACE-MLS
Since the training and evaluation process is very computational expensive, you will do the following steps on one of the local workstations (guybrush, horselover, coraline or trillian).

1. Get a 'struc_\*.xyz' file (from Molecular dynamics simulation, from a database, etc.) and save it in a folder named 'struc_\*' (same structure number as the 'struc_\*'-file).


2. After that execute the following lines of code in your terminal:
    ```Bash
    source /opt/intel/oneapi/setvars.sh; ulimit -s unlimited
    mpirun -np x <path_to>/fhi-aims.231212_1/build/aims.231212_1.scalapack.mpi.x >> aims.out
    ```

3. As a next step create a 'jobcraft_run.py'-file with the following code:
    
    *Note:* The following files should be present in your folder:
    * `struc_*.xyz`-file
    * `aims.out` file

    ```Python
    import os
    import sys
    import ase
    import ase.io
    import numpy as np
    from ase.io.aims import read_aims_output


    def save_results_to_xyz(mol, results, prefix="dft_", saved_name="struc_saved.xyz"):
        for res in results:
            if type(results[res]) == float:
                mol.info[f"{prefix}{res}"] = results[res]
            elif len(results[res]) == len(mol):
                mol.arrays[f"{prefix}{res}"] = results[res]
            else: 
                mol.info[f"{prefix}{res}"] = results[res]
        
        ase.io.write(saved_name, mol)


    if __name__ == "__main__":
        folder = sys.argv[1]
        xyz_file = [file for file in os.listdir(folder) if f"{folder.replace('/', '')}.xyz" in file][0]
        aims_file = [file for file in os.listdir(folder) if "aims.out" in file][0]
        file = f'{folder}/{xyz_file}'

        # Load data
        aims = read_aims_output(file=folder + aims_file)
        xyz = ase.io.read(filename=folder + xyz_file, format='extxyz')

        # Update xyz-file and save data
        save_results_to_xyz(mol=xyz, 
                            results={"energy": aims.get_forces(), 
                                     "forces": aims.get_total_energy()})
    ```

    This file can be executed by the following bash command:
    
    ```Bash
    python3 Jobcraft_run.py $<root_folder>
    ```
    
    If you have multiple 'struc_*'-files/folders you can use the following command. Adjust the paths to your needs:
    
    ```Bash
    for struc in <root_folder>/struc_*; do python3 Jobcraft_run.py "${struc}"; done
    ```


4. After the programm has finished, you need to combine all the 'struc_saved.xyz'-files into one master (if you have multiple 'struc_*'-folders).
    
    ```Bash
    for folder in */; do cat "${folder}${folder[::-1]}.xyz" >> master.xyz; done
    ```


5. Next, we need the atomization energies of all present elements.

    *Note:* You need the following files (in every of your folders):
    * struc_\*.xyz      (just one atom)
    * geometry.in       (random geometry file)
    * control.in        (will be created in the following)
    
    1. Locate the 'fhi-aims.231212_1' folder (get your current path with '`pwd`' and note it).

    2. Create a new folder (name = <atom_of_your_choice>), go in there and create file called 'control.in' and insert the following text into it:

        ```
        xc                                 pbe
        relativistic                       atomic_zora scalar
        compute_forces                     .true.
        charge                             0
        fixed_spin_moment                  <XY>
        spin                               collinear
        ```
        *Note:* 
        * 'fixed_spin_moment' <XY> depends on the Atom you look at
        * The example is set up for hydrogen, yours might look different
    
    3. After you created all '`<atom>`/control.in' files you can execute the following command in the terminal to add additional data the the 'control.in' file:
        ```Bash
        cat <path_to>/fhi-aims.231212_1/species_defaults/defaults_2020/<mode>/<ordernumber + tab> >> control.in
        ```
        *Note:*
        * `<mode>` can be either 'light', 'tight' or 'intermediate'
        * Iterate through all present atom folders.

    4. After everything is setup, you can start the calculation by executing the following command:

        ```Bash
        for element in */; do cd $element; mpirun -np x <path_to>/fhi-aims.231212_1/build/aims.231212_1.scalapack.mpi.x >> aims.out; cd ../; done
        ```
    
    5. Look for the following Line at the end of the `aims.out` file:
        ```
        | Total energy of the DFT / Hartree-Fock s.c.f. calculation      :
        ```
    
    6. Note the values with there corresponding order numbers (In form of a dict).


6. Create a new file called 'train_model.sh' and paste the following Code in it:

    ```Bash
    mace_run_train \
        --name="MACE_model" \
        --train_file="<your_xyz_file>.xyz" \
        --valid_fraction=0.05 \
        --config_type_weights='{"Default":1.0}' \
        --E0s='{1:-13.598030178, 8:-2043.220684884}' \          # See Step 5.
        --model="MACE"\
        --hidden_irreps='128x0e + 128x1o' \
        --r_max=5.0 \
        --batch_size=10 \
        --max_num_epochs=1500 \
        --swa \
        --energy_key="dft_energy" \
        --forces_key="dft_forces" \
        --start_swa=1200 \
        --ema \
        --ema_decay=0.99 \
        --amsgrad \
        --restart_latest \
        --device=cuda \
    ```
    *Note:* 
    * Documentation can be found [here](https://mace-docs.readthedocs.io/en/latest/guide/training.html)
    * '--EOs=': Insert/Paste your calculated atomization energies here


7. Evaluate the model by creating and running the file 'eval_model.sh' with the following code:

    ```Bash
    python3 /usr/local/lib/python3.10/dist-packages/mace/cli/eval_configs.py \
            --configs="<path_to_source_xyz>" \
            --model="<path_to_your_model>" \
            --output="<path_of_your_output_file>" \
    ```
    *Note:* Documentation can be found [here](https://mace-docs.readthedocs.io/en/latest/guide/evaluation.html)