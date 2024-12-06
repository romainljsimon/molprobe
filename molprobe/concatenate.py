import os
import re

def fast_scandir(path, dirname):
    subfolders= [f.path for f in os.scandir(path) if (f.is_dir() and dirname in f.path)]
    return subfolders
# Example usage

def read_lammps_file(filename):
    """
    Reads a LAMMPS file and returns a list of timesteps with corresponding data.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    data = []
    timestep_data = []
    current_timestep = None

    for i, line in enumerate(lines):
        if line.startswith("ITEM: TIMESTEP"):
            timestep_index = i + 1
            if current_timestep is not None:
                data.append((current_timestep, timestep_data))
                timestep_data = []
            current_timestep = int(lines[i + 1].strip())

        elif current_timestep is not None and i > timestep_index:
            timestep_data.append(line)
    
    if current_timestep is not None:
        data.append((current_timestep, timestep_data))

    return data


def write_lammps_file(filename, combined_data):
    """
    Writes combined LAMMPS data to a new file.
    """
    with open(filename, 'w') as file:
        for timestep, data in combined_data:
            file.write(f"ITEM: TIMESTEP\n{timestep}\n")
            file.writelines(data)


def concatenate_lammps_files(input_directory, output_file):
    """
    Concatenates multiple LAMMPS files while incrementing timesteps and removing duplicates.
    """
    combined_data = []
    last_timestep = 0
    seen_timesteps = set()
    def extract_number(filename):
        """Extract numerical part of the filename for sorting."""
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else float('inf')

    files = sorted(
        [f for f in os.listdir(input_directory) if (os.path.isfile(os.path.join(input_directory, f)))],
        key=extract_number
    )
    files = [os.path.join(input_directory, f) for f in files if re.match(r'^dump_tw\d+\.lammptrj$', os.path.basename(f))]
    for filename in files:
        file_data = read_lammps_file(filename)
        for timestep, data in file_data:
            adjusted_timestep = timestep + last_timestep
            if adjusted_timestep not in seen_timesteps:
                combined_data.append((adjusted_timestep, data))
                seen_timesteps.add(adjusted_timestep)
        
        if file_data:
            last_timestep = max(timestep + last_timestep for timestep, _ in file_data)

    write_lammps_file(output_file, combined_data)


    # Get list of files and sort them by the numerical part in their filenames


# Example usage
path = "/Volumes/T7/simulations/trimer/k30sig1a0.9b1c1.1/configs"
lammps_folder_pattern = "equi_lammps_config"
lammps_traj_pattern = "lammpsoutXYZ/dump_tw*.lammptrj"
#phos = [1.07, 1.1, 1.14, 1.16, 1.18, 1.19, 1.195, 1.2]
temps = [0.92, 0.95, 0.97]

for temp in temps:
    path_folder = f"{path}/temp{temp}/dt0.01/"
    list_folders = fast_scandir(path_folder, lammps_folder_pattern)
    for subfolder in list_folders:
        input_dir = os.path.join(subfolder, 'lammpsoutXYZ/')
        output_file = os.path.join(input_dir, "dump.lammptrj")  # Path for the output file
        concatenate_lammps_files(input_dir, output_file)