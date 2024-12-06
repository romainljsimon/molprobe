import os
import numpy as np

def nearest_image(pos, cube_length):
    pos = np.where(pos < 0, pos+cube_length, pos)
    pos = np.where(pos > cube_length, pos-cube_length, pos)
    return pos

def calc_cm_pos(mol_pos, cube_length):
    mol_pos = nearest_image(mol_pos, cube_length)
    rel_pos = np.copy(mol_pos)
    for i, elt in enumerate(mol_pos[1:]):
        diff = rel_pos[i+1] - rel_pos[i]
        mask1, mask2 = diff > cube_length/2, diff < -cube_length/2
        rel_pos[i+1, mask1] = rel_pos[i+1, mask1] - cube_length
        rel_pos[i+1, mask2] = -rel_pos[i+1, mask2] + cube_length
    cm_pos = np.mean(rel_pos, axis=0)
    cm_pos = nearest_image(cm_pos, cube_length)
    return cm_pos

def calc_u_vector(cm_pos, mol_pos, types):
    mask = np.array(types) == 2
    u_vector = np.array(mol_pos)[mask] - cm_pos
    u_vector = u_vector / np.linalg.norm(u_vector)
    return u_vector.flatten()

def parse_lammps_trajectory(file_path):
    """
    Parse a LAMMPS trajectory file (.lammpstrj) and extract data.

    Args:
        file_path (str): Path to the LAMMPS trajectory file.

    Returns:
        dict: A dictionary containing trajectory data with the following structure:
            {
                timestep: {
                    "box": [x_bounds, y_bounds, z_bounds],
                    "atoms": [
                        {"id": id, "type": type, "x": x, "y": y, "z": z, ...}, ...
                    ]
                }
            }
    """
    trajectory_data = {}
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, "r") as f:
        while True:
            # Read the timestep
            line = f.readline()
            if not line:
                break
            
            if "ITEM: TIMESTEP" in line:
                timestep = int(f.readline().strip())
                trajectory_data[timestep] = {}
                
                # Read the number of atoms
                f.readline()  # ITEM: NUMBER OF ATOMS
                num_atoms = int(f.readline().strip())
                
                # Read the box bounds
                f.readline()  # ITEM: BOX BOUNDS
                box_bounds = []
                for _ in range(3):
                    bounds = list(map(float, f.readline().strip().split()))
                    box_bounds.append(bounds)
                trajectory_data[timestep]["box"] = box_bounds
                
                # Read atom data
                f.readline()  # ITEM: ATOMS id type x y z ...
                atom_data = []
                for _ in range(num_atoms):
                    atom_info = f.readline().strip().split()
                    atom_data.append({
                        "id": int(atom_info[0]),
                        "type": int(atom_info[1]),
                        "mol": int(atom_info[2]),
                        "x": float(atom_info[3]),
                        "y": float(atom_info[4]),
                        "z": float(atom_info[5]),
                        # Add more fields if needed (e.g., vx, vy, vz)
                    })
                trajectory_data[timestep]["atoms"] = atom_data
    return trajectory_data

def write_lammps_mol_trajectory(output_path, trajectory_data):
    """
    Write trajectory data back to a LAMMPS trajectory file.

    Args:
        output_path (str): Path to save the output LAMMPS trajectory file.
        trajectory_data (dict): The trajectory data in the following format:
            {
                timestep: {
                    "box": [[xlo, xhi], [ylo, yhi], [zlo, zhi]],
                    "atoms": [
                        {"id": id, "type": type, "x": x, "y": y, "z": z, ...}, ...
                    ]
                }
            }
    """
    with open(output_path, "w") as f:
        for timestep, data in trajectory_data.items():
            # Write timestep
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{timestep}\n")
            
            # Write number of atoms
            num_atoms = len(data["atoms"]) // 3
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{num_atoms}\n")
            
            # Write box bounds
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            for bounds in data["box"]:
                f.write(f"{bounds[0]} {bounds[1]}\n")
                cube_length = bounds[1]
            # Write atom data
            f.write("ITEM: ATOMS id type x y z vx vy vz\n")
            mol_id = 1
            mol_pos = []
            types = []
            for atom in data["atoms"]:
                if atom["mol"] != mol_id:
                    pos_cm = calc_cm_pos(np.array(mol_pos), cube_length)
                    u_vector = calc_u_vector(pos_cm, mol_pos, types)
                    atom_line = f"{mol_id} 1 {pos_cm[0]:.5f} {pos_cm[1]:.5f} {pos_cm[2]:.5f} {u_vector[0]:.5f} {u_vector[1]:.5f} {u_vector[2]:.5f}\n"
                    f.write(atom_line)
                    mol_pos = []
                    types=[]
                    mol_id += 1
                types.append(atom["type"])
                mol_pos.append([atom['x'], atom['y'], atom['z']])
            pos_cm = calc_cm_pos(np.array(mol_pos), cube_length)
            u_vector = calc_u_vector(pos_cm, mol_pos, types)
            atom_line = f"{mol_id} 1 {pos_cm[0]:.5f} {pos_cm[1]:.5f} {pos_cm[2]:.5f} {u_vector[0]:.5f} {u_vector[1]:.5f} {u_vector[2]:.5f}\n"
            f.write(atom_line)