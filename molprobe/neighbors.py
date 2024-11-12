import grispy as gsp
import numpy as np

def create_neigbors(positions, skin, length_box):
    periodic = {0: (0, length_box), 1: (0, length_box), 2: (0, length_box)}
    grid = gsp.GriSPy(positions, periodic=periodic, N_cells=int(length_box/skin))
    _, ind = grid.bubble_neighbors(positions, distance_upper_bound=skin)
    out = []
    for i, elt in enumerate(ind):
        #ind = i - i%3
        mol_ind = np.array([i]) #, i+1, i+2])
        #print(elt)
        #print(np.setxor1d(elt, mol_ind))
        #print(np.intersect1d(elt, np.setxor1d(elt, mol_ind)))
        out.append(np.intersect1d(elt, np.setxor1d(elt, mol_ind)))
        
    return out

def common_neighbors(neigh1, neigh2):
    out_array = []

    for elt1, elt2 in zip(neigh1, neigh2):
        out_array.append(len(np.intersect1d(elt1, elt2)))
    
    return np.array(out_array)

def create_overlap(init_positions, positions, skin, length_box):
    periodic = {0: (0, length_box), 1: (0, length_box), 2: (0, length_box)}
    grid = gsp.GriSPy(positions, periodic=periodic, N_cells=int(length_box/skin))
    _, ind = grid.bubble_neighbors(init_positions, distance_upper_bound=skin)
    return ind