
from collections import defaultdict
import atooms.postprocessing as pp
import atooms.trajectory
import numpy
from atooms.postprocessing.helpers import setup_t_grid, linear_grid
import scipy.special as sp
import atooms
from atooms.trajectory import TrajectoryXYZ
from atooms.system import System, Particle
import io
import copy

class MoleculeTrajectory(TrajectoryXYZ):
    def __init__(self, trajectory, filename, len_mol=3, mode='r'):
        self.len_mol = len_mol
        self.create_mol_trajectory(trajectory, filename)
        super(MoleculeTrajectory, self).__init__(filename, mode)

    def create_mol_trajectory(self, trajectory, filename):
        with TrajectoryXYZ(filename, mode='w') as mol_traj:
            mol_traj.variables=['species', 'x', 'y', 'z', 'ux', 'uy', 'uz']
            for system, step in zip(trajectory, trajectory.read_steps()):
                mol_system = self.get_mol_system(system)
                mol_traj.write(mol_system, step)


    def get_mol_system(self, system):
        """
        Generate a molecular system from a particle system by grouping particles 
        into molecules, calculating their center-of-mass positions and orientations.

        Args:
            system: A system containing particles with positions and species attributes.

        Returns:
            System: A new system where each particle represents a molecule, with 
            center-of-mass position and orientation.
        """
        mol_system = []  # List to store molecular particles
        mol_pos, types = [], []  # Temporary lists for current molecule's positions and types
        self.cm_pos, self.u_vector = [], []  # Lists for molecule center-of-mass positions and orientations

        # Generate molecular indices for each particle
        mol_index = numpy.repeat(numpy.arange(len(system.particle) // self.len_mol), self.len_mol)
        current_mol_id = 0  # Tracks the current molecule ID

        # Loop through particles in the system
        for mol_id, particle in zip(mol_index, system.particle):
            if mol_id == current_mol_id:
                # Add particle data to the current molecule
                mol_pos.append(particle.position)
                types.append(particle.species)
            else:
                # Finalize the current molecule
                self._add_molecule(mol_pos, types, mol_system, system.cell)

                # Start a new molecule
                mol_pos = [particle.position]
                types = [particle.species]
                current_mol_id = mol_id

        # Finalize the last molecule
        self._add_molecule(mol_pos, types, mol_system, system.cell)

        # Create and return the new system
        return System(particle=mol_system, cell=system.cell)

    def _add_molecule(self, mol_pos, types, mol_system, cell):
        """
        Helper function to calculate center-of-mass and orientation of a molecule 
        and add it as a particle to the molecular system.

        Args:
            mol_pos: List of particle positions in the molecule.
            types: List of particle species in the molecule.
            mol_system: List where the new molecular particle will be added.
        """
        # Calculate center-of-mass position and orientation
        cm_pos = calc_cm_pos(numpy.array(mol_pos), cell.side[0])
        u_vector = calc_orientation(numpy.array(cm_pos), numpy.array(mol_pos), numpy.array(types))

        # Create a new molecular particle
        mol_particle = Particle(position=cm_pos)
        mol_particle.ux, mol_particle.uy, mol_particle.uz = u_vector

        # Add to the molecular system
        mol_system.append(mol_particle)


def nearest_image(pos, cube_length):
    pos = numpy.where(pos < 0, pos+cube_length, pos)
    pos = numpy.where(pos > cube_length, pos-cube_length, pos)
    return pos

def calc_cm_pos(mol_pos, cube_length):
    mol_pos = nearest_image(mol_pos, cube_length)
    rel_pos = numpy.copy(mol_pos)
    for i, _ in enumerate(mol_pos[1:]):
        diff = rel_pos[i+1] - rel_pos[i]
        mask1, mask2 = diff > cube_length/2, diff < -cube_length/2
        rel_pos[i+1, mask1] = rel_pos[i+1, mask1] - cube_length
        rel_pos[i+1, mask2] = -rel_pos[i+1, mask2] + cube_length
    cm_pos = numpy.mean(rel_pos, axis=0)
    cm_pos = nearest_image(cm_pos, cube_length)
    return cm_pos

def calc_orientation(cm_pos, mol_pos, types):
    mask = numpy.array(types) == '2'
    u_vector = numpy.array(mol_pos)[mask] - cm_pos
    u_vector = u_vector / numpy.linalg.norm(u_vector)
    return u_vector.flatten()


class RotationalCorrelator2(pp.correlation.Correlation):
    """
    Mean square displacement.

    If the time grid `tgrid` is None, the latter is redefined in a way
    controlled by the variable `rmax`. If `rmax` is negative
    (default), the time grid is linear between 0 and half of the
    largest time in the trajectory. If `rmax` is positive, the time
    grid comprises `tsamples` entries linearly spaced between 0 and
    the time at which the square root of the mean squared displacement
    reaches `rmax`.

    Additional parameters:
    ----------------------

    - sigma: value of the interparticle distance (usually 1.0). It is
    used to limit the fit range to extract the diffusion coefficient
    and to determine the diffusion time
    """

    symbol = 'C_l'
    short_name = 'C_2(t)'
    long_name = 'Rotational Correlator'
    phasespace = ['ux', 'uy', 'uz']

    def __init__(self, trajectory, tgrid=None, tsamples=30, l=2, norigins=None, **kwargs):
        pp.correlation.Correlation.__init__(self, trajectory, tgrid, norigins=norigins, **kwargs)
        self.l = l
        self._norigins = norigins
        #self.phasepsace = ['orientation']
        if self.grid is None:
            self.grid = linear_grid(0.0, trajectory.total_time * 0.10, tsamples)
        self._discrete_tgrid = setup_t_grid(self.trajectory, tgrid, offset=norigins != '1')

    def _get_u_vector(self):
        self._u_vector = []
        for i in range(len(self.trajectory.read_steps())):
            u=numpy.zeros((len(self.trajectory.read(0).particle), 3))
            for j, elt in enumerate(self.trajectory.read(i).particle):
                u[j] = [elt.ux, elt.uy, elt.uz]
            self._u_vector.append(u)
        
        
    
    def _compute(self):
        def f(x, y):
            cos_theta = numpy.sum(x * y, axis=1)
            cl_i = sp.eval_legendre(self.l, cos_theta)
            return numpy.mean(cl_i)

        self._get_u_vector()
        self.grid, self.value = pp.correlation.gcf_offset(f, self._discrete_tgrid, self.trajectory.block_size,
                                           self.trajectory.steps, self._u_vector)
        self.grid = [ti * self.trajectory.timestep for ti in self.grid]


