import numpy as np
from .params import par
from numba import jit
import healpy as hp
import h5py
from astropy.cosmology import FlatLambdaCDM, FlatwCDM
from astropy.units import km, eV, Mpc
from scipy.optimize import least_squares
import os, sys

from cosmogridv11 import utils_logging, utils_arrays
LOGGER = utils_logging.get_logger(__file__)

def build_cosmo(param, verbosity=0):
    """
    Builds a astropy cosmology out of the baryonification paramter
    :param param: baryonification paramter
    :param verbosity: How much output should be generated
    :return: cosmology object from astropy
    """

    if np.isclose(param.cosmo.w0, -1.0):
        if verbosity > 0:
            print("w0 is very close to -1, using LambdaCDM")
        cosmo = FlatLambdaCDM(H0=100*param.cosmo.h0, Ob0=param.cosmo.Ob, Om0=param.cosmo.Om, Tcmb0=2.7255,
                              Neff=3.046, m_nu=param.cosmo.m_nu*eV)
    else:
        cosmo = FlatwCDM(H0=100*param.cosmo.h0, Ob0=param.cosmo.Ob, Om0=param.cosmo.Om, Tcmb0=2.7255,
                         Neff=3.046, m_nu=param.cosmo.m_nu*eV, w0=param.cosmo.w0)

    return cosmo

def check_points(points, min_val, max_val, buffer=1e-9):
    """
    A function that checks if all points are either within min_val or large than max_val
    used to check if a box is completely inside the lightcone or completely outside
    :param points: an array of points with shape [N_points, N_dims]
    :param min_val: inner radius of the shell
    :param max_val: outer radius of the shell
    :param buffer: subtracted from min_val and added to max_val to stabilize floating point comparision
    :return: False if all points are outside of the shell or inside, Ture otherwise
    """

    assert buffer > 0, "Buffer has to be positive"

    # get the radi of the points
    r_points = np.linalg.norm(points, axis=1)

    # if all points are larger than the outer shell return false
    if np.all(r_points > max_val + buffer):
        return False
    # if all points are smaller than the outer shell return false
    if np.all(r_points < min_val - buffer):
        return False
    # Otherwise we return True
    return True


def get_offset_indices(boxsize, min_val, max_val, max_repli=7):
    """
    Gets all indices that intersect with a given shell
    :param boxsize: Size of the box
    :param min_val: inner radius of the shell
    :param max_val: outer radius of the shell
    :param max_repli: maximum number of replicates per side
    :return: a list of np.array([i,j,k]) arrays with the offset indices
    """

    assert max_repli > 0, "Number of replciates has to be bigger than 0"

    # list for indices
    box_indices = []

    # cycle through all boxes
    for i in range(-max_repli, max_repli):
        for j in range(-max_repli, max_repli):
            for k in range(-max_repli, max_repli):
                # set the origin
                origin = np.array([i, j, k])*boxsize
                # check all the corners
                corners = np.zeros((8, 3))
                corner_index = 0
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            # set corner offset
                            corner_offset = np.array([ii, jj, kk])*boxsize
                            # set corner
                            corners[corner_index] = origin + corner_offset
                            # update
                            corner_index += 1
                # check
                check_box = check_points(corners, min_val, max_val)
                if check_box:
                    box_indices.append(np.array([i,j,k]))

    return box_indices

def read_shell(file_name, shell_type=None, verbosity=0):
    """
    Reads in the particles from a shell given its format
    :param file_name: file name of the shell
    :param shell_type: data format (currently only healpy FITS), it is infered if None
    :param verbosity: how much output should be printed
    :return: particles of the shell
    """


    if shell_type is None:
        root, ext = os.path.splitext(file_name)
        if ext == ".fits":
            shell_type = 'FITS'
        else:
            raise IOError("Shell format could not be infered from file name {}".format(file_name))

    if shell_type == 'FITS':
        parts = hp.read_map(file_name, verbose=verbosity)
        return parts
    else:
        raise IOError("Shell format nor supported {}".format(shell_type))


@jit
def get_split_indices_numba(array):
    """
    Numba compiled version of <get_split_indices>
    """
    old_val = array[0]
    split_indices = []
    for i in range(1, len(array)):
        val = array[i]
        if val != old_val:
            split_indices.append(i)
        old_val = val
    return split_indices


def get_split_indices(array, sort=False):
    """
    returns indices such that the array is split into segments with identical entries
    Example:
    test_array = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4])
    splits = get_split_indices(test_array) # splits = [3, 5, 9]
    np.split(test_array, indices_or_sections=splits) -

    results in:

    [array([1, 1, 1]), array([2, 2]), array([3, 3, 3, 3]), array([4, 4])]

    :param array: 1d array to get the split indices from
    :param sort: sort the array before splitting
    :return: list of indices that can be passed to np.split
    """
    assert array.ndim == 1, "Array has to be 1d..."

    if sort:
        array = np.sort(array)

    # call the numba compiled version
    return get_split_indices_numba(array)


def get_intersection_angle(r1, r2, d):
    """
    Calculates the angle of the intersection of two spheres. The first located at (0, 0, 0) with radius r1 and the
    second located at (0, 0, d) with radius r2, can be used for the healpix query_disc routine
    :param r1: radius of the first sphere (e.g. the shell)
    :param r2: radius of the second sphere (e.g. the halo)
    :param d: distance of the spheres (e.g. comoving distance of the halo)
    :return: the angle that can be used the query the relevant pixels. If the spheres do not overlap, None is returned
    """

    # check if the spheres overlap or are inside of each other
    if d > r1 + r2 or d < np.abs(r1 - r2):
        return None
    # return the intersection angle (Kosinussatz)
    return np.arccos((r1**2 + d**2 - r2**2)/(2.0*r1*d))


def bunch_to_lists(bunch):
    """
    Transforms a bunch object into a nested list of tuples that can be used for saving and restoring
    The reverse operation can be achieved with lists_to_bunch
    :param bunch: a bunch e.g. baryonification params
    :return: a nested list
    """
    main_groups = []
    for key in bunch.__dict__.keys():
        sub_bunch = getattr(bunch, key)
        sub_group = []
        for sub_key in sub_bunch.__dict__:
            sub_group.append((sub_key, getattr(sub_bunch, sub_key)))
        main_groups.append((key, sub_group))
    return main_groups


def lists_to_bunch(lists):
    """
    Inverse operation of bunch_to_lists. Transform a nested list into baryonifiaction params
    :param lists: a nested structure of lists
    :return: baryonification params
    """
    params = par()
    for main_group in lists:
        main_key = main_group[0]
        sub_group = main_group[1]
        sub_bunch = getattr(params, main_key)
        for key, value in sub_group:
            setattr(sub_bunch, key, value)
    return params


def dump_lists_to_h5(group: h5py.Group, lists):
    """
    Dumps the lists into a h5 group to save baryonification params on disk in h5 format
    The reverse can be achieved with get_lists_from_h5
    :param group: the h5 group object to save the stuff
    :param lists: a nestes list, e.g. from bunch_to_lists
    """
    for main_group in lists:
        main_key = main_group[0]
        sub_lists = main_group[1]
        sub_group = group.create_group(main_key)
        for key, value in sub_lists:
            sub_group.attrs[key] = value


def get_lists_from_h5(group: h5py.Group):
    """
     Reads out the baryonifiaction params from a h5 group and returns structured lists
     :param group: a h5 group to extract the data from
     :return: a nested structure of lists, can be used as input to lists_to_bunch to retrieve the params as bunch
    """

    main_groups = []
    for main_key in group.keys():
        sub_list = []
        sub_group = group[main_key]
        for sub_key in sub_group.attrs:
            sub_list.append((sub_key, sub_group.attrs[sub_key]))
        main_groups.append((main_key, sub_list))
    return main_groups


def part_dict_to_map(particle_dict, hp_map):
    """
    Adds the paricles of a particle_dict to a given map (inplace)
    :param particle_dict: particle dict containing hp indices as keys and number of parts as values
    :param hp_map: a map (1d array) with enough pixels (high enough nside)
    """

    for index, value in particle_dict.items():
        hp_map[index] += value


@jit
def numba_asign(array, indices, weights, adds):
    """
    This is a very easy function that wraps a for loop into a numba jit to get a factor ~10 in performance
    :param array: array to tranform
    :param indices: indicies to add stuff to
    :param weights: wights to multiply adds with
    :param adds: what we should add at each index
    :return:
    """
    for i in range(len(indices)):
        for j in range(4):
            array[indices[i,j]] += adds[i]*weights[i,j]


def truncate_zeros(array):
    """
    Returns a truncated version of the input array, such that the array end with a non zero element
    :param array: 1d array
    :return: the array where all tailing zeros are truncated
    """
    for i in reversed(range(len(array))):
        if array[i] != 0:
            break
    return array[:i + 1]


def binary_root_finder(f, x_low, x_high, f_low, f_high, atol=1e-6):
    """
    A simple binary search root finder for a monotone decreasing function 1d function
    It assumes that f_high < 0 < f_low
    :param f: The function to find the root
    :param x_low: The lowest possible value of the interval
    :param x_high: The highest possible value of the interval
    :param f_low: f(x_low) must be > 0
    :param f_high: f(x_high) must be < 0
    :param atol: absolute tolerance for the root, the algorithm terminates once |f(x_root)| < atol
    :return: x_root and f(x_root)
    """

    if not (f_low > 0 and f_high < 0):
        raise ValueError("This function expects f_low > 0 and f_high < 0. " + \
                         "f_low: {}, f_high: {}".format(f_low, f_high))

    if np.abs(f_low) < atol:
        return x_low, f_low
    if np.abs(f_high) < atol:
        return x_high, f_high

    x_new = 0.5 * (x_low + x_high)
    f_new = f(x_new)

    if f_new < 0:
        return binary_root_finder(f, x_low, x_new, f_low, f_new, atol)
    else:
        return binary_root_finder(f, x_new, x_high, f_new, f_high, atol)

def fit_NWF_from_enclosedMass(r_bins, M_bins, rhos_init, rs_init, verbose=0):
    """
    Fits a NWF profile onto a array of enclosed masses. It was compared to the colossus routine and gives almost
    identical results but ~2 times faster. The improvement comes from the fact that colossus integrates the density
    of the halo to get the enclosed mass while this function uses the analytical form.
    :param r_bins: The radii at which the enclosed masses are defined [kpc/h]
    :param M_bins: The enclosed massed at given radii [Msun/h]
    :param rhos_init: Initial rhos NFW density parameter
    :param rs_init: Initial rs NFW radius parameter
    :param verbose: If output should be printed out
    :return: The same output as scipy.optimize.least_squares produces
    """
    diff_func = lambda x: 4*np.pi*x[0]*x[1]**3* (np.log((x[1] + r_bins)/x[1]) - r_bins/(x[1] + r_bins)) - M_bins
    x_init = np.array([rhos_init, rs_init])

    return least_squares(diff_func, x_init, verbose=verbose)


def mNFWtr_fct(x, t):
    """
    Truncated NFW mass profile. Normalised.
    """
    pref = t ** 2.0 / (1.0 + t ** 2.0) ** 3.0 / 2.0
    first = x / ((1.0 + x) * (t ** 2.0 + x ** 2.0)) * (
                x - 2.0 * t ** 6.0 + t ** 4.0 * x * (1.0 - 3.0 * x) + x ** 2.0 + 2.0 * t ** 2.0 * (1.0 + x - x ** 2.0))
    second = t * ((6.0 * t ** 2.0 - 2.0) * np.arctan(x / t) + t * (t ** 2.0 - 3.0) * np.log(
        t ** 2.0 * (1.0 + x) ** 2.0 / (t ** 2.0 + x ** 2.0)))
    return pref * (first + second)


def MNFWtr_fct(x, r_bins, rho_c):
    """
    Truncated NFW mass profile. With just on arg for scipy least squares
    """
    # everything must be positive
    c = np.abs(x[0])
    t = np.abs(x[1])
    rvir = np.abs(x[2])
    Mvir = 4.0/3.0*np.pi*rvir**3*200*rho_c
    return Mvir*mNFWtr_fct(c*r_bins/rvir,t)/mNFWtr_fct(c,t)


def fit_trNWF_from_enclosedMass(r_bins, M_bins, c_init, t_init, rvir_init, rho_c, verbose=0):
    """
    Fits a truncated NWF profile onto a array of enclosed masses. It was compared to the colossus routine and gives almost
    idetically results but ~2 times faster. The improvement comes from the fact that colossus integrates the density
    of the halo to get the enclosed mass while this function uses the analytical form.
    :param r_bins: The radii at which the enclosed masses are defined [kpc/h]
    :param M_bins: The enclosed massed at given radii [Msun/h]
    :param ...
    :param verbose: If output should be printed out
    :return: The same output as scipy.optimize.least_squares produces
    """

    diff_func = lambda x: MNFWtr_fct(x, r_bins, rho_c) - M_bins

    x_init = np.array([c_init, t_init, rvir_init])
    
    return least_squares(diff_func, x_init, verbose=verbose)


def read_pkd_halos(file_name, preamble='', count=-1):

    """
    Reads the halos from a pkd binary file and fits a NFW profile if possible
    :param file_name: The file name to read
    :param preamble: A string that is printed before each logging statement, e.g. can make the output clearer in case
    of multiple workers...
    :return: The halos in the same data format as read from an AHF file
    """

    # dtype for the halos
    pkd_halo_dtype = np.dtype([("rPot", ("i4", 3)),
                               ("minPot", "f4"),
                               ("rcen", ("f4", 3)),
                               ("rcom", ("f4", 3)),
                               ("vcom", ("f4", 3)), 
                               ("angular", ("f4", 3)),
                               ("inertia", ("f4", 6)),
                               ("sigma", "f4"),
                               ("rMax", "f4"),
                               ("fMass", "f4"),
                               ("fEnvironDensity0", "f4"),
                               ("fEnvironDensity1", "f4"),
                               ("rHalf", "f4"),
                               ("fRvir", "f4"),
                               ("profile", ('i4', 20)),
                               ("virprofile", ('i4', 20))])
    
    # read the file
    halo_data = np.fromfile(file_name, dtype=pkd_halo_dtype, count=count)
    LOGGER.info(preamble + "Successfully read {} halos from {}".format(len(halo_data), file_name))

    return halo_data


def get_halofiles_from_z_boundary(z_boundaries, log_file, prefix):
    """
    Creates a list of halo files matching the z_boundaries
    :param z_boundaries: A 2D array with shapes [n_shells, 2] where each row contains the lower and upper redshift
                         bounds of a shell
    :param log_file: The log file of the simulation to map the z_boundaries to steps
    :param prefix: The prefix of the halo files, will be appended with <step>.fofstats.0
    :return: A list of halo files with the same length as z_boundaries
    """

    # get the redshifts from the log file, this takes also care of restarts
    z_out = np.sort(np.unique(np.genfromtxt(log_file)[:,1]))[::-1]

    # map z_boundaries to steps
    steps = []
    
    for z_boundary in z_boundaries:
        index_set = np.where(np.isclose(z_boundary[0], z_out))[0]
        if index_set.size != 1:
            raise ValueError(f"No match found in log file for boundary {z_boundary}!")
        steps.append(index_set[0])

    # get the halo files
    halo_files = []
    for step in steps:
        halo_files.append(f"{prefix}.{step:05g}.fofstats.0")

    return halo_files


def match_shell(shell_info, halo_shell):
    """
    This functions matches the shell infos from the shell file and from the halo file, such that you get the shell and
    shell info from the one corresponding to the halo_shell
    :param shell_info: A 1D array with length n_shells containing the shell info
    :param halo_shell: The halo_shell to match
    :return: The index of shells and shell_info that match halo_shell, if none is found a value error is raised
    """

    # match lower z
    index_set = np.where(np.isclose(shell_info['lower_z'], halo_shell['lower_z']))[0]
    if index_set.size != 1:
        raise ValueError(f"No match found for lower z: {halo_shell['lower_z']} of halo shell!")
    lower_index = index_set[0]

    # match upper z
    index_set = np.where(np.isclose(shell_info['upper_z'], halo_shell['upper_z']))[0]
    if index_set.size != 1:
        raise ValueError(f"No match found for upper z: {halo_shell['upper_z']} of halo shell!")
    upper_index = index_set[0]

    if upper_index != lower_index:
        raise ValueError(f"Upper and lower index do not match: {upper_index} vs {lower_index}")

    return lower_index
    
def dequantize_pkd_halos(halo_data, Lbox, rho_c0, part_mass, preamble=''):
    """
    :param halo_data: rec array output of function read_pkd_halos
    :param rho_c0: The critical density at z=0
    :param part_mass: The mass of a single particle [Msun/h]
    :param Lbox: The boxsize of the simulation [Mpc/h]
    # add position and unique id
    :return: The input halo_data with additional columns contains dequantized quantities [x, y, z, mass, rvir]
    """

    pos = dequantize_halo_pos(halo_data["rPot"], halo_data["rcen"], Lbox)
    halo_data = utils_arrays.add_cols(halo_data, names=['x:f4', 'y:f4', 'z:f4'])
    halo_data['x'] = pos[:,0]
    halo_data['y'] = pos[:,1]
    halo_data['z'] = pos[:,2]

    tipsy_fac = get_tipsy_fac(rho_c0, Lbox, part_mass)
    halo_data = utils_arrays.add_cols(halo_data, names=['mass:f8', 'rvir:f4'])
    halo_data['mass'] = np.round(halo_data["fMass"] * tipsy_fac, 0) * part_mass
    halo_data['rvir'] = halo_data["fRvir"] * Lbox

    return halo_data

def dequantize_halo_pos(rPot, rcen, Lbox):

    # from integer pos to float pos
    int_fac = 1.0 / 0x80000000
    pos = 1000 * Lbox * (rPot * int_fac + rcen + 0.5)
    return pos



def get_tipsy_fac(rho_c0, Lbox, part_mass):

    # tipsy factor to transform masses to M_sun/h
    tipsy_fac = rho_c0 * Lbox ** 3
    # from tipsy mass to nparts
    tipsy_fac /= part_mass

    return tipsy_fac


def load_baryonification_params(fname):

    from cosmogridv11 import baryonification
    sys.path.append(os.path.join(baryonification.__path__[0], '..')) # this is for back-compat with baryonification_params.py scripts
    import importlib.util
    spec = importlib.util.spec_from_file_location("baryonification_params.py", fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["baryonification_params.py"] = mod
    spec.loader.exec_module(mod)
    LOGGER.info(f'using baryonification_params={mod}')
    return mod.par

def get_offset_indices(boxsize, min_val, max_val, max_repli=7):
    """
    Gets all indices that intersect with a given shell
    :param boxsize: Size of the box
    :param min_val: inner radius of the shell
    :param max_val: outer radius of the shell
    :param max_repli: maximum number of replicates per side
    :return: a list of np.array([i,j,k]) arrays with the offset indices
    """

    assert max_repli > 0, "Number of replciates has to be bigger than 0"

    # list for indices
    box_indices = []

    # cycle through all boxes
    for i in range(-max_repli, max_repli):
        for j in range(-max_repli, max_repli):
            for k in range(-max_repli, max_repli):
                # set the origin
                origin = np.array([i, j, k])*boxsize
                # check all the corners
                corners = np.zeros((8, 3))
                corner_index = 0
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            # set corner offset
                            corner_offset = np.array([ii, jj, kk])*boxsize
                            # set corner
                            corners[corner_index] = origin + corner_offset
                            # update
                            corner_index += 1
                # check
                check_box = check_points(corners, min_val, max_val)
                if check_box:
                    box_indices.append(np.array([i,j,k]))

    return box_indices



