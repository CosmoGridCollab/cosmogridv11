import numpy as np
from .params import par
from numba import jit
import healpy as hp
import h5py
from astropy.cosmology import FlatLambdaCDM, FlatwCDM
from astropy.units import km, eV, Mpc
from scipy.optimize import least_squares
import os

from cosmogridv11 import utils_logging
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

def read_pkd_halos(file_name, rho_c, rho_c0, Lbox, part_mass, delta=200, Nmin=250, count="tot",
                   mode="FPNFW", VP_truncate=True, preamble="", n_read_halos=-1):
    """
    Reads the halos from a pkd binary file and fits a NFW profile if possible
    :param file_name: The file name to read
    :param rho_c: The critical density at the redshift of the file
    :param rho_c0: The critical density at z=0
    :param Lbox: The boxsize of the simulation [Mpc/h]
    :param part_mass: The mass of a single particle [Msun/h]
    :param delta: the overdensity criterion to define rvir and Mvir (default: 200, as in Schneider et. al)
    :param Nmin: Minimum number of particles in a halo (further defined with count argument)
    :param count: How to cound Nvir of the halo
        rvir: count particles inside rvir -> leads to more aggressive masking of the halos
        tot:  count total number of particles inside the halo
    :param mode: What kind of measured profile to use and what type of profile to fit
        FP: full profile, use the fixed binned mass profile to fit target halo profile
        VP: viral profile, use the binned mass profile up to rvir to fit target halo profile
            (assumes at least 25 parts inside the viral radius even if count is tot)
        NFW: fit a normal NFW profile
        TNFW: fit a truncated NFW profile
    :param VP_truncate: If mode is VP then truncate the profile to the first bin with delta_halo < delta
           (because of pkd profile bug)
    :param preamble: A string that is printed before each logging statement, e.g. can make the output clearer in case
    of multiple workers...
    :param n_read_halos: How many halos to read out from the file, defaults to -1 -> read all halos
    :return: The halos in the same data format as read from an AHF file
    """

    # dtype for the halos
    pkd_halo_dtype = np.dtype([("rPot", ("i4", 3)),
                               ("minPot", "f4"),
                               ("rcen", ("f4", 3)),
                               ("rcom", ("f4", 3)),
                               ("vcom", ("f4", 3)), # this is a typo, should be vcom (velocity)
                               ("angular", ("f4", 3)),
                               ("inertia", ("f4", 6)),
                               ("sigma", "f4"),
                               ("rMax", "f4"),
                               ("fMAss", "f4"),
                               ("fEnvironDensity0", "f4"),
                               ("fEnvironDensity1", "f4"),
                               ("rHalf", "f4"),
                               ("rvir", "f4"),
                               ("profile", ('i4', 20)),
                               ("virprofile", ('i4', 20))])
    
    out_halo_dtype = np.dtype([('ID', np.uint64),
                               ('IDhost', np.uint64),
                               ('Mvir', '<f8'),
                               ('Nvir', '<i8'),
                               ('x', '<f8'),
                               ('y', '<f8'),
                               ('z', '<f8'),
                               ('rvir', '<f8'),
                               ('cvir', '<f8'),
                               ("vcom_x", 'f4'),
                               ("vcom_y", 'f4'),
                               ("vcom_z", 'f4'),
                               ('tfNFW_cvir', '<f8'),
                               ('tfNFW_tau', '<f8'),
                               ('tfNFW_Mvir', '<f8')])

    # read the file
    LOGGER.info(preamble + "Reading in halos from file {}...".format(file_name))
    halo_data = np.fromfile(file_name, dtype=pkd_halo_dtype, count=n_read_halos)
    LOGGER.info(preamble + "Succesfully read {} halos...".format(len(halo_data)))

    # tipsy factor to transform masses to M_sun/h
    tipsy_fac = rho_c0 * Lbox ** 3
    # from tipsy mass to nparts
    tipsy_fac /= part_mass

    # now we need rho_c in M_sun h^2/kpc^3
    rho_c /= 1e9

    # from integer pos to float pos
    int_fac = 1.0 / 0x80000000

    # now we start masking, we begin with the counts
    if count == 'tot':
        Nparts = np.round(halo_data["fMAss"] * tipsy_fac, 0)
    elif count == 'vir':
        Nparts = np.sum(halo_data["virprofile"], axis=1)
    else:
        raise IOError(preamble + "This type of count is note supported: {}...".format(count))

    mask = Nparts >= Nmin
    halo_data = halo_data[mask]

    # report on selection
    reduction = 100 * (len(mask) - np.sum(mask)) / len(mask)
    LOGGER.info(preamble + "Nmin reduces the number of halos by "
              "{:5.2f}% ({} total) to {}...".format(reduction, len(mask) - np.sum(mask), len(halo_data)))

    # Now we deal with viral stuff
    if mode.startswith("VP"):

        
        # calculate the delta of the halos
        Nvir = np.sum(halo_data["virprofile"], axis=1)
        Mvir = Nvir / tipsy_fac
        delta_data = Mvir / (4.0 / 3.0 * np.pi * halo_data["rvir"] ** 3)
            
        # take only what is close to delta and has at least 25 parts in rvir
        mask = np.logical_and(np.isclose(delta_data, delta, atol=0.1), Nvir >= 25)
        halo_data = halo_data[mask]

        # report how many were selected
        reduction = 100 * (len(mask) - np.sum(mask)) / len(mask)
        LOGGER.info(preamble + "Viral profiling reduces the number of halos by {:5.2f}% ({} total) to {}...".format(reduction, len(mask) - np.sum(mask), len(halo_data)))

    LOGGER.info(preamble + "Starting with profile fitting...")

    # cycle though halos
    halo_lines = []
    halo_id = 0
    IDhost = -1
    n_fail_root = 0
    n_fail_small = 0
    for index, halo in LOGGER.progressbar(list(enumerate(halo_data)), at_level='info', desc=f'profiling halos with mode={mode}'):

        # get some params
        if mode.startswith("VP"):
            # count parts inside viral radius
            Nvir = np.sum(halo["virprofile"])
            # bins up to rvir
            r_bin = np.logspace(np.log(0.005 * 3), np.log(halo["rvir"] * Lbox), 20, base=np.e) * 1000
        
        else:
            # get the number of particles (as int)
            Nvir = int(np.round(halo["fMAss"] * tipsy_fac, 0))
            # fixed bins
            r_bin = np.logspace(np.log(0.005 * 3), np.log(5.0), 20, base=np.e) * 1000

        # set the halo id
        ID = halo_id
        halo_id += 1

        # position of the halos in kpc/h
        pos = 1000 * Lbox * (halo["rPot"] * int_fac + halo["rcen"] + 0.5)
        x = pos[0]
        y = pos[1]
        z = pos[2]

        # truncate the pkd profile
        if mode.startswith("VP"):
            
            if VP_truncate:
                
                # we truncate to the first bin where delta_halo < delta - 1 (and truncate zeros...)
                profile_parts = truncate_zeros(halo["virprofile"])
                binned_crit = 4.0 / 3.0 * np.pi * (r_bin[:len(profile_parts)]) ** 3 * rho_c
                enclosed_delta = np.cumsum(profile_parts) * part_mass / binned_crit
                part_list = []
                for n_p, e_d in zip(profile_parts, enclosed_delta):
                    if e_d <= delta - 1:
                        part_list.append(n_p)
                        break
                    else:
                        part_list.append(n_p)

                profile_parts = np.array(part_list)

                if count == 'vir' and np.sum(profile_parts) < Nmin:
                    LOGGER.debug("Unable to fit a profile to halo with id "
                              "{} because of Nmin with VP_truncate and vir counting, skipping...".format(ID))
                    continue
                
                if count == 'tot' and np.sum(profile_parts) < 25:
                    LOGGER.debug("Unable to fit a profile to halo with id "
                              "{} because of Nmin with VP_truncate and tot counting, skipping...".format(ID))
                    continue
            else:
                
                profile_parts = truncate_zeros(halo["virprofile"])
        else:
            
            profile_parts = truncate_zeros(halo["profile"])
        
        enclosed_mass = np.cumsum(profile_parts) * part_mass

        # fit
        if mode.endswith("TNFW"):
            
            res = fit_trNWF_from_enclosedMass(r_bin[:len(enclosed_mass)], enclosed_mass,
                                              c_init=6, t_init=4 * 6, rvir_init=500, rho_c=rho_c, verbose=0)

            # Mvir from TNFW is not really Mvir (see https://arxiv.org/pdf/1101.0650.pdf)
            # so we still need to find Mvir (or rvir in this case)
            fitted_enclosed_mass = lambda r: MNFWtr_fct(res["x"], r, rho_c)

            # params to save
            tr_cvir = np.abs(res["x"][0])
            tr_tau = np.abs(res["x"][1])
            tr_Mvir = np.abs(4.0 / 3.0 * res["x"][2] ** 3 * 200 * rho_c)

            # get rvir
            f_root = lambda R: fitted_enclosed_mass(R) / (4.0 / 3.0 * np.pi * R ** 3 * rho_c) - delta

            # rvir from TNW as rvir init
            r_init = res["x"][2]

        else:
            
            # fit the profile
            res = fit_NWF_from_enclosedMass(r_bin[:len(enclosed_mass)], enclosed_mass,
                                            rhos_init=1e7, rs_init=100, verbose=0)
            if not res["success"]:
                LOGGER.debug("Unable to fit a profile to halo with id {} because of least square fail, "
                          "skipping...".format(ID))
                continue

            # function for the enclosed mass
            fitted_params = res["x"]
            fitted_enclosed_mass = lambda r: 4 * np.pi * fitted_params[0] * fitted_params[1] ** 3 * \
                                             (np.log((fitted_params[1] + r) / fitted_params[1]) - r / (
                                                         fitted_params[1] + r))

            # params to save
            tr_cvir = -1.0
            tr_tau = -1.0
            tr_Mvir = -1.0

            # get rvir
            f_root = lambda R: fitted_enclosed_mass(R) / (4.0 / 3.0 * np.pi * R ** 3 * rho_c) - delta

            # rs as rvir init
            r_init = fitted_params[1]

        # binary search
        r_low = 0.01 * r_init
        f_low = f_root(r_low)
        r_high = 100.0 * r_init
        f_high = f_root(r_high)

        try:
            rvir, fvir = binary_root_finder(f_root, r_low, r_high, f_low, f_high)
            if index % 100 == 0:
                pkd_rvir = halo["rvir"] * Lbox * 1000
                LOGGER.debug(preamble + "Halo ID {:>8d}, PKD rvir {:5.2f},  NFW rvir {:5.2f}, Diff {:5.2f} ".format(ID, pkd_rvir, rvir, (pkd_rvir-rvir)/rvir*100))
            # LOGGER.debug(preamble + "Halo ID:  {}".format(ID))
            # LOGGER.debug(preamble + "PKD rvir: {}".format(halo["rvir"] * Lbox * 1000))
            # LOGGER.debug(preamble + "NFW rvir: {}".format(rvir))
            # LOGGER.debug(preamble + "Diff:     {}".format(100 * (halo["rvir"] * Lbox * 1000 - rvir) / rvir, "%"))
        
        except Exception as err:
            n_fail_root += 1
            LOGGER.debug(preamble + "Halo ID {:>8d}, unable to fit a profile to halo because of rvir root finder fail, errmsg={} skipping...".format(ID, str(err)))
            continue

        # calculate the other params
        Mvir = 4.0 / 3.0 * np.pi * rvir ** 3 * rho_c * delta

        # less than 1e12 Msun/h is unsensible and leads to a negative fraction of satelite galaxies
        if Mvir < 1e12:
            n_fail_small += 1
            LOGGER.debug(preamble + "Halo ID {:>8d}, unable to fit a profile to halo with id because of Mvir too small, skipping...".format(ID))
            continue

        cvir = -1.0 if mode.endswith("TNFW") else rvir / r_init

        # append the date
        halo_lines.append((ID, IDhost, Mvir, Nvir, x, y, z, rvir, cvir, halo['vcom'][0], halo['vcom'][1], halo['vcom'][2], tr_cvir, tr_tau, tr_Mvir))

    LOGGER.info(preamble + "Finished fitting halos in file {}, fitted {}/{} halos, n_fail_small={}, n_fail_root={}".format(file_name, len(halo_lines), len(halo_data), n_fail_small, n_fail_root))

    return np.array(halo_lines, dtype=out_halo_dtype)


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
