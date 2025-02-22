from __future__ import print_function
from __future__ import division
from .utils import *
from .profiles import *
from astropy.units import Mpc, Msun
from scipy.integrate import quad
from multiprocessing import Pool, cpu_count
# this type of pool spawn non daemonic processes, meaning you can have "grandchildren",
# i.e. the spawned process can spawn more processes
from concurrent.futures import ProcessPoolExecutor
from collections import namedtuple
import healpy as hp
import numpy as np
import h5py
import os
import time

from cosmogridv11 import utils_logging, utils_arrays
LOGGER = utils_logging.get_logger(__file__)


def get_single_halo_shell_wrapper(args):
    """
    A quick and dirty wrapper for the imap routine of the multiprocessing module.
    Unpacks the arguments for get_single_halo_shell
    :param args: args for get_single_halo_shell
    :return: get_single_halo_shell(*args)
    """
    return get_single_halo_shell(*args)

def get_shell_cov_dist(z_boundary, cosmo):

    # get the comoving distances
    from astropy.units import Mpc, Msun
    cov_inner = cosmo.comoving_distance(z=z_boundary[0]).to(Mpc/cosmo.h).value
    cov_outer = cosmo.comoving_distance(z=z_boundary[1]).to(Mpc/cosmo.h).value
    # we take the same shell comoving distance as UFalcon
    shell_cov = cosmo.comoving_distance(z=np.mean(z_boundary)).to(Mpc/cosmo.h).value

    return shell_cov, cov_inner, cov_outer



def get_single_halo_shell(param, halo_file, z_boundary, i_file, max_repli=7, test=False):
    """
    Creates a halo file for the lightcone, given the halo files of the snapshots and their cut out shells
    :param param: baryonification params
    :param halo_file: The halo file to read halos from
    :param z_boundary: array of length 2 containing the lower and upper redshift bound of the shell
    :param i_file: The number of the file used as shell_id
    :param max_repli: maximum number of replicates
    :return: name of the written file
    """

    # Set the logging preamble
    preamble = "File ID {}: ".format(i_file)

    # check that int8 is enough for the shell index
    # if i_file > 255:
    #     raise ValueError(preamble + "The shell index must be representable as int8, therefore a maximum number of 255 "
    #                                 "shells is allowed!")

    # get a cosmology object
    cosmo = build_cosmo(param=param)

    # save the relevant lines
    relevant_lines = []

    # a list for the shell file
    shell_dtype = np.dtype([("shell_id", np.int32),
                            ("lower_z", np.float32), 
                            ("upper_z", np.float32),
                            ("lower_com", np.float32), 
                            ("upper_com", np.float32), 
                            ("shell_com", np.float32)])

    # print out
    LOGGER.info(preamble + "Starting with file {} and boundaries {}".format(halo_file, z_boundary))


    shell_cov, cov_inner, cov_outer = get_shell_cov_dist(z_boundary, cosmo)
    
    # create the shell data (shell_id, lower z, upper z, lower com, upper com, shell com)
    shell_data = np.array([(i_file, z_boundary[0], z_boundary[1], cov_inner, cov_outer, shell_cov)],
                            dtype=shell_dtype)

    # get the param mask
    if param.files.halofile_format == 'AHF-TEXT':
        # here the last 3 are just place holders
        names = "ID,IDhost,Mvir,Nvir,x,y,z,rvir,cvir,tfNFW_cvir,tfNFW_tau,tfNFW_Mvir"
        h = np.genfromtxt(halo_file, 
                          usecols=(0, 1, 3, 4, 5, 6, 7, 11, 42, 12, 13, 14), 
                          comments='#',
                          dtype=None, 
                          names=names)

        # set defaults
        h["tfNFW_cvir"] = -1.0
        h["tfNFW_tau"] = -1.0
        h["tfNFW_Mvir"] = -1.0

        # use only halos that have sensible values
        base_mask = np.logical_and(h['Nvir'] >= param.sim.Nmin_per_halo,
                                   np.logical_and(h['cvir'] > 0, h['IDhost'] < 0.0))
        h = h[base_mask]

    elif param.files.halofile_format == "PKD-BINARY":
        
        # check if we already did it
        # if os.path.exists(halo_file + ".npy"):
        #     LOGGER.info(preamble + "Found already fitted halos, using them... \n WARNING: This can lead to errors if "
        #                      "<Nmin_per_halo> changed!")
        #     h = np.load(halo_file + ".npy")

        # else:

        # get rho_c at given redshift
        z_shell = np.mean(z_boundary)
        rho_c = cosmo.critical_density(z_shell).to(cosmo.h ** 2 * Msun / Mpc ** 3).value
        rho_c0 = cosmo.critical_density0.to(cosmo.h ** 2 * Msun / Mpc ** 3).value

        # particle mass
        rho_crit = cosmo.critical_density(0).to(cosmo.h ** 2 * Msun / Mpc ** 3).value
        part_mass = cosmo.Om0 * rho_crit
        part_mass *= (param.sim.Lbox / param.sim.nparts) ** 3

        # get the halos
        h = read_pkd_halos(file_name=halo_file,
                           rho_c=rho_c,
                           rho_c0=rho_c0,
                           Lbox=param.sim.Lbox,
                           part_mass=part_mass,
                           delta=200,
                           Nmin=param.sim.Nmin_per_halo,
                           preamble=preamble,
                           mode=param.code.mode,
                           VP_truncate=param.code.VP_truncate,
                           count=param.code.count,
                           n_read_halos=100 if test else -1)

        # if save_intermediate:
        #     np.save(halo_file + ".npy", h)

    else:
        raise ValueError(preamble + "This halo format is currently not supported {}...".format(param.files.halofile_format))

    # update the ids
    h["ID"] = h["ID"] + i_file*10**8

    # N tot for counting
    N_tot = 0

    # get the positions and transform
    halos = np.array([h["x"], h["y"], h["z"]]).T/1000.0  # to Mpc

    # copy the halo lines
    halo_lines = h.copy()

    assert len(halo_lines) == len(halos), preamble + "There seems to be something wrong with the halo file, " \
                                                     "more than one comment on top?"

    # get the offset indices
    offset_indices = get_offset_indices(boxsize=param.sim.Lbox, min_val=cov_inner, max_val=cov_outer,
                                        max_repli=max_repli)

    # cycle and project
    old_offset = np.zeros(3)

    # select all relevant halos
    for offset in offset_indices:
        
        # get the new offset and add to positions
        mpc_offset = (offset - old_offset) * param.sim.Lbox
        halos += mpc_offset

        # update halos
        halo_lines['x'] += mpc_offset[0] * 1000
        halo_lines['y'] += mpc_offset[1] * 1000
        halo_lines['z'] += mpc_offset[2] * 1000

        # update
        old_offset = offset

        # convert to radii
        radii = np.linalg.norm(halos, axis=1)

        # get the masks
        mask_inner = np.abs(radii - cov_inner) < param.code.halo_buffer
        mask_middle = np.logical_and(radii >= cov_inner, radii < cov_outer)
        mask_outer = np.abs(radii - cov_outer) < param.code.halo_buffer
        n_pass = np.sum(mask_inner) + np.sum(mask_middle) + np.sum(mask_outer)
        N_tot += n_pass
        LOGGER.info(preamble + "Current offset: {:>10s}, Total number of collected halos: {}/{}".format(str(offset), N_tot, len(halos)))

        # if there are no halos in the shell do nothing
        if n_pass == 0:
            continue

        # get what is relevant + shell id
        for id_offset, mask in zip([-1, 0, 1], [mask_inner, mask_middle, mask_outer]):
            local_pass = np.sum(mask)
            if local_pass > 0:
                extended_halo_dtype = halo_lines.dtype.descr + [("shell_id", np.int32), ("halo_buffer", np.int8)]
                relevant = np.zeros(local_pass, dtype=extended_halo_dtype)
                relevant["shell_id"] = i_file + id_offset
                relevant["halo_buffer"] = id_offset
                masked_lines = halo_lines[mask]
                for name in halo_lines.dtype.names:
                    # if name != "shell_id":
                    relevant[name] = masked_lines[name]
                relevant_lines.append(relevant)

    LOGGER.warning(preamble + "Finished file: {}, total number of halos: {}".format(halo_file, N_tot))

    return relevant_lines, shell_data


def prep_lightcone_halos(param, halo_files, shells_z, max_repli=7, test=False):
# def prep_lightcone_halos(param, halo_files, shells_z, output_path="./", max_repli=7, out_type='AHF-NUMPY', test=False):
    """
    Creates a halo file for the lightcone, given the halo files of the snapshots and their cut out shells
    :param param: baryonification params
    :param halo_files: list of halo files
    :param shells_z: An array containing the redshifts of the shells with shape [n_halo_files, 2], where the first
                     column is the lower bound and the second column represents the upper bound
    :param output_path: path where to save the file
    :param max_repli: maximum number of replicates
    :param out_type: currently only 'AHF-NUMPY'
    :return: name of the written file
    """

    # check for consitency
    assert len(halo_files) == len(shells_z), "Halo file and boundaries need to match in first dimension"

    # check that int8 is enough for the shell index
    # if len(halo_files) > 255:
    #     raise ValueError("The shell index must be representable as int8, therefore a maximum number of 255 shells is "
    #                      "allowed!")

    # save the relevant lines and shell_data
    relevant_lines = []
    shell_data = []

    # get n proc
    # if n_proc is None:
    #     n_proc = cpu_count()

    # set up the pool
    # pool = Pool(n_proc)

    # input args
    # input_args = []
    # i_file = 0
    # for halo_file, z in zip(halo_files, shells_z):
    #     input_args.append((param, halo_file, z, i_file, max_repli))
    #     i_file += 1

    # # cycle
    # for halos_local, shell_local in pool.imap(get_single_halo_shell_wrapper, input_args):

    #     # append halos and shell data
    #     shell_data.append(shell_local)
    #     relevant_lines.extend(halos_local)

    for i_file, (halo_file, z) in enumerate(zip(halo_files, shells_z)):
            
        LOGGER.warning(f'=========> Processed halo file {i_file:>4d}/{len(halo_files)} {halo_file}')

        halos_local, shell_local = get_single_halo_shell(param, halo_file, z, i_file, max_repli=max_repli, test=test)
        shell_data.append(shell_local)
        relevant_lines.extend(halos_local)
        

    # wrap up the pool
    # pool.close()

    # concat and save
    # output_file = paths.create_path(identity="Halofile",
    #                                 out_folder=output_path,
    #                                 defined_parameters={"MinParts": param.sim.Nmin_per_halo},
    #                                 suffix=".npz")

    # concat shell data
    shell_data = np.concatenate(shell_data)

    # concat the halos
    halo_data = np.concatenate(relevant_lines, axis=0)
    if param.code.halo_buffer > 0:
        sorted_args = np.argsort(halo_data["shell_id"])
        halo_data = halo_data[sorted_args]

    # we trimm impossible shell ids
    mask = np.logical_and(halo_data["shell_id"] != -1, halo_data["shell_id"] != len(shell_data))
    halo_data = halo_data[mask]

    # get all the summaries
    N_tot = len(halo_data)
    id_set = set(halo_data["ID"])
    N_lim = np.arange(50, 1001, 50)
    N_num = np.sum(halo_data["Nvir"][:,None] >= N_lim, axis=0)

    LOGGER.error("Finished collection of Halos")
    LOGGER.error("Number of halos:        {}".format(N_tot))
    LOGGER.error("Number of unique halos: {}".format(len(id_set)))
    LOGGER.error("Number selected halos: \n")
    for lim, num in zip(N_lim, N_num):
        LOGGER.info("N_min: %03d N_halos: %i" %(lim, num))

    # we project the halo coordinates onto the shell they fall in
    norm_kPc = np.sqrt(halo_data['x'] ** 2 + halo_data['y'] ** 2 + halo_data['z'] ** 2)
    shell_cov = shell_data["shell_com"][halo_data["shell_id"]]
    halo_data['x'] *= 1000 * shell_cov / norm_kPc
    halo_data['y'] *= 1000 * shell_cov / norm_kPc
    halo_data['z'] *= 1000 * shell_cov / norm_kPc

    # if out_type == 'AHF-NUMPY':
    #     np.savez_compressed(output_file, halos=halo_data, shells=shell_data)

    return halo_data, shell_data

def get_cosmo_data(param):
    """
    This function calculates the cosmo data necessary for the projection (redshift agnostic)
    :param param: A Baryonifiction parameter instance
    :return: a cosmo data array containing [bin_r, bin_m, bin_var, bin_corr]
    """

    # read out param
    Om = param.cosmo.Om
    ns = param.cosmo.ns
    s8 = param.cosmo.s8

    kmin = param.code.kmin
    kmax = param.code.kmax
    rmin = param.code.rmin
    rmax = param.code.rmax

    # we execute the same code as in the beginning (cosmo.py) but drop all redshift dependence
    # load transfer function
    TFfile = param.files.transfct
    try:
        if param.files.transfct_format == "CAMB":
            names = "k, Ttot"
            TF = np.genfromtxt(TFfile, usecols=(0, 6), comments='#', dtype=None, names=names)
        if param.files.transfct_format == "HDF5":
            with h5py.File(TFfile, "r") as f:
                # get rho
                rho_cdm_b = f['background']['rho_cdm+b'][-1]
                rho_ncdm = f['background']['rho_ncdm[0]'][-1]
                rho_crit = f['background']['rho_crit'][-1]

                # get the transfer functions (assume Mpc for now)
                k_out_file = f['perturbations']['k'][:]
                d_cdm_b = f['perturbations']['delta_cdm+b'][-1]
                d_ncdm = f['perturbations']['delta_ncdm[0]'][-1]

                # get h
                h = float(f["class_params"].attrs['H0']) / 100

            # get the weighted transfer function (norm is different from sigma_8 calculation)
            delta_tot = (rho_cdm_b * d_cdm_b + rho_ncdm * d_ncdm) / rho_crit

            # norm everything correctly
            file_k_camb_norm = k_out_file / h
            file_t_camb_norm = -delta_tot / (file_k_camb_norm * h) ** 2

            TF = {'k': file_k_camb_norm,
                  'Ttot': file_t_camb_norm}


    except IOError:
        LOGGER.info('IOERROR: Cannot read transfct. Try: par.files.transfct = "/path/to/file"')
        exit()

    # spline
    TF_tck = splrep(TF['k'], TF['Ttot'])

    # Normalize power spectrum
    R = 8.0
    itd = lambda logk: np.exp((3.0 + ns) * logk) * splev(np.exp(logk), TF_tck) ** 2.0 * wf(np.exp(logk) * R) ** 2.0
    itl = quad(itd, np.log(kmin), np.log(kmax), epsrel=5e-3, limit=100)
    A_NORM = 2.0 * np.pi ** 2.0 * s8 ** 2.0 / itl[0]
    LOGGER.info('Normalizing power-spectrum done!')

    bin_N = 100
    bin_r = np.logspace(np.log(rmin), np.log(rmax), bin_N, base=np.e)
    bin_m = 4.0 * np.pi * Om  * bin_r ** 3.0 / 3.0 # * rhoc_of_z(param.cosmo.z, param.cosmo.Om)

    bin_var = []
    bin_corr = []

    for i in range(bin_N):
        bin_var += [variance(bin_r[i], TF_tck, A_NORM, param)]
        bin_corr += [correlation(bin_r[i], TF_tck, A_NORM, param)]
    bin_var = np.array(bin_var)
    bin_corr = np.array(bin_corr)

    # save as dataset
    cosmo_data = np.transpose([bin_r, bin_m, bin_var, bin_corr]).astype(np.float64)
    return cosmo_data

def get_pix_and_displ_init(*collected_info):
    """
    A init routine for a parallelized call of pix_and_displ_init
    :param collected_info: a collected info named tuple made globally available
    """

    global shared_info
    # the comma is necessary for the unpacking, callected_info is a tubple (x,)
    shared_info, = collected_info

# this function returns the pixels and displacements for a given halo slice, used for parallelization
# @profile
def get_pix_and_displ(halo_slice, shared_info, DFDM=None):

    # global shared_info

    # create a shell collection
    shell_info = dict(n_halos=0,
                      n_halos_unique=0,
                      n_parts=0,
                      n_parts_unique_FDM=0,
                      n_parts_unique_BAR=0,
                      shell_path=shared_info.shell_path)

    # get all halos with the same id
    same_id_halos = shared_info.current_halos[halo_slice]
    first_halo = same_id_halos[0]

    # rball
    if shared_info.params.code.maxRvir * first_halo['rvir'] < shared_info.params.code.rmax:
        rball = shared_info.params.code.maxRvir * first_halo['rvir']
    else:
        rball = shared_info.params.code.rmax

    # get the intersection angle
    pos = np.array([first_halo['x'], first_halo['y'], first_halo['z']])
    r_halo = np.linalg.norm(pos)
    alpha = get_intersection_angle(shared_info.shell["shell_com"], rball, r_halo)

    # other properties that won't change for identical halos
    Mvir = first_halo["Mvir"]
    cvir = first_halo["cvir"]
    tr_Mvir = first_halo["tfNFW_Mvir"]
    tr_cvir = first_halo["tfNFW_cvir"]
    tr_tau = first_halo["tfNFW_tau"]

    # bias and corr
    cosmo_bias = shared_info.bias_func(Mvir)
    cosmo_corr = shared_info.corr_func(shared_info.rbin)
    Mc_z = shared_info.params.baryon.Mc*(1.0 + shared_info.current_z)**shared_info.params.baryon.nu

    if DFDM is None:
    
        rbin, DDMB ,DFDM = get_displ_vec(Mvir, cvir, tr_Mvir, tr_cvir, tr_tau, shared_info)

    # get the displacement splines
    DDMB_tck = splrep(shared_info.rbin, DDMB, s=0, k=3)
    DFDM_tck = splrep(shared_info.rbin, DDMB, s=0, k=3)

    # cycle though all positions
    for n_subhalo, sub_halo in enumerate(same_id_halos):

        pos = np.array([sub_halo['x'], sub_halo['y'], sub_halo['z']])
        r_halo = np.linalg.norm(pos)
        pos_norm = pos / r_halo

        # get the particles
        pixels = hp.query_disc(nside=shared_info.shell_nside, vec=pos_norm, radius=alpha)

        # if we have less than four particles we set the angle to None and take the neighbors
        if not shared_info.params.code.force_parts and len(pixels) < 4:
            
            # we do not have enough particles so we skip the halo
            continue
        
        elif len(pixels) < 4:
            
            # pixels and weights have shape (4, N)
            theta, phi = hp.vec2ang(pos_norm)
            pixels, _ = hp.get_interp_weights(nside=shared_info.shell_nside, theta=theta, phi=phi)
            # reshape and get pars
            pixels = pixels.ravel()

        LOGGER.debug(f" Shell {shared_info.shell['shell_id']} is dealing with halo {sub_halo['ID']} "
                    f"({n_subhalo+1}/{len(same_id_halos)}), found {len(pixels)} particles...")

        # if we get to here, we update the info (we count empty pixels as well)
        if n_subhalo == 0:
            shell_info["n_halos_unique"] += 1
        shell_info["n_halos"] += 1
        shell_info["n_parts"] += len(pixels)

        # now that we have the pixels we calculate the displacement
        parts_pos = np.stack(hp.pix2vec(nside=shared_info.shell_nside, ipix=pixels), axis=1)
        
        # blow up to radius
        parts_pos *= shared_info.current_com

        # relative distance
        rpDM = np.linalg.norm(parts_pos - pos, axis=1, keepdims=True)

        # displace - this is the main magic here
        DrpFDM = splev(rpDM, DFDM_tck, der=0, ext=1)
        DrpBAR = splev(rpDM, DBAR_tck, der=0, ext=1)
        displacement_FDM = (parts_pos - pos) * DrpFDM / rpDM
        displacement_BAR = (parts_pos - pos) * DrpBAR / rpDM
        np.nan_to_num(displacement_FDM, copy=False, nan=0.0, posinf=None, neginf=None)
        np.nan_to_num(displacement_BAR, copy=False, nan=0.0, posinf=None, neginf=None)

    return shell_info, pixels, displacement_FDM, displacement_BAR




# @profile
def get_displ_vec(Mvir, cvir, tfNFW_Mvir, tfNFW_cvir, tfNFW_tau, shared_info):


    # bias and corr
    cosmo_bias = shared_info.bias_func(Mvir)
    cosmo_corr = shared_info.corr_func(shared_info.rbin)
    Mc_z = shared_info.params.baryon.Mc*(1.0 + shared_info.current_z)**shared_info.params.baryon.nu

    # this is the time-consuming part
    frac, dens, mass = profiles(rbin=shared_info.rbin,
                                Mvir=Mvir,
                                cvir=cvir,
                                Mc=Mc_z,
                                mu=shared_info.params.baryon.mu,
                                thej=shared_info.params.baryon.thej,
                                cosmo_corr=cosmo_corr,
                                cosmo_bias=cosmo_bias,
                                param=shared_info.params,
                                rhoc=shared_info.current_rho,
                                tr_cvir=tfNFW_cvir,
                                tr_tau=tfNFW_tau,
                                tr_Mvir=tfNFW_Mvir)
    
    # get the displacement splines
    DFDM = displ(shared_info.rbin, mass['DMi_pro'], mass['DMf_pro'])
    DBAR = displ(shared_info.rbin, mass['BARi_pro'], mass['BARf_pro'])

    return shared_info.rbin, DFDM, DBAR


def sample_sequence_in_hull(x, n_samples, sequence='sobol'):
    
    from scipy.spatial import Delaunay, ConvexHull
    from sklearn.preprocessing import MinMaxScaler
    import chaospy
    from scipy.stats import qmc

    def sample_sequence(seq, n_dim, n_samples):
        
        dists = [chaospy.Uniform(0, 1) for _ in range(n_dim)]
        uniform_cube = chaospy.J(*dists)
        s = uniform_cube.sample(size=n_samples, rule=seq)
        return s.T

    scaler = MinMaxScaler().fit(x)
    xt = scaler.transform(x)
    hull = ConvexHull(xt)
    dela = Delaunay(xt[hull.vertices])
    list_grid_xt = [xt[hull.vertices]]

    n_found = 0
    n_sampled = n_samples
    n_extend = n_samples//20
    while n_found < n_samples:
        grid_xt = sample_sequence(seq=sequence, n_dim=2, n_samples=n_extend+n_sampled)
        select = in_hull(grid_xt, dela)
        n_found = np.count_nonzero(select)
        n_sampled += n_extend

    grid_xt = grid_xt[select]
    grid_xt = np.concatenate([grid_xt, xt[hull.vertices]], axis=0)
    grid_x = scaler.inverse_transform(grid_xt)

    return grid_x


def scale_range(x, factor=1.01):

    mean_x = np.mean(x, axis=0)
    return (x - mean_x)*factor + mean_x


def interp_halo_displacement_profiles(halos, current_info, n_grid=int(2**14)):

    assert np.all(halos['tfNFW_cvir']==-1), 'halo interpolation speedup works only for halo model with tfNFW_cvir=-1'

    LOGGER.info(f'building halo profiles interpolator with n_grid={n_grid} training points')

    # create sobol sampling with slightly extended convex hull of the data, in log space
    # this is needed to avoid numerical error through the log10 transform
    x = np.vstack([halos['Mvir'], halos['cvir']]).T
    grid_x = 10**sample_sequence_in_hull(x=np.log10(scale_range(x, factor=1.001)),
                                         n_samples=n_grid, 
                                         sequence='hammersley')

    # evaluate grid using the full method - create training set
    list_ddmb = []
    list_r = []
    for i, s in LOGGER.progressbar(list(enumerate(grid_x)), at_level='debug', desc='building interpolator'):    
        r, ddmb = get_displ_vec(Mvir=s[0], 
                                cvir=s[1], 
                                tfNFW_Mvir=halos['tfNFW_Mvir'][0],
                                tfNFW_cvir=halos['tfNFW_cvir'][0], 
                                tfNFW_tau=halos['tfNFW_tau'][0], 
                                shared_info=current_info)
        list_r.append(r)
        list_ddmb.append(ddmb)
    ddmb = np.array(list_ddmb)
    r = np.array(list_r)

    halos_x = np.vstack([halos['Mvir'], halos['cvir']]).T
    LOGGER.info(f'interpolating {len(halos_x)} points with CloughTocher2DInterpolator')
    from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
    time_start = time.time()
    interp = CloughTocher2DInterpolator(points=grid_x, values=ddmb, fill_value=0, rescale=True)
    LOGGER.info(f'done! time={time.time()-time_start:5.2f}s')

     
    ddmb_interp = interp(halos_x)

    # import pudb; pudb.set_trace();
    # pass
    # np.save('/cluster/work/refregier/tomaszk/projects/220708_cosmogrid_paper/005_cosmogridv11/bad_halos_x.npy', halos_x)
    # np.save('/cluster/work/refregier/tomaszk/projects/220708_cosmogrid_paper/005_cosmogridv11/bad_log_x.npy', np.log10(x))
    # np.save('/cluster/work/refregier/tomaszk/projects/220708_cosmogrid_paper/005_cosmogridv11/bad_grid_x.npy', grid_x)
    # np.save('/cluster/work/refregier/tomaszk/projects/220708_cosmogrid_paper/005_cosmogridv11/bad_ddmb_interp.npy', ddmb_interp)
    # np.save('/cluster/work/refregier/tomaszk/projects/220708_cosmogrid_paper/005_cosmogridv11/bad_ddmb.npy', ddmb)
    # np.save('/cluster/work/refregier/tomaszk/projects/220708_cosmogrid_paper/005_cosmogridv11/bad_halos.npy', halos)


    return r, ddmb_interp

def check_displacement_interpolation_accuracy(pixels_nointerp, displacement_nointerp, pixels_interp, displacement_interp, err_max=1e-4):

    def sort_pix(pix, disp):
        sorting = np.argsort(pix)
        pix = pix[sorting]
        disp = disp[sorting]
        return pix, disp

    pixels_nointerp, displacement_nointerp = sort_pix(pixels_nointerp, displacement_nointerp)
    pixels_interp, displacement_interp = sort_pix(pixels_interp, displacement_interp)

    select = np.in1d(pixels_interp, pixels_nointerp)
    pixels_interp_common = pixels_interp[select]
    displacement_interp_common = displacement_interp[select]
    
    select = np.in1d(pixels_nointerp, pixels_interp)
    pixels_nointerp_common = pixels_nointerp[select]
    displacement_nointerp_common = displacement_nointerp[select]

    assert len(pixels_nointerp_common) == len(pixels_interp_common), 'wrong pixel counts between interpolated and original displacements'

    err = np.sqrt(np.mean((displacement_nointerp_common.ravel()-displacement_interp_common.ravel())**2))

    if err>err_max:

        LOGGER.warning(f'large error for interpolating halo profile displacements {err} > {err_max}, try increasring interpolation grid size interp_ngrid')
        # raise Exception(f'large error for interpolating halo profile displacements {err} > {err_max}, try increasring interpolation grid size interp_ngrid')

    return err


# def displace_shell_wrapper(args):
#     """
#     A quick and dirty wrapper for the get_shell_collection function, such that it can be called with imap
#     :param args: arguments for get_shell_collection
#     :return: get_shell_collection(*args)
#     """
#     return displace_shell(*args)
# @profile
def displace_shell(params, shell_particles, particle_shell_info, halos, cosmo_data, delta_shell=True, nside_out=None, n_proc=1, interp_ngrid=None):
    """
    Given some params, a shell and the relevant halos, this function create a ShellCollection of halos...
    :param params: baryonification params
    :param shell_particles: A shell of particles to displace
    :param particle_shell_info: the shell info of the particle shell
    :param halos: the halos in the common format
    :param cosmo_data: redshift agnostic cosmo data to calculate the displacements
    :param delta_shell: Instead of outputting the shells output the difference (baryon shell - normal shell)
    :param nside_out: nside of the output shell, defaults to input nside
    :param n_proc: number of procs used to parallelize over halos
    :return: a ShellCollection
    """

    LOGGER.debug("Starting with shell {}...".format(particle_shell_info["shell_name"]))

    # we start by doing the cosmology related stuff
    cosmo = build_cosmo(param=params)

    # deal with cosmo data
    vc_r = cosmo_data[:, 0]
    vc_m = cosmo_data[:, 1]
    vc_var = cosmo_data[:, 2]
    vc_corr = cosmo_data[:, 3]

    # get the splines
    var_tck = splrep(vc_m, vc_var, s=0)
    corr_tck = splrep(vc_r, vc_corr, s=0)

    # get the z dependent actual terms
    dc = params.cosmo.dc

    # the bins are fixed for all halos for the projection integration (times to so that we integrate over at least rmax)
    n_rbins = 100 
    rbin = np.logspace(np.log10(params.code.rmin), np.log10(params.code.rmax), n_rbins, base=10)

    # evolution params of this shell
    current_com = particle_shell_info["shell_com"]
    current_z = 0.5*(particle_shell_info["lower_z"] + particle_shell_info["upper_z"])
    current_a = 1.0/(1.0 + current_z)
    D0 = growth_factor(a=1.0, cosmo=cosmo)
    current_growth = growth_factor(a=current_a, cosmo=cosmo)/D0
    current_rho = cosmo.critical_density(current_z).to(cosmo.h**2*Msun/Mpc**3).value

    # we fix the bias function for this shell
    bias_func = lambda vc_m: bias(splev(vc_m / current_rho, var_tck), dc / current_z)
    corr_func = lambda vc_r: splev(vc_r, corr_tck) * current_growth ** 2

    # first we format the halos
    args = np.argsort(halos["ID"])
    # sorted such that the same id occurs subsequently
    current_halos = halos[args]

    # get the splits (all halos with identical ID)
    halo_splits = get_split_indices(current_halos['ID'])
    # add elements to the splits such that we can always get two
    halo_splits.insert(0, 0)
    halo_splits.append(len(current_halos))


    # get nside
    shell_nside = hp.npix2nside(len(shell_particles))

    # the map containing the particle displacements
    displ_map_FDM = np.zeros((hp.nside2npix(shell_nside), 3), dtype=np.float32)
    displ_map_BAR = np.zeros((hp.nside2npix(shell_nside), 3), dtype=np.float32)

    # create a shell collection
    shell_path = particle_shell_info['shell_name']
    shell_info = dict(n_halos=0,
                      n_halos_unique=0,
                      n_parts=0,
                      n_parts_unique_FDM=0,
                      n_parts_unique_BAR=0,
                      shell_path=shell_path)

    # we pack everything into a names tuple to make the init slightly less painful
    collected_info = namedtuple("collected_info", "current_halos, params, shell, bias_func, corr_func, rbin, current_rho, shell_nside, shell_path, current_com, current_z")
    current_info = collected_info(current_halos, params, particle_shell_info, bias_func, corr_func, rbin, current_rho, shell_nside, shell_path, current_com, current_z)

    if interp_ngrid is not None:

        r, ddmb_interp = interp_halo_displacement_profiles(current_halos[halo_splits[:-1]], current_info, n_grid=int(interp_ngrid))
    
    for i in LOGGER.progressbar(range(len(halo_splits) - 1), at_level='info', desc=f'processing halos in shell {particle_shell_info["shell_id"]}'):
        
        halos_slice = slice(halo_splits[i], halo_splits[i + 1])

        if interp_ngrid is None:
            
            shell_info, pixels, displacement_FDM, displacement_BAR = get_pix_and_displ(halos_slice, current_info, DDMB=None)

        else:
            
            shell_info, pixels, displacement_FDM, displacement_BAR = get_pix_and_displ(halos_slice, current_info, DDMB=ddmb_interp[i])
            # sub_info, pixels, displacement = get_pix_and_displ_interp(ddmb_interp[i], halos_slice, current_info)

            # accuracy check 
            if i % 10000 == 0:
                sub_info_nointerp, pixels_nointerp, displacement_FDM_nointerp, _ = get_pix_and_displ(halos_slice, current_info, DFDM=None)
                check_displacement_interpolation_accuracy(pixels_nointerp, displacement_FDM_nointerp, pixels, displacement_FDM)

        displ_map_FDM[pixels] += displacement_FDM
        displ_map_BAR[pixels] += displacement_BAR
        shell_info["n_halos"] += sub_info["n_halos"]
        shell_info["n_halos_unique"] += sub_info["n_halos_unique"]
        shell_info["n_parts"] += sub_info["n_parts"]
        shell_info["n_parts_unique_FDM"] += sub_info["n_parts_unique_FDM"]
        shell_info["n_parts_unique_BAR"] += sub_info["n_parts_unique_BAR"]

    # with ProcessPoolExecutor(max_workers=n_proc, initializer=get_pix_and_displ_init, initargs=(current_info, )) as pool:
    #     for sub_info, pixels, displacement in pool.map(get_pix_and_displ, slice_gen(), chunksize=chunksize):
    #         # add displacement to map
    #         displ_map[pixels] += displacement
    #         shell_info["n_halos"] += sub_info["n_halos"]
    #         shell_info["n_halos_unique"] += sub_info["n_halos_unique"]
    #         shell_info["n_parts"] += sub_info["n_parts"]
    #         shell_info["n_parts_unique"] += sub_info["n_parts_unique"]

    # update the unique parts (we count empty pixels as well)
    displ_norm_FDM = np.linalg.norm(displ_map_FDM, axis=1)
    displ_norm_BAR = np.linalg.norm(displ_map_BAR, axis=1)
    shell_info["n_parts_unique_FDM"] = np.sum(displ_norm_FDM > 1e-8)
    shell_info["n_parts_unique_BAR"] = np.sum(displ_norm_BAR > 1e-8)

    # get the relevant displacements
    displ_mask_FDM = np.logical_and(shell_particles > 0.5, displ_norm_FDM)
    displ_mask_BAR = np.logical_and(shell_particles > 0.5, displ_norm_BAR)
    relevant_displ_FDM = displ_map_FDM[displ_mask]
    relevant_displ_BAR = displ_map_BAR[displ_mask]
    relevant_pix_FDM = np.arange(hp.nside2npix(shell_nside))[displ_mask_FDM]
    relevant_pix_BAR = np.arange(hp.nside2npix(shell_nside))[displ_mask_BAR]

    # create positions and relative distance to halo
    parts_pos_FDM = np.stack(hp.pix2vec(nside=shell_nside, ipix=relevant_pix_FDM), axis=1)
    parts_pos_BAR = np.stack(hp.pix2vec(nside=shell_nside, ipix=relevant_pix_BAR), axis=1)
    # blow up to radius
    parts_pos_FDM *= current_com
    parts_pos_BAR *= current_com

    # displace
    parts_pos_FDM += relevant_displ_FDM
    parts_pos_BAR += relevant_displ_BAR

    # normalize
    parts_r_FDM = np.linalg.norm(parts_pos_FDM, ord=2, axis=1, keepdims=True)
    parts_r_BAR = np.linalg.norm(parts_pos_BAR, ord=2, axis=1, keepdims=True)
    parts_pos_FDM /= parts_r_FDM
    parts_pos_BAR /= parts_r_BAR
    np.nan_to_num(parts_pos_FDM, copy=False, nan=0.0, posinf=None, neginf=None)
    np.nan_to_num(parts_pos_BAR, copy=False, nan=0.0, posinf=None, neginf=None)

    # project with interpolation
    theta_FDM, phi_FDM = hp.vec2ang(parts_pos_FDM)
    # pixels and weights have shape (4, N)
    pixels_FDM, weights_FDM = hp.get_interp_weights(nside=shell_nside, theta=theta_FDM, phi=phi_FDM)
    # transpose to (N, 4)
    pixels_FDM = pixels_FDM.T
    weights_FDM = weights_FDM.T.astype(np.float32)

    theta_BAR, phi_BAR = hp.vec2ang(parts_pos_BAR)
    pixels_BAR, weights_BAR = hp.get_interp_weights(nside=shell_nside, theta=theta_BAR, phi=phi_BAR)
    pixels_BAR = pixels_BAR.T
    weights_BAR = weights_BAR.T.astype(np.float32)

    # asign
    displ_shell_FDM = np.zeros(hp.nside2npix(shell_nside), dtype=np.float32)
    displ_shell_BAR = np.zeros(hp.nside2npix(shell_nside), dtype=np.float32)
    numba_asign(displ_shell_FDM, pixels_FDM, weights_FDM, shell_particles[relevant_pix_FDM])
    numba_asign(displ_shell_BAR, pixels_BAR, weights_BAR, shell_particles[relevant_pix_BAR])

    if delta_shell:
        # get the difference for dark matter
        displ_shell_FDM[relevant_pix_FDM] -= shell_particles[relevant_pix_FDM]
    else:
        # add the irrelevant pixel
        inv_mask_FDM = np.logical_not(displ_mask_FDM)
        inv_mask_BAR = np.logical_not(displ_mask_BAR)
        displ_shell_FDM[inv_mask_FDM] += shell_particles[inv_mask_FDM]
        displ_shell_BAR[inv_mask_BAR] += shell_particles[inv_mask_BAR]

    if nside_out is not None:
        if nside_out != shell_nside:
            displ_shell_FDM = hp.ud_grade(displ_shell_FDM, nside_out=nside_out, power=-2)
            displ_shell_BAR = hp.ud_grade(displ_shell_BAR, nside_out=nside_out, power=-2)

    return particle_shell_info, shell_info, displ_shell_FDM, displ_shell_BAR

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    # https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0        

# @profile
def get_baryon_shells(param, shells, shells_info, halos, delta_shell=True, nside_out=None, test=False, interp_halo_displacements=False, interp_ngrid=int(2**13)):
    """
    Creates a set of shells that includes baryon effects
    :param param: baryonification params that should be used
    :param shell_file: shell file containing all shells and the corresponding info
    :param halo_file: the halo file, automatically determines the name of the Collection
    :param out_path: where to save the output shells, can be directory or file name
    :param delta_shell: Instead of outputting the shells output the difference (baryon shell - normal shell)
    :param nside_out: nside of the output shell, defaults to input nside
    :return: The absolute path of the saved file
    """


    # load the shells
    # LOGGER.info("Loading shells...")
    # shells = np.load(shell_file)
    # particle_shells = shells['shells']
    # particle_shells_info = shells['shell_info']
    # LOGGER.info("Done...")

    # load the halos
    # with h5py.File(halo_file) as f:
    #     shells = np.array(f["shell_data"])
    #     halos = np.array(f["halo_data"])
    #     halos_MinParts = f["halo_data"].attrs['MinParts']
    
    # if param.sim.Nmin_per_halo != halos_MinParts:
    #     LOGGER.info("Minimum number of particles from the baryonification params ({}) and the halo file ({}) "
    #           "do not match, overwriting params with halo file value...".format(param.sim.Nmin_per_halo,
    #                                                                             param_dict["MinParts"][0]))
    #     param.sim.Nmin_per_halo = halos_MinParts




    # # we start by getting the parameters from the
    # param_dict, _ = paths.get_parameters_from_path(halo_file)

    # # we check for consitency between param object and the loaded parameter
    # if param.sim.Nmin_per_halo != param_dict["MinParts"][0]:
    #     LOGGER.info("Minimum number of particles from the baryonification params ({}) and the halo file ({}) "
    #           "do not match, overwriting params with halo file value...".format(param.sim.Nmin_per_halo,
    #                                                                             param_dict["MinParts"][0]))
    #     param.sim.Nmin_per_halo = param_dict["MinParts"][0]

    # we get the cosmo data
    cosmo_data = get_cosmo_data(param=param)

    # # get the number of procs if not declared
    # if n_proc is None:
    #     n_proc = cpu_count() - 1

    # load the halo file and get the halos and the shells

    # LOGGER.info("Loading in halos...")
    # file_arrays = np.load(halo_file)
    # shells = file_arrays["shells"]
    # halos = file_arrays["halos"]

    # adopt units
    halos = mpc_to_gpc(halos, fields=['x', 'y', 'z', 'rvir'])
    # halos['x'] = halos['x'] / 1000.0
    # halos['y'] = halos['y'] / 1000.0
    # halos['z'] = halos['z'] / 1000.0
    # halos['rvir'] = halos['rvir'] / 1000.0

    # we split the halos by shell
    split_indices = get_split_indices(halos["shell_id"])
    shelled_halos = np.split(halos, indices_or_sections=split_indices)

    shells_stats, shells_back, baryon_shells_FDM, baryon_shells_BAR = [], [], [], []

    for i, h in LOGGER.progressbar(list(enumerate(shelled_halos)), desc='displacing shells', at_level='warning'):

        LOGGER.info(f'displacing shell {int(h[0]["shell_id"]):d} with n_halos={len(h)}')

        if test:

            # n_halos=np.array([len(h) for h in shelled_halos])
            # select_shell = np.argmax(n_halos)
            # h = shelled_halos[select_shell]
            # LOGGER.warning(f'testing -> using shell with the largest number of halos {select_shell} n_halos={len(h)}')
            shell_id_ = 17
            h = halos[halos['shell_id']==shell_id_]
            LOGGER.warning(f'testing -> using shell_id={shell_id_}')


        s = shells_info[shells_info["shell_id"] == h[0]["shell_id"]][0]
        shell_index = match_shell(shell_info=shells_info, halo_shell=s)

        shell_info, shell_stats, baryon_shell_FDM, baryon_shell_BAR = displace_shell(params=param, 
                                                               shell_particles=shells[shell_index], 
                                                               particle_shell_info=shells_info[shell_index], 
                                                               halos=h, 
                                                               cosmo_data=cosmo_data, 
                                                               delta_shell=delta_shell, 
                                                               nside_out=nside_out,
                                                               interp_ngrid=interp_ngrid if (interp_halo_displacements and (len(h)>2*interp_ngrid)) else None)

        shells_stats.append(shell_stats)
        shells_back.append(shell_info)
        baryon_shells_FDM.append(baryon_shell_FDM)
        baryon_shells_BAR.append(baryon_shell_BAR)
        LOGGER.debug("Added shell {} to the halocone...".format(shell_stats["shell_path"]))

        if test:
            LOGGER.warning('----------> testing! skipping other baryon shells')
            break
                
    # this adds shells which did not have any baryonification applied
    # if delta_shell is False we want to add the shells without halos as well
    if not delta_shell:
        
        for displ_shell, shell_info in LOGGER.progressbar(list(zip(shells, shells_info)), desc='adding unbaryonified shells', at_level='info'):
        
            if shell_info not in shells_back:
        
                # get the shell path
                shell_path = shell_info["shell_name"]

                # load and downsample
                if nside_out is not None:
                    if nside_out != hp.npix2nside(len(displ_shell)):
                        displ_shell = hp.ud_grade(displ_shell.astype(np.float32), nside_out=nside_out, power=-2)

                # create a shell collection
                shell_stats = dict(n_halos=0,
                                  n_halos_unique=0,
                                  n_parts=0,
                                  n_parts_unique_FDM=0,
                                  n_parts_unique_BAR=0,
                                  shell_path=shell_path)

                # append
                shells_stats.append(shell_stats)
                shells_back.append(shell_info)
                baryon_shells_FDM.append(displ_shell_FDM)
                baryon_shells_BAR.append(displ_shell_BAR)

    shells_stats_arr = shell_stats_list_to_arr(shells_stats)
    shells_back = np.array(shells_back)
    sorting = np.argsort(shells_back['shell_id'])
    shells_back = shells_back[sorting]
    shells_stats_arr = shells_stats_arr[sorting]
    baryon_shells_FDM = np.array([baryon_shells_FDM[i] for i in sorting], dtype=np.float32)
    baryon_shells_BAR = np.array([baryon_shells_BAR[i] for i in sorting], dtype=np.float32)

    # sort according to the shell id
    # baryon_shells = baryon_shells[sorting]

    return baryon_shells_FDM, baryon_shells_BAR, shells_stats_arr, shells_back

def shell_stats_list_to_arr(shell_stats):

    from cosmogridv1 import utils_arrays
    shell_stats_arr = np.empty(len(shell_stats), dtype=utils_arrays.get_dtype(['n_halos:i4', 'n_halos_unique:i4', 'n_parts:i4', 'n_parts_unique_FDM:i4', 'n_parts_unique_BAR:i4', 'shell_path:a256']))
    for i, s_ in enumerate(shell_stats):
        for k in s_.keys():
            shell_stats_arr[i][k] = s_[k]

    return shell_stats_arr


def displ(rbin,MDMO,MDMB):

    """
    Calculates the displacement of all particles as a function
    of the radial distance from the halo centre
    This function is the only one that was needed from <displ.py> and was copied in here
    """

    try:
        MDMBinv_tck = splrep(MDMB, rbin, s=0, k=3)
        rDMB = splev(MDMO, MDMBinv_tck, der=0)

    # sometimes splrep crashes, use interp1d cubic
    except ValueError as err:
        
        from scipy.interpolate import interp1d
        rDMB = interp1d(MDMB, rbin,  kind='cubic', fill_value='extrapolate', bounds_error=False)(MDMO)
    
    DDMB = rDMB - rbin

    return DDMB

def mpc_to_gpc(rec, fields):

    for f in fields:
        rec[f] = rec[f]/1000

    return rec















# code graveyard


    # generator for the splits
    # def slice_gen():
    #     for i in range(len(halo_splits) - 1):
    #         yield slice(halo_splits[i], halo_splits[i + 1])
    # if we only have a single processor, there is only one chunk
    # if n_proc == 1:
    #     chunksize = len(halo_splits)
    # # get the chunk size (at least 100 at most 1000)
    # elif len(halo_splits) / n_proc > 1000:
    #     chunksize = 1000
    # else:
    #     chunksize = 100

    # def f_wrap(i, halo_splits, current_info, fast_mode):

    #     halos_slice = slice(halo_splits[i], halo_splits[i + 1])
    #     sub_info, pixels, displacement = get_pix_and_displ(halos_slice, current_info, fast_mode=fast_mode)
    #     return sub_info, pixels, displacement

    # import jax
    # from functools import partial
    # f_vec = jax.vmap(partial(get_displ_spline, shared_info=current_info, fast_mode=fast_mode))
    # out_jax = f_vec(current_halos['Mvir'][:10], current_halos['rvir'][:10], current_halos['cvir'][:10], current_halos['tfNFW_Mvir'][:10], current_halos['tfNFW_cvir'][:10], current_halos['tfNFW_tau'][:10])

    # n_grid = int(1e4)
    # from scipy.stats import qmc 
    # r, ddmb = get_displ_vec(Mvir=current_halos['Mvir'][0], 
    #                         rvir=current_halos['rvir'][0],
    #                         cvir=current_halos['cvir'][0], 
    #                         tfNFW_Mvir=current_halos['tfNFW_Mvir'][0],
    #                         tfNFW_cvir=current_halos['tfNFW_cvir'][0], 
    #                         tfNFW_tau=current_halos['tfNFW_tau'][0], 
    #                         shared_info=current_info)

    # sampler = qmc.Sobol(d=3,  scramble=False)
    # sample = sampler.random(n_grid)
    # sample = qmc.scale(sample, 
    #                   l_bounds=[np.min(np.log10(current_halos['Mvir'])), np.min(current_halos['rvir']), np.min(current_halos['cvir'])], 
    #                   u_bounds=[np.max(np.log10(current_halos['Mvir'])), np.max(current_halos['rvir']), np.max(current_halos['cvir'])])

    # list_ddmb = []
    # list_r = []
    # for i, s in LOGGER.progressbar(list(enumerate(sample))):    
    #     r, ddmb = get_displ_vec(Mvir=10**s[0], 
    #                             rvir=s[1],
    #                             cvir=s[2], 
    #                             tfNFW_Mvir=current_halos['tfNFW_Mvir'][0],
    #                             tfNFW_cvir=current_halos['tfNFW_cvir'][0], 
    #                             tfNFW_tau=current_halos['tfNFW_tau'][0], 
    #                             shared_info=current_info, 
    #                             fast_mode=False)
    #     list_r.append(r)
    #     list_ddmb.append(ddmb)
    # ddmb = np.array(list_ddmb)


        # this is the time-consuming part
        # frac, dens, mass = profiles(rbin=shared_info.rbin,
        #                             Mvir=Mvir,
        #                             cvir=cvir,
        #                             Mc=Mc_z,
        #                             mu=shared_info.params.baryon.mu,
        #                             thej=shared_info.params.baryon.thej,
        #                             cosmo_corr=cosmo_corr,
        #                             cosmo_bias=cosmo_bias,
        #                             param=shared_info.params,
        #                             rhoc=shared_info.current_rho,
        #                             tr_cvir=tr_cvir,
        #                             tr_tau=tr_tau,
        #                             tr_Mvir=tr_Mvir)



# def get_pix_and_displ_interp(ddmb, halo_slice, shared_info):

#     # global shared_info

#     # create a shell collection
#     shell_info = dict(n_halos=0,
#                       n_halos_unique=0,
#                       n_parts=0,
#                       n_parts_unique=0,
#                       shell_path=shared_info.shell_path)

#     # get all halos with the same id
#     same_id_halos = shared_info.current_halos[halo_slice]
#     first_halo = same_id_halos[0]

#     # get the displacement splines
#     DDMB_tck = splrep(shared_info.rbin, ddmb, s=0, k=3)

#     # rball
#     if shared_info.params.code.maxRvir * first_halo['rvir'] < shared_info.params.code.rmax:
#         rball = shared_info.params.code.maxRvir * first_halo['rvir']
#     else:
#         rball = shared_info.params.code.rmax

#     # get the intersection angle
#     pos = np.array([first_halo['x'], first_halo['y'], first_halo['z']])
#     r_halo = np.linalg.norm(pos)
#     alpha = get_intersection_angle(shared_info.shell["shell_com"], rball, r_halo)

#     # other properties that won't change for identical halos
#     Mvir = first_halo["Mvir"]
#     cvir = first_halo["cvir"]
#     tr_Mvir = first_halo["tfNFW_Mvir"]
#     tr_cvir = first_halo["tfNFW_cvir"]
#     tr_tau = first_halo["tfNFW_tau"]

#     # bias and corr
#     cosmo_bias = shared_info.bias_func(Mvir)
#     cosmo_corr = shared_info.corr_func(shared_info.rbin)
#     Mc_z = shared_info.params.baryon.Mc*(1.0 + shared_info.current_z)**shared_info.params.baryon.nu

#     # get the displacement splines
#     DDMB_tck = splrep(shared_info.rbin, ddmb, s=0, k=3)

#     # cycle though all positions
#     for n_subhalo, sub_halo in enumerate(same_id_halos):

#         pos = np.array([sub_halo['x'], sub_halo['y'], sub_halo['z']])
#         r_halo = np.linalg.norm(pos)
#         pos_norm = pos / r_halo

#         # get the particles
#         pixels = hp.query_disc(nside=shared_info.shell_nside, vec=pos_norm, radius=alpha)

#         # if we have less than four particles we set the angle to None and take the neighbors
#         if not shared_info.params.code.force_parts and len(pixels) < 4:
            
#             # we do not have enough particles so we skip the halo
#             continue
        
#         elif len(pixels) < 4:
            
#             # pixels and weights have shape (4, N)
#             theta, phi = hp.vec2ang(pos_norm)
#             pixels, _ = hp.get_interp_weights(nside=shared_info.shell_nside, theta=theta, phi=phi)
#             # reshape and get pars
#             pixels = pixels.ravel()

#         LOGGER.debug(f" Shell {shared_info.shell['shell_id']} is dealing with halo {sub_halo['ID']} "
#                     f"({n_subhalo+1}/{len(same_id_halos)}), found {len(pixels)} particles...")

#         # if we get to here, we update the info (we count empty pixels as well)
#         if n_subhalo == 0:
#             shell_info["n_halos_unique"] += 1
#         shell_info["n_halos"] += 1
#         shell_info["n_parts"] += len(pixels)

#         # now that we have the pixels we calculate the displacement
#         parts_pos = np.stack(hp.pix2vec(nside=shared_info.shell_nside, ipix=pixels), axis=1)
        
#         # blow up to radius
#         parts_pos *= shared_info.current_com

#         # relative distance
#         rpDMB = np.linalg.norm(parts_pos - pos, axis=1, keepdims=True)

#         # displace - this is the main magic here
#         DrpDMB = splev(rpDMB, DDMB_tck, der=0, ext=1)
#         displacement = (parts_pos - pos) * DrpDMB / rpDMB

#     return shell_info, pixels, displacement

















# def get_halo_displacement_grid(halos, current_info, n_grid=int(2**14), sequence='hammersley'):

#     assert np.all(halos['tfNFW_cvir']==-1), 'halo interpolation speedup works only for halo model with tfNFW_cvir=-1'

#     LOGGER.info(f'getting halo displacements for n_grid={n_grid} points in Mvir, cvir')

#     # create sobol sampling with slightly extended convex hull of the data, in log space
#     # this is needed to avoid numerical error through the log10 transform
#     x = np.vstack([halos['Mvir'], halos['cvir']]).T
#     grid_x = 10**sample_sequence_in_hull(x=np.log10(scale_range(x, factor=1.001)),
#                                          n_samples=n_grid, 
#                                          sequence=sequence)

#     # evaluate grid using the full method - create training set
#     list_ddmb = []
#     list_r = []
#     for i, s in LOGGER.progressbar(list(enumerate(grid_x)), at_level='debug', desc='building interpolator'):    
#         r, ddmb = get_displ_vec(Mvir=s[0], 
#                                 cvir=s[1], 
#                                 tfNFW_Mvir=halos['tfNFW_Mvir'][0],
#                                 tfNFW_cvir=halos['tfNFW_cvir'][0], 
#                                 tfNFW_tau=halos['tfNFW_tau'][0], 
#                                 shared_info=current_info)
#         list_r.append(r)
#         list_ddmb.append(ddmb)
#     ddmb = np.array(list_ddmb)
#     r = np.array(list_r)

#     return grid_x, ddmb, r


# def interp_halo_displacement_profiles(halos, current_info, n_grid=int(2**14)):

#     from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator

#     assert np.all(halos['tfNFW_cvir']==-1), 'halo interpolation speedup works only for halo model with tfNFW_cvir=-1'

#     n_tries = 10

#     list_grid_x = []
#     list_ddmb = []
#     list_r = []
#     frac_test = 0.1
#     err_thresh = 1e-4
#     for i in range(n_tries):

#         n_grid_train = n_grid*(2**(i))
#         n_grid_test = int(n_grid_train*frac_test)

#         grid_x_train, ddmb_train, r_train = get_halo_displacement_grid(halos, current_info, n_grid=n_grid_train, sequence='hammersley')
#         grid_x_test, ddmb_test, r_est = get_halo_displacement_grid(halos, current_info, n_grid=n_grid_test, sequence='latin_hypercube')

#         list_grid_x.append(grid_x_train)
#         list_ddmb.append(ddmb_train)
#         list_r.append(r_train)

#         grid_x_all = np.concatenate(list_grid_x, axis=0)
#         ddmb_all = np.concatenate(list_ddmb, axis=0)
#         r_all = np.concatenate(list_r, axis=0)

#         interp = CloughTocher2DInterpolator(points=grid_x_all, values=ddmb_all, fill_value=0, rescale=True)
#         ddmb_pred = interp(grid_x_test)
#         max_err = np.max( np.sqrt(np.mean((ddmb_test-ddmb_pred)**2, axis=1))) 
#         LOGGER.info(f'n_grid_current={len(grid_x_all)} max_err={max_err:2.4e} thresh={err_thresh}')
#         if max_err<err_thresh:
#             break


#     # LOGGER.info(f'building halo profiles interpolator with n_grid={n_grid} training points')

#     # # create sobol sampling with slightly extended convex hull of the data, in log space
#     # # this is needed to avoid numerical error through the log10 transform
#     # x = np.vstack([halos['Mvir'], halos['cvir']]).T
#     # grid_x = 10**sample_sequence_in_hull(x=np.log10(scale_range(x, factor=1.001)),
#     #                                      n_samples=n_grid, 
#     #                                      sequence='hammersley')

#     # # evaluate grid using the full method - create training set
#     # list_ddmb = []
#     # list_r = []
#     # for i, s in LOGGER.progressbar(list(enumerate(grid_x)), at_level='debug', desc='building interpolator'):    
#     #     r, ddmb = get_displ_vec(Mvir=s[0], 
#     #                             cvir=s[1], 
#     #                             tfNFW_Mvir=halos['tfNFW_Mvir'][0],
#     #                             tfNFW_cvir=halos['tfNFW_cvir'][0], 
#     #                             tfNFW_tau=halos['tfNFW_tau'][0], 
#     #                             shared_info=current_info)
#     #     list_r.append(r)
#     #     list_ddmb.append(ddmb)
#     # ddmb = np.array(list_ddmb)
#     # r = np.array(list_r)

#     import pudb; pudb.set_trace();
#     pass

#     halos_x = np.vstack([halos['Mvir'], halos['cvir']]).T
#     LOGGER.info(f'interpolating {len(halos_x)} points with CloughTocher2DInterpolator')
#     time_start = time.time()
#     interp = CloughTocher2DInterpolator(points=grid_x_all, values=ddmb_all, fill_value=0, rescale=True)
#     # interp = CloughTocher2DInterpolator(points=grid_x[len(grid_x)//10:], values=ddmb[len(grid_x)//10:], fill_value=0, rescale=True)
#     # ddmb_pred = interp(grid_x[:len(grid_x)//10])
#     # ddmb_test = ddmb[:len(grid_x)//10]
#     # max_err = np.max( np.sqrt(np.mean((ddmb_test-ddmb_pred)**2, axis=1))) 
#     LOGGER.info(f'done! time={time.time()-time_start:5.2f}s')


#     ddmb_interp = interp(halos_x)

#     # import pudb; pudb.set_trace();
#     # pass
#     # np.save('/cluster/work/refregier/tomaszk/projects/220708_cosmogrid_paper/005_cosmogridv11/bad_halos_x.npy', halos_x)
#     # np.save('/cluster/work/refregier/tomaszk/projects/220708_cosmogrid_paper/005_cosmogridv11/bad_log_x.npy', np.log10(x))
#     # np.save('/cluster/work/refregier/tomaszk/projects/220708_cosmogrid_paper/005_cosmogridv11/bad_grid_x.npy', grid_x)
#     # np.save('/cluster/work/refregier/tomaszk/projects/220708_cosmogrid_paper/005_cosmogridv11/bad_ddmb_interp.npy', ddmb_interp)
#     # np.save('/cluster/work/refregier/tomaszk/projects/220708_cosmogrid_paper/005_cosmogridv11/bad_ddmb.npy', ddmb)
#     # np.save('/cluster/work/refregier/tomaszk/projects/220708_cosmogrid_paper/005_cosmogridv11/bad_halos.npy', halos)


#     return r, ddmb_interp
