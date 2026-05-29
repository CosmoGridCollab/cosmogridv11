
import numpy as np, healpy as hp, h5py
from . import utils_logging
LOGGER = utils_logging.get_logger(__file__)

# from numba import njit

# @njit 
def map_double_sum(m, shells, weight_shell, weight):

    for j in range(len(shells)):

        for i in range(0, j):

            w_ = weight_shell[j] * weight[i,j]

            if w_ > 0:
                     
                m += shells[i]*shells[j]*w_

def project_shells(shells, weight, weight_shell, order='linear', integ_const=0):
    """Summary
    
    Parameters
    ----------
    shells : TYPE
        Description
    weight : TYPE
        Description
    weight_shell : TYPE
        Description
    order : str, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    """

    n_pix = shells.shape[1]
    m = np.zeros(n_pix, dtype=np.float64)
    # m_temp = np.zeros(n_pix, dtype=np.float64)

    single_loop = len(weight.shape) == 1

    if single_loop:

        for i in range(len(shells)):

            if weight[i] != 0:

                # m_temp[:] = shells[i]
                # m_temp[:] *= np.float64(weight[i])
                # m += m_temp
                m += np.float64(shells[i]) * np.float64(weight[i])

    else:

        # remove mean
        shells = shells.astype(np.float64)
        for i in range(len(shells)):
            shells[i] -= np.mean(shells[i])

        map_double_sum(m, shells, weight_shell, weight)

        # for j in range(len(shells)):

        #     # sum over lens shells
        #     for i in range(0, j):

        #         w_ = (weight_shell[j] * weight[i,j])

        #         if w_ > 0:
                     
        #             # m_temp[:]  = shells[i]
        #             # m_temp[:] *= shells[j]
        #             # m_temp[:] *= w_
        #             # m += m_temp

        #             m += shells[i]*shells[j]*w_

    if integ_const != 0:
        m -= integ_const

    if order=='quadratic':
        m[:] *= m[:] # limber-los approximation

    return m.astype(np.float32)


# def project_shells_source_clustering(shells, weight, weight_shell):
#     """
#     :param shells:
#     :param weight:
#     :param weight_shell:
#     :return m:
#     """


#     # prefactor_sim = probe_weights.sim_spec_prefactor(n_pix=hp.nside2npix(n_side), 
#     #                                                  box_size=box_size, 
#     #                                                  n_particles=n_particles)

#     n_pix = shells.shape[1]
#     weight = np.where(weight<0, 0, weight)
#     assert weight.ndim == 2, f'weight.shap={weight.shape}, need to pass 2 dimensional lensing weight array'

#     # sum over source shells
#     # arxiv:2105.13548 equation 46, version for real space kappa (not harmonic shear)
#     # accumulator in float64
#     m = np.zeros(n_pix, dtype=np.float64)

#     # remove mean
#     shells = shells.astype(np.float64)
#     for i in range(len(shells)):
#         shells[i] -= np.mean(shells[i])

#     # main magic loop
#     m_temp = np.zeros(n_pix, dtype=np.float64)
#     for j in LOGGER.progressbar(range(len(shells)), desc='projecting source clustering shells', at_level='debug'):

#         # sum over lens shells
#         for i in range(0, j):

#             w_ = (weight_shell[j] * weight[i,j])

#             if w_ > 0:
                 
#                 m_temp[:]  = shells[i]
#                 m_temp[:] *= shells[j]
#                 m_temp[:] *= w_
#                 m += m_temp
               

#     return m.astype(np.float32)


def project_all_probes(shells, nz_info, shell_weight):

    available_probes = ['dg2', 'dg', 'kg', 'ia', 'kcmb', 'kd', 'dh', 'dh2']

    # output container
    probe_maps = {}

    # project against different kernels
    for i, nzi in enumerate(nz_info):

        sample = nzi['name']
        probes = nzi['probes']

        for j, probe in enumerate(probes):
            
            integ_const = 0

            LOGGER.info(f'creating map probe={probe:>4s} sample={sample}')
            probe_maps.setdefault(probe, {})

            if probe not in available_probes:
                raise Exception(f'unknown probe {probe}')

            try:

                # decide shell type
                if probe in ['dg', 'kg', 'ia', 'kd', 'kcmb']:
                    shells_key = 'particles'
                
                elif probe == 'dg2':
                    shells_key = 'particles_smoothed'

                elif probe == 'dh':
                    shells_key = 'halos_rho'
                
                elif probe == 'dh2':
                    shells_key = 'halos_rhosq'

                
                # decide order
                if probe == 'dg2':
                    order = 'quadratic' 
                    integ_const = nzi['c_dg2']

                else:
                    order = 'linear'

                # decide kernel functions
                if probe in ['dh', 'dh2']:
                    weight='w_dg'
                else:
                    weight = f'w_{probe}'

                # main magic
                m = project_shells(shells[shells_key], nzi[weight], shell_weight, order, integ_const)

                probe_maps[probe][sample] = m


            except Exception as err:

                LOGGER.error(f'failed to create map probe={probe} errmsg={err}')

                probe_maps[probe][sample] = None

    return probe_maps



def add_highest_redshift_shell(probe_maps, nz_info, sim_params, parslist_all, seed):

    np.random.seed(int(seed))

    for nz in nz_info:

        for probe in nz['probes']:

            if 'file_high_z_cls' in nz.keys():

                for zbin in nz['file_high_z_cls'].keys():

                    cl = load_highz_cls(nz['file_high_z_cls'][zbin])

                    # cl object has a row for each row in parslist, 2521 rows, cross match by path_par
                    select = np.nonzero(sim_params['path_par'] == parslist_all['path_par'])[0][0]

                    # get cl and nside
                    cl_current = cl[select]
                    nside = hp.npix2nside(len(probe_maps[probe][nz['name']]))

                    LOGGER.info(f'adding z>3.5 synfast map for probe {probe} bin {nz["name"]}')
                    
                    # generate a synfast map
                    m = hp.sphtfunc.synfast(cl_current, nside=nside, pixwin=True, verbose=False)

                    # add to probe map
                    probe_maps[probe][nz['name']] += m


    return probe_maps

def load_highz_cls(fname):

    with h5py.File(fname, 'r') as f:
        id_params = np.array(f['id_params'])
        deltas = np.array(f['deltas'])
        benchmarks = np.array(f['benchmarks'])
        cl_kk = np.array(f['cl_kk'])

    return cl_kk

def check_zbound(shell_info, s, z_lim_up, probe):

    if z_lim_up < shell_info['lower_z'][s]:
                       
        # LOGGER.debug(f"shell outside redshift range max_z_bin={z_lim_up:2.4f} min_z_shell={shell_info['lower_z'][s]:2.4f}")
        return False

    else:

        # LOGGER.debug(f'========== probe={probe} shell {s: 4d}/{len(shell_info)}')
        return True




#################################################################################################
#################################################################################################
###
### Kernel functions
###
#################################################################################################
#################################################################################################


def get_kernel_shell(shell_info, astropy_cosmo, sim_params, kw_ufalcon, test=False):

    from UFalcon import probe_weights
    
    z_lim_up = kw_ufalcon['z_lim_up']
    w_shell = np.zeros(len(shell_info))

    prefactor_delta = probe_weights.delta_prefactor_nosim(cosmo=astropy_cosmo)
    prefactor_sim = probe_weights.sim_spec_prefactor(**sim_params)
    weight_generator_shell = probe_weights.Dirac_clustering()

    for s in range(len(shell_info)):

        if check_zbound(shell_info, s, z_lim_up, probe='shell weight'):

            w = weight_generator_shell(z_low=shell_info['lower_z'][s], 
                                       z_up=shell_info['upper_z'][s], 
                                       cosmo=astropy_cosmo,
                                       lin_bias=1)

            w_shell[s] = w * prefactor_delta * prefactor_sim

    return w_shell


def get_kernel_kg(sample, shell_info, astropy_cosmo, sim_params, kw_ufalcon, test=False):

    from UFalcon import probe_weights

    probe = 'kg'
    z_lim_up = kw_ufalcon['z_lim_up']
    sample[f'w_kg'] = np.zeros((len(shell_info)))
    
    prefactor_kappa = probe_weights.kappa_prefactor_nosim(cosmo=astropy_cosmo)
    prefactor_sim = probe_weights.sim_spec_prefactor(**sim_params)
    weight_generator_kg = probe_weights.Continuous_lensing(n_of_z=sample['nz'], 
                                                           fast_mode=2,
                                                           **kw_ufalcon)
    weight_generator_kg.set_cosmo(cosmo=astropy_cosmo)

    # lens shells
    for s in range(len(shell_info)):

        if check_zbound(shell_info, s, z_lim_up, probe):

            w_kg = weight_generator_kg(z_low=shell_info['lower_z'][s], 
                                       z_up=shell_info['upper_z'][s], 
                                       cosmo=astropy_cosmo)

            sample['w_kg'][s] = w_kg * prefactor_kappa * prefactor_sim

    
    return sample


def get_kernel_kd(sample, shell_info, astropy_cosmo, sim_params, kw_ufalcon, test=False):
    
    from UFalcon import probe_weights

    probe = 'kd'
    z_lim_up = kw_ufalcon['z_lim_up']
    sample[f'w_kd'] = np.zeros((len(shell_info), len(shell_info)))
    
    prefactor_kappa = probe_weights.kappa_prefactor_nosim(cosmo=astropy_cosmo)
    prefactor_sim = probe_weights.sim_spec_prefactor(**sim_params)
    weight_generator_kd = probe_weights.Continuous_lensing(n_of_z=sample['nz'], 
                                                           fast_mode=2,
                                                           **kw_ufalcon)
    weight_generator_kd.set_cosmo(cosmo=astropy_cosmo)

    # lens shells
    for s1 in range(len(shell_info)):

        if check_zbound(shell_info, s1, z_lim_up, probe):

            # source shells
            for s2 in range(s1, len(shell_info)):

                w_kd = weight_generator_kd(z_low=shell_info['lower_z'][s1], 
                                           z_up=shell_info['upper_z'][s1],
                                           source_z_lims=[shell_info['lower_z'][s2], shell_info['upper_z'][s2]])

                sample['w_kd'][s1, s2] = w_kd * prefactor_kappa * prefactor_sim

        if test:

            w_kg_full = weight_generator_kd(z_low=shell_info['lower_z'][s1], 
                                            z_up=shell_info['upper_z'][s1]) * prefactor_kappa
            w_kg_split = np.sum(nz_info[j]['w_kd'], axis=1)
            LOGGER.debug('source clustering: checking split lensing kernel integration, {} split={:2.5e} full={:2.5e} diff={:2.5e}'.format(s1, w_kg_split[s1], w_kg_full, w_kg_split[s1]-w_kg_full))

    sample['nz_norm'] = weight_generator_kd.nz_norm

    return sample


def get_kernel_dg(sample, shell_info, astropy_cosmo, sim_params, kw_ufalcon, test=False):

    from UFalcon import probe_weights

    probe = 'dg'
    z_lim_up = kw_ufalcon['z_lim_up']
    sample[f'w_dg'] = np.zeros(len(shell_info))    

    weight_generator_dg = probe_weights.Continuous_clustering2(n_of_z=sample['nz'],
                                                               **kw_ufalcon)

    for s in range(len(shell_info)):

        if check_zbound(shell_info, s, z_lim_up, probe):
            
            w_dg = weight_generator_dg(z_low=shell_info['lower_z'][s], 
                                        z_up=shell_info['upper_z'][s], 
                                        bias=1,
                                        **sim_params,
                                        cosmo=astropy_cosmo,
                                        integ_const=False)

            sample['w_dg'][s] = w_dg

    sample['nz_norm'] = weight_generator_dg.nz_norm

    return sample

# def get_kernel_dg(sample, shell_info, astropy_cosmo, sim_params, kw_ufalcon, test=False):

#     probe = 'dg'
#     z_lim_up = kw_ufalcon['z_lim_up']
#     sample[f'w_dg'] = np.zeros(len(shell_info))

#     prefactor_delta = probe_weights.delta_prefactor_nosim(cosmo=astropy_cosmo)
#     prefactor_sim = probe_weights.sim_spec_prefactor(**sim_params)
#     weight_generator_dg = probe_weights.Continuous_clustering(n_of_z=sample['nz'],
#                                                               **kw_ufalcon)

#     for s in range(len(shell_info)):

#         if check_zbound(shell_info, s, z_lim_up, probe):

#             w_dg = weight_generator_dg(z_low=shell_info['lower_z'][s], 
#                                        z_up=shell_info['upper_z'][s], 
#                                        cosmo=astropy_cosmo,
#                                        lin_bias=1)

#             sample['w_dg'][s] = w_dg * prefactor_delta * prefactor_sim
    
#     sample['nz_norm'] = weight_generator_dg.nz_norm

#     return sample


def get_kernel_dg2(sample, shell_info, astropy_cosmo, sim_params, kw_ufalcon, test=False):

    from UFalcon import probe_weights

    probe = 'dg2'
    z_lim_up = kw_ufalcon['z_lim_up']
    sample[f'w_dg2'] = np.zeros(len(shell_info))    
    sample[f'c_dg2'] = np.zeros(len(shell_info))    

    weight_generator_dg2 = probe_weights.Continuous_clustering_quadratic_bias(n_of_z=sample['nz'],
                                                                              **kw_ufalcon)


    for s in range(len(shell_info)):

        if check_zbound(shell_info, s, z_lim_up, probe):
            
            w_dg2, c_dg2 = weight_generator_dg2(z_low=shell_info['lower_z'][s], 
                                                z_up=shell_info['upper_z'][s], 
                                                bias=1,
                                                **sim_params,
                                                cosmo=astropy_cosmo,
                                                integ_const=True)

            sample['w_dg2'][s] = w_dg2
            sample['c_dg2'][s] = c_dg2

    sample['c_dg2'] = np.sum(sample['c_dg2'])
    sample['nz_norm'] = weight_generator_dg2.nz_norm

    LOGGER.info(f"integ_const dg2: {sample['c_dg2']:2.4e}")

    return sample

# def get_kernel_ia(sample, shell_info, astropy_cosmo, sim_params, kw_ufalcon, test=False):
    
#     probe = 'ia'
#     z_lim_up = kw_ufalcon['z_lim_up']
#     sample[f'w_ia'] = np.zeros(len(shell_info))

#     prefactor_delta = probe_weights.delta_prefactor_nosim(cosmo=astropy_cosmo)
#     prefactor_sim = probe_weights.sim_spec_prefactor(**sim_params)
#     weight_generator_ia = probe_weights.Continuous_intrinsic_alignment(n_of_z=sample['nz'], 
#                                                                        IA=1.0, 
#                                                                        **kw_ufalcon)
    
#     for s in range(len(shell_info)):

#         if check_zbound(shell_info, s, z_lim_up, probe):

#             w_ia = weight_generator_ia(z_low=shell_info['lower_z'][s], 
#                                        z_up=shell_info['upper_z'][s], 
#                                        cosmo=astropy_cosmo)

#             sample['w_ia'][s] = w_ia * prefactor_delta * prefactor_sim
    
#     sample['nz_norm'] = weight_generator_ia.nz_norm

#     return sample



def get_kernel_ia(sample, shell_info, astropy_cosmo, sim_params, kw_ufalcon, test=False):

    from UFalcon import probe_weights
    
    probe = 'ia'
    z_lim_up = kw_ufalcon['z_lim_up']
    sample[f'w_ia'] = np.zeros(len(shell_info))
    weight_generator_ia = probe_weights.Continuous_intrinsic_alignment2(n_of_z=sample['nz'], 
                                                                       **kw_ufalcon)
    
    for s in range(len(shell_info)):

        if check_zbound(shell_info, s, z_lim_up, probe):

            w_ia = weight_generator_ia(z_low=shell_info['lower_z'][s], 
                                       z_up=shell_info['upper_z'][s], 
                                       A_IA=1,
                                       eta_IA=0,
                                       z_0=0.5, # does nothing anyway for eta_IA=0
                                       **sim_params,
                                       cosmo=astropy_cosmo)

            sample['w_ia'][s] = w_ia 
    
    sample['nz_norm'] = weight_generator_ia.nz_norm

    return sample


    # def get_kernel_dg(sample, shell_info, astropy_cosmo, sim_params, kw_ufalcon, test=False):

    # probe = 'dg'
    # z_lim_up = kw_ufalcon['z_lim_up']
    # sample[f'w_dg'] = np.zeros(len(shell_info))    

    # weight_generator_dg = probe_weights.Continuous_clustering2(n_of_z=sample['nz'],
    #                                                            **kw_ufalcon)

    # for s in range(len(shell_info)):

    #     if check_zbound(shell_info, s, z_lim_up, probe):
            
    #         w_dg = weight_generator_dg(z_low=shell_info['lower_z'][s], 
    #                                     z_up=shell_info['upper_z'][s], 
    #                                     bias=1,
    #                                     **sim_params,
    #                                     cosmo=astropy_cosmo,
    #                                     integ_const=False)

    #         sample['w_dg'][s] = w_dg

    # sample['nz_norm'] = weight_generator_dg.nz_norm

    # return sample

def get_kernel_kcmb(sample, shell_info, astropy_cosmo, sim_params, kw_ufalcon, test=False):

    from UFalcon import probe_weights

    probe = 'kcmb'
    z_lim_up = kw_ufalcon['z_lim_up']

    sample[f'w_kcmb'] = np.zeros(len(shell_info))

    prefactor_kappa = probe_weights.kappa_prefactor_nosim(cosmo=astropy_cosmo)
    prefactor_sim = probe_weights.sim_spec_prefactor(**sim_params)
    weight_generator_kg = probe_weights.Dirac_lensing(z_source=sample['z_cmb'])

    for s in range(len(shell_info)):

        if check_zbound(shell_info, s, z_lim_up, probe):

            w_kcmb = weight_generator_kg(z_low=shell_info['lower_z'][s], 
                                          z_up=shell_info['upper_z'][s], 
                                          cosmo=astropy_cosmo)

            sample['w_kcmb'][s] = w_kcmb * prefactor_kappa * prefactor_sim

    return sample

probe_kernel_funcs = {'kg': get_kernel_kg,
                      'dg': get_kernel_dg,
                      'ia': get_kernel_ia,
                      'dg2': get_kernel_dg2,
                      'kd': get_kernel_kd,
                      'kcmb': get_kernel_kcmb,
                      'dh': get_kernel_dg,
                      'dh2': get_kernel_dg2}



def store_probe_kernels(filename_out, nz_info, w_shell, mode='w'):

    all_probes = ['kd', 'kg', 'ia', 'dg', 'dg2', 'kcmb', 'dh', 'dh2']
    probe_weights_dict = {'kd':'w_kd', 'kg':'w_kg', 'ia':'w_ia', 'dg':'w_dg', 'dg2': 'w_dg2', 'kcmb':'w_kcmb', 'dh':'w_dg', 'dh2':'w_dg2'}
    
    dataset_name = lambda nzi, probe: f"kernel/{probe}/{nzi['name']}"

    datasets_stored = []
    with h5py.File(filename_out, mode) as f:

        f.create_dataset(name='w_shell', data=w_shell, compression='lzf', shuffle=True)

        for nzi in nz_info:

            if 'nz' in nzi.keys():

                d = f'nz/{nzi["name"]}'
                f.create_dataset(name=d, data=nzi['nz'], compression='lzf', shuffle=True)
                if 'nz_norm' in nzi.keys():
                    f[d].attrs['nz_norm'] = nzi['nz_norm']
                if 'c_dg2' in nzi.keys():
                    f[d].attrs['c_dg2'] = nzi['c_dg2']

            for probe in all_probes:

                w_key = probe_weights_dict[probe]
                if w_key in nzi.keys():

                    try:
                        d = dataset_name(nzi, probe)
                        f.create_dataset(name=d, data=nzi[w_key], compression='lzf', shuffle=True)
                        datasets_stored.append(d)

                    except Exception as err:
                        LOGGER.error(f'failed to store dataset {d} err={str(err)}')


    LOGGER.info(f'stored {filename_out} with {len(datasets_stored)} datasets')




def load_probe_kernels(filename_kernels, conf):

    nz_weights = {}

    with h5py.File(filename_kernels, 'r') as f:

        w_shell = np.array(f['w_shell'])

        for i, sample in enumerate(conf['redshifts_nz']):

            for probe in sample['probes']:


                # # source clustering uses lensing weights array (source-lens weights)
                # if probe  ==  'kd':
                  
                #     nz_weights.setdefault('kg_split', {})
                #     nz_weights['kg_split'][sample['name']] = np.array(f['kg_split'][sample['name']])

                # elif probe  ==  'dg2':

                #     assert f"dg/{sample['name']}" in f, 'kernels for linear clustering dg are needed for calculation of quadratic clustering'

                # # standard weights
                # else:

                nz_weights.setdefault(probe, {})
                nz_weights[probe][sample['name']] = np.array(f[probe][sample['name']])

    LOGGER.info(f'loading probe weigths {filename_kernels}') 

    return w_shell, nz_weights