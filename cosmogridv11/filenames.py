import os, sys

def get_filename_probe_kernels(dir_out):

    return os.path.join(dir_out, f'probe_weights.h5')

def get_dirname_probe_kernels(dir_out, cosmo_params, project_tag):

    dir_output = os.path.join(dir_out, cosmo_params['path_par'].replace('raw', project_tag))
    return dir_output

def get_filepath_projected_maps(dir_out, variant):

    return os.path.join(dir_out, f'projected_probes_maps_{variant}.h5')

def get_filepath_patches(dir_out, variant):

    return os.path.join(dir_out, f'probes_patches_{variant}.h5')

def get_filepath_power_spectra(dir_out, variant):

    return os.path.join(dir_out, f'power_spectra_{variant}.h5')    

def get_filepath_permutations_index(dir_out):

    return os.path.join(dir_out, f'shell_permutations_index.h5')    

def get_dirname_projected_maps(dir_out, cosmo_params, project_tag, id_run):

    dirtag = f'/proj_{project_tag}/'
    path_par = cosmo_params['path_par'].replace('/bary/', dirtag).replace('/raw/', dirtag)
    dir_output = os.path.join(dir_out, path_par, f"run_{id_run}")
    return dir_output

def get_dirname_permuted_maps(dir_out, cosmo_params, project_tag, id_perm):

    dir_output = os.path.join(dir_out, cosmo_params['path_par'].replace('raw', project_tag), f'perm_{id_perm:04d}')
    return dir_output

def get_filepath_features(dir_out):

    return os.path.join(dir_out, f'features.h5')    

def get_filepath_features_merged(dir_out):

    return os.path.join(dir_out, f'features_merged.pkl')    

def get_filename_patch_maps(dir_out, index_perm, probe, index_patch, id_sample1, id_sample2):

    return os.path.join(dir_out, f"xmap{index_perm}_{probe}_patch{index_patch}_sample{id_sample1}x{id_sample2}.fits")

def get_filename_baryonified_shells(dir_out, tag):

    return os.path.join(dir_out, f"baryonified_shells_{tag}.h5")

def get_filename_profiled_halos(dir_out, tag):

    return os.path.join(dir_out, f"profiled_halos_{tag}.h5")

def get_filename_baryonification_info(dir_out, tag):
    
    return os.path.join(dir_out, f"baryonified_shells_{tag}.info")

def get_filepath_permlist(dir_out):

    return os.path.join(dir_out, f'metainfo_perms.npy')    

def get_filepath_barylist(dir_out, tag):

    return os.path.join(dir_out, f'CosmoGridV11_metainfo_bary_{tag}.h5')    


def get_filepath_haloshells(dir_out, tag):

    return os.path.join(dir_out, f'haloshells_{tag}.h5')

def get_filename_compressed_shells(path_sim):

    # high redshift resolution shells are stored in h5 as opposed to npz
    fname = 'compressed_shells.h5' if 'redshift_resolution' in path_sim else 'compressed_shells.npz'
    
    return fname