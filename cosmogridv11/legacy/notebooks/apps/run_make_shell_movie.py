"""
Make shell movie

Created July 2022
author: Tomasz Kacprzak
"""
import matplotlib
matplotlib.use('agg')

import io, os, sys, warnings, argparse, h5py, numpy as np, time, logging, itertools, shutil
from cosmogrid_des_y3 import utils_io, utils_logging, utils_config, utils_cosmogrid
from cosmogrid_des_y3.filenames import *
import healpy as hp
from cosmogrid_des_y3 import utils_io, utils_cosmogrid
from UFalcon import probe_weights

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

import sys, os, io, numpy as np, h5py, bz2, pylab as plt, importlib, seaborn, pickle, pandas, healpy as hp, warnings, glob
from pandas import DataFrame
from tqdm.auto import tqdm, trange
warnings.filterwarnings("ignore", category=DeprecationWarning) 
plt.matplotlib.rcParams.update({"axes.grid" : False})

simslist_grid, parslist_grid, shell_info_grid = utils_cosmogrid.get_simulations_list(set_type='grid')
simslist_fiducial, parslist_fiducial, shell_info_fiducial = utils_cosmogrid.get_simulations_list(set_type='fiducial')

path='/home/ipa/refreg/data/data_products/'
host='tomaszk@login.phys.ethz.ch'
n_shells = 69
shells_ids_all = range(n_shells)
# redshift_z_groups = np.arange(0.05, 2.65, 0.1)
redshift_z_groups =  np.concatenate([np.linspace(0, 0.35, 8), np.arange(0.45, 3.15, 0.1)])
nside_out = 1024
fontsize_suptitle = 22
fontsize_link = 18
pos_suptitle=0.8
fontsize_title = 19
ticklabel_size = 16

model_colors = ['red', 'blue', 'green', 'violet', 'lime']

def setup(args):

    description = 'Make maps'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=str, default='info', choices=('critical', 'error', 'warning', 'info', 'debug'), 
                        help='logging level')
    parser.add_argument('--config', type=str, required=True, 
                        help='configuration yaml file')
    parser.add_argument('--dir_out', type=str, required=True, 
                        help='output dir for the results')
    parser.add_argument('--map_scale', type=str, default='lin', 
                        help='map scale: linear [lin] or log [log] (with caveats)')
    parser.add_argument('--test', action='store_true',
                        help='test mode')
    
    args = parser.parse_args(args)

    utils_logging.set_all_loggers_level(args.verbosity)

    # get absolute paths
    args.config = utils_io.get_abs_path(args.config)
    args.dir_out = utils_io.get_abs_path(args.dir_out)
    utils_io.robust_makedirs(args.dir_out)


    return args


def resources(args):
    
    res = {'main_memory':96000,
           'main_time_per_index':4, # hours
           'main_scratch':6500,
           'merge_memory':64000,
           'merge_time':24}
           # 'pass':{'constraint': 'knl', 'account': 'des', 'qos': 'regular'}} # Cori
    
    return res


def select_cases():

    # download sims

    id1 = select_param(s8=0.6, w0=-1, Om=0.45, delta=0.03)
    id2 = select_param(s8=1.2, w0=-1, Om=0.15, delta=0.03)
    id3 = select_param(s8=0.75, w0=-1, Om=0.225, delta=0.03)
    id4 = select_param(s8=1.05, w0=-1, Om=0.3, delta=0.03)
    id5 = select_param(s8=0.84, w0=-0.4, Om=0.26, delta=0.03)
    ids = [-1, id1, id2, id3, id4, id5]

    return ids


def preprocess(indices, args):

    args = setup(args)
    conf = utils_config.load_config(args.config)
    ids = select_cases()

    for i in ids:
        download_sim(i, cmd_only=False)

    
    convert_shells_to_hdf_nside(ids, nside=nside_out)


def main(indices, args):


    args = setup(args)
    conf = utils_config.load_config(args.config)

    ids = select_cases()
    ids_pars = simslist_grid[ids]['id_param']

    shell_info_use = shell_info_fiducial[parslist_fiducial[0]['path_par']]
    redshift_z = (shell_info_use['upper_z'] + shell_info_use['lower_z'])/2.
    tags = [make_tag(id_sim) for id_sim in ids]

    # import ipdb; ipdb.set_trace(); 
    # pass

    redshift_z_groups_lower = [0,  6, 11, 15, 18, 20] + list(np.arange(21,68))
    redshift_z_groups_upper = [6, 11, 15, 18, 20, 21] + list(np.arange(22,69))

    for i in indices:

        LOGGER.info(f'============================ index={i}')

        if i>=len(redshift_z_groups_lower):
            yield 0

        s1 = redshift_z_groups_lower[i] 
        s2 = redshift_z_groups_upper[i] 
        select = np.arange(s1, s2, dtype=int)
        LOGGER.info(f"using shells {s1}, {s2}, {select}")
        select_z = np.round((shell_info_use[s1]['lower_z'] + shell_info_use[s2]['upper_z'])/2., decimals=2)
        LOGGER.info(f'z={np.round(select_z, decimals=2):2.2f}')

        if True:

            if len(select)==0:
                LOGGER.warning(f'no shells selected, skipping.. z={redshift_z_groups[i-1]}-{redshift_z_groups[i]}')
                continue
                
            list_m = []
            for id_sim in ids:
                with h5py.File(f'sim{id_sim}_nside{nside_out}.h5', 'r') as f:
                    m = np.array([f[f'{i}'] for i in select])
                list_m.append(m)
            m = np.array(list_m)
            LOGGER.info(f'loaded array with shape={m.shape}')

            maps_current = appy_map_transform(m, scale=args.map_scale)

            if len(select) == 1:
                title = 'CosmoGrid, redshift z' + r'$\sim$' + '{:2.2f}, shell  {: 2d}'.format(select_z, np.min(select)+1, np.max(select)+1)
            else:
                title = 'CosmoGrid, redshift z' + r'$\sim$' + '{:2.2f}, shells {: 2d} -{: 2d}'.format(select_z, np.min(select)+1, np.max(select)+1)

            LOGGER.info(f"{title} {str(select)}")
            store_fig(i, maps_current, 
                      tags_sim=tags, 
                      ids_sims=ids_pars,
                      test=args.test,
                      title=title,
                      tag='redshift',
                      dir_out=args.dir_out);

        yield i


def appy_map_transform(m, scale='linear'):

    if scale=='lin':    

        # avertage all shells to create one map
        maps_current = np.mean(m, axis=1)

        # density constrast
        maps_current = maps_current/np.mean(maps_current, axis=-1, keepdims=True)-1

    elif scale=='log':

        # sum all shells to create one map
        maps_current = np.sum(m, axis=1)

        # log transform starting with 1, move away from zero
        maps_current = np.log(maps_current+1)

    elif scale=='clg':

        # sum all shells to create one map
        maps_current = np.sum(m, axis=1)+1

        # density constrast
        maps_current = maps_current/np.mean(maps_current, axis=-1, keepdims=True)

        # log transform starting with 1, move away from zero
        maps_current = np.log(maps_current)


    LOGGER.info(f'applied transformation {scale}, maps_current.shape={maps_current.shape}, min={maps_current.min()}, max={maps_current.max()}')

    return maps_current

    



def select_param(s8, w0, Om, delta):

    par_title = lambda s: f"Omega_m={s['Om']:2.2f} sigma_8={s['s8']:2.2f} w0={s['w0']:2.2f}"
    
    select = (np.abs(simslist_grid['s8']-s8)<delta) & (np.abs(simslist_grid['w0']-w0)<delta) & (np.abs(simslist_grid['Om']-Om)<delta)
    LOGGER.info('selected {} shells'.format(np.count_nonzero(select)))
    ind = np.nonzero(select)[0][0]
    LOGGER.info('parameter string: {}'.format(par_title(simslist_grid[ind])))
    return ind

def download_sim(id_sim, cmd_only=False, sim_type='grid'):
    
    if id_sim == -1:
        cmd = f"rsync -avR --progress {host}:{path}/./CosmoGrid/raw/fiducial/cosmo_fiducial/run_0/compressed_shells.npz  ."
    else:
        if sim_type == 'grid':
            cmd = f"rsync -avR --progress {host}:{path}/./{simslist_grid[id_sim]['path_sim']}/compressed_shells.npz  ."        
        elif sim_type == 'fiducial':
            cmd = f"rsync -avR --progress {host}:{path}/./{simslist_fiducial[id_sim]['path_sim']}/compressed_shells.npz  ."        
        
    
    LOGGER.info(cmd)
    if not cmd_only:
        os.system(cmd)

def make_tag(id_par):
    
    if id_par==-1:
        s8 = 0.84
        om = 0.26
        w0 = -1
    else:
        s8 = simslist_grid[id_par]['s8']
        om = simslist_grid[id_par]['Om']
        w0 = simslist_grid[id_par]['w0']
    tag = r'$\Omega_m={:2.2f} \quad \sigma_8={:2.2f} \quad w_0={:2.1f}$'.format(om, s8, w0)
    return tag


def nanlog(x):
    x_log = np.log(x)
    x_log[~np.isfinite(x_log)]=np.min(x_log[np.isfinite(x_log)])
    return x_log

def plot_skymapper_maps(maps_sphere, tags, title, ids_sims, vmin=None, vmax=None, test=False, cmap='plasma', lat0=0, lon0=0, size_deg=30, graticule_sep=10):

    import skymapper as skm
    lon_0, lat_0, lat_1, lat_2 = 40, 60, 20, 20
    proj = skm.Albers(lon_0, lat_0, lat_1, lat_2)
    
    nx, ny = 2, 3; fig, ax = plt.subplots(nx, ny, figsize=(ny * 6, nx * 8), squeeze=False); axc=ax[0,0]; axl=ax[0,:];
    axr = ax.ravel()
    
    for i, map_sphere in LOGGER.progressbar(list(enumerate(maps_sphere)), desc='making plot panels'):
        
        fiducial = i==0

        m = skm.Map(proj, ax=axr[i])

    #     log_map = (map_sphere-np.mean(map_sphere))/np.mean(map_sphere)
#         log_map = nanlog(map_sphere/np.mean(map_sphere))
#         log_map = (map_sphere/np.mean(map_sphere)-1)
        log_map = map_sphere

        if not test:

            if vmin is None:
                vmin = np.percentile(log_map, 2)

            if vmax is None:
                vmax = np.percentile(log_map, 98)

            m.healpix(log_map, cmap=cmap, vmin=vmin, vmax=vmax)

        m.focus(lon=np.array([lon0,lon0+size_deg]), lat=np.array([lat0,lat0+size_deg]))
        m.grid(sep=graticule_sep, lw=2, c='k', alpha=1)
    #     axl[0].set_title(tag, fontsize=fontsize_title)

        tx = axr[i].text(0.5, 0.91, tags[i], fontsize=fontsize_title, 
                         transform=axr[i].transAxes, ha='center', va='center', zorder=1000)
        tx.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='black',  boxstyle='square,pad=0.25'))

        if fiducial:
            tag_sim_id = r'$\bigstar$ fiducial'
        else:
            tag_sim_id = f'model {ids_sims[i]}/2500'

        tx2 = axr[i].text(0.5, 0.75, tag_sim_id, fontsize=fontsize_title, 
                         transform=axr[i].transAxes, ha='center', va='bottom', zorder=1000)
        tx2.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='black',  boxstyle='square,pad=0.25'))


        axr[i].tick_params(axis='both', labelsize=ticklabel_size)
    #     axl[0].yaxis.tick_left()
    #     axl[0].xaxis.tick_bottom()


    ax[0,0].set(xticks=[])
    ax[0,1].set(xticks=[])
    ax[0,2].set(xticks=[])
        
    ax[0,1].set(yticks=[])
    ax[0,2].set(yticks=[])
    ax[1,1].set(yticks=[])
    ax[1,2].set(yticks=[])


            
    fig.suptitle(title, fontsize=fontsize_suptitle, y=pos_suptitle, x=0.05, fontfamily='monospace', ha='left')
    fig.text(x=1.025, y=0.17, s='www.cosmogrid.ai', ha='right', fontsize=fontsize_link)
    # fig.subplots_adjust(hspace=0.0001)
    
    return fig

def load_hp_shells(id_sim, ids_shell, nside_out=256):
    
    if id_sim == -1:
        fname = "CosmoGrid/raw/fiducial/cosmo_fiducial/run_0/compressed_shells.npz"
    else:
        fname = f"{simslist_grid[id_sim]['path_sim']}/compressed_shells.npz"
    
    LOGGER.info('loading {}'.format(fname))
    sim_shells = np.array(np.load(fname)['shells'])
    s_map = []
    for id_shell in tqdm(ids_shell):
        s_map_ = hp.ud_grade(sim_shells[id_shell], nside_out)
        s_map.append(s_map_)
    return s_map

def store_fig(i, maps_current, ids_sims, tags_sim, test=False, title='', tag='pastlook', dir_out='.'):
    
    vmin, vmax = np.percentile(maps_current, [2, 98])
    LOGGER.info(f"creating map {i}, min={maps_current.min()}, max={maps_current.max()}, lims=[{vmin}, {vmax}]")
    
    fname = os.path.join(dir_out, f'map_{tag}{i:03d}.png')
    
    if os.path.isfile(fname):

        LOGGER.critical(f'image frame exists, skipping ... {fname}')

    else:
    
        fig = plot_skymapper_maps(maps_current, 
                                  ids_sims=ids_sims, 
                                  tags=tags_sim,
                                  test=test,
                                  cmap='plasma',
                                  vmin=vmin,
                                  vmax=vmax,
                                  title=title)

        fig.savefig(fname, bbox_inches='tight', dpi=300)
        LOGGER.info(f'stored {fname}')
    

def convert_shells_to_hdf_nside(ids, nside):

    for id_sim in ids:
        LOGGER.info(id_sim)

        fname = f'sim{id_sim}_nside{nside_out}.h5'

        if os.path.isfile(fname):

            LOGGER.info(f'file {fname} exists, skipping...')

        else:

            shells = load_hp_shells(id_sim, ids_shell=shells_ids_all, nside_out=nside)
            with h5py.File(fname, 'w') as f:
                for id_shell, shell in LOGGER.progressbar(list(enumerate(shells)), desc=f'storing sim {id_sim}'):
                    f.create_dataset(name=f'{id_shell}', data=shell, compression='lzf', shuffle=True)



