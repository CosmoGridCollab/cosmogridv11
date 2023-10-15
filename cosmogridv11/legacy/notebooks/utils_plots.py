import numpy as np, healpy as hp, h5py, seaborn
from tqdm.auto import trange, tqdm
import pylab as plt



def average_power_spectra(n_sims=400, **kw_load):
    
    n_errors=0
    dict_ps_merge = {}
    for i in trange(0,n_sims):
        try:
            dict_ps = load_power_spectrum(id_sim=i, **kw_load)
        except Exception as err:
            n_errors += 1 
            continue
        
        for k1 in dict_ps.keys():
            if k1 not in dict_ps_merge.keys():
                dict_ps_merge[k1] = {}
            for k2 in dict_ps[k1].keys():
                if k2 not in dict_ps_merge[k1].keys():
                    dict_ps_merge[k1][k2] = dict_ps[k1][k2].copy()
                else:
                    dict_ps_merge[k1][k2] = np.concatenate([dict_ps_merge[k1][k2], dict_ps[k1][k2]])
    print(f'n_errors={n_errors}')
    return dict_ps_merge

def load_power_spectrum(id_sim=0, projtype='singlerun', variant='nobaryons', path_results='preproc_cosmogrid_stage3_v1'):
    dict_ps = {}
    prefix = 'run' if projtype=='singlerun' else 'perm'
    fname = f'{path_results}/{projtype}/CosmoGrid/raw/fiducial/cosmo_fiducial/{prefix}_{id_sim}/power_spectra_{variant}.h5'
    with h5py.File(fname, 'r') as f:
        for k1 in list(f.keys()):
            dict_ps[k1] = {}
            for k2 in list(f[k1].keys()):
                dict_ps[k1][k2] = np.atleast_2d(f[k1][k2])
    return dict_ps

def get_cl_sets(bin_name='stage3_lensing'):

    set_auto = [
                f'dg_{bin_name}1__dg_{bin_name}1',
                f'kg_{bin_name}1__kg_{bin_name}1',
                f'dg_{bin_name}2__dg_{bin_name}2',
                f'kg_{bin_name}2__kg_{bin_name}2',
                f'dg_{bin_name}3__dg_{bin_name}3',
                f'kg_{bin_name}3__kg_{bin_name}3',
                f'dg_{bin_name}4__dg_{bin_name}4',
                f'kg_{bin_name}4__kg_{bin_name}4',
                ]
    
    print('set_auto', len(np.unique(set_auto)))

    set_cross = [f'kg_{bin_name}4__kg_{bin_name}1',
                 f'dg_{bin_name}4__dg_{bin_name}1',
                 f'dg_{bin_name}1__kg_{bin_name}1',
                 f'dg_{bin_name}2__kg_{bin_name}2',
                 f'dg_{bin_name}3__kg_{bin_name}3',
                 f'dg_{bin_name}4__kg_{bin_name}4',
                 f'dg_{bin_name}1__kg_{bin_name}4',
                 f'dg_{bin_name}4__kg_{bin_name}1']

    print('set_cross', len(np.unique(set_cross)))

    return set_auto, set_cross


def plot_cls_diff_stack(dict_ps_data, cases, ylim=[0, 1e-5], ylim_frac=[-0.2, 0.2], title='', cmap='tab10'):

    sim_lw=5
    fontsize_label = 16
    fontsize_title = 14
    fontsize_legend = 14
    
    smoothing = lambda x: np.convolve(x, np.ones(50)/50, mode='same')
    trans_ell_cls = lambda ell, cl: (ell.mean(axis=0), np.log10(smoothing(cl.mean(axis=0)) ))
      
    gs_kw = dict(width_ratios=[1.4, 1], height_ratios=[1, 2])
    nx, ny = 2, 1; fig, ax = plt.subplots(nx, ny, 
                                          figsize=(ny * 6, nx * 3), 
                                          squeeze=False, 
                                          gridspec_kw=dict(height_ratios=[2, 1.]),
                                          sharex=True); 
    axl = ax.ravel()
    colors = seaborn.color_palette(cmap, len(cases))
    ylim_frac = [-0.1, 0.1]
    yticks = [-0.05, 0, 0.05]

    for i, k in enumerate(cases):

        ps_data = dict_ps_data[k]

        ell = ps_data['simulation_ell']
        cls_sim =   ps_data['simulation_xcl'] * (hp.nside2npix(2048)/hp.nside2npix(512))**2
        cls_theory = ps_data['theory_xcl']
        
        axl[0].plot(*trans_ell_cls(ell, cls_sim),   '-',  c=colors[i], label=key_to_label(k), lw=sim_lw)
        label = None if i != len(cases)-1 else 'HaloFit'
        axl[0].plot(*trans_ell_cls(ell, cls_theory), '-', c='navy', label=label)
        axl[0].legend(fontsize=fontsize_legend, ncol=1, loc='upper right')

        axl[0].set(ylim=ylim, 
                   xlim=[100, 1400], 
                   yscale='linear')
        axl[0].set_ylabel(r'$log_{10} \ C_\ell$', fontsize=fontsize_label)
#         axl[0].set_title(r'auto power spectra: $[\kappa_g, \delta_g] \times [z_1, z_2, z_3, z_4]$', fontsize=fontsize_title)
        axl[0].set_title(title, fontsize=fontsize_title)
        axl[0].grid(True, ls='--')
        axl[0].set_yticks(axl[0].get_yticks()[1:-1])

        diff_cls = (smoothing(cls_sim.mean(axis=0))-cls_theory.mean(axis=0))/cls_theory.mean(axis=0)
        axl[1].plot(ell.mean(axis=0), diff_cls, label='(cosmogrid-synfast)/synfast', c=colors[i], lw=sim_lw)
        axl[1].axhline(0, c='k')
        axl[1].fill_between([100, 1500], -0.05, 0.05, color='silver', alpha=0.2, lw=0)
        axl[1].set(ylim=ylim_frac, 
                   xlim=[100, 1500], 
                   xlabel=r'$\ell$')
        axl[1].set_ylabel(r'$\Delta C_\ell$', fontsize=fontsize_label)
        axl[1].set_xlabel(r'$\ell$', fontsize=fontsize_label)
        axl[1].set_yticks(yticks)
        axl[1].grid(True, ls='--')
    fig.subplots_adjust(hspace=0)
    return fig

def key_to_label(k):
    p1, p2 = k[:2], k[20:22]
    b1, b2 = int(k[17:18]), int(k[37:38])
    symbol_p = {'kg': '\kappa', 'dg': '\delta'}
    label = r'${}'.format(symbol_p[p1]) 
    label += r'_'
    label += r'{}'.format(b1)
    label += r' \times'
    label += r'{}_'.format(symbol_p[p2]) 
    label += r''
    label += r'{}'.format(b2)
    label += r'$'
    return label