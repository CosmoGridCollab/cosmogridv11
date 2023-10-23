# CosmoGridV11

Codes for CosmoGridV1.1 simulations.

* Free software: GNU General Public License v3
* Documentation: www.cosmogrid.ai 
* Paper: Kacprzak et al. 2023 https://arxiv.org/abs/2209.04662


## Installation

For now, please use:

``python setup.py install``



## Usage


There is a number of apps that can be used to accomplish consecutive stages of processing, from baryonification to projected probe maps.

1)  Baryonification      

    1.1 Run halocone creation and halo profiling:      
    ``python cosmogridv1/apps/run_baryonification.py profile_halos --help``     

    1.2 Displace shells using the halocone:      
    ``python cosmogridv1/apps/run_baryonification.py displace_shells --help``     

3)  Create projected probe maps      

    2.1 Create shell permutation table, including perturbations to n(z):      
    ``python cosmogridv1/apps/run_create_permutation_table.py --help``      

    2.2 Create probe kernels:      
    ``python cosmogridv1/apps/run_create_probe_kernels.py --help``      

    2.3 Create projected probe maps:      
     ``python cosmogridv1/apps/run_project_shells.py --help``      

The output files will contain full-sky probe maps, without noise, masks, and systematics.
The intrinsic alignment map can be added to the convergence map with an appropriate amplitude.
Galaxy overdensity maps can be multiplied by a bias parameter.
Downstream processig for applying survey masks, noise, systematics, etc, is not part of this pipeline and left for the user.
For usage of each app, use the help command, example ``cosmogridv1/apps/run_baryonification.py --help``.




## CosmoGridV11: LSST-DESC Y1 Trial


Probe maps that can be used for making forecasts for LSST-DESC-Y1 are stored in `CosmoGrid/lsstdescy1trial` and condain: full sky projected weak lensing, intrinsic alignment, and galaxy clustering maps at nside=1024 for a LSST Y1.
The redshift bins used are [here](https://github.com/LSSTDESC/forecasting/tree/main/updated_forecasts/datafiles/z_bins).
This data is described in [Kacprzak et al. 2022]https://arxiv.org/abs/2209.04662).

The file structure is 

`CosmoGrid/lsstdescy1trial/dataset_type/cosmology/realization/`

where `dataset_type=[fiducial, grid]` and `realization=perm_*` is the semi-independent realization. In the `lsstdescy1trial`, there are 10 realizations for `fiducial` and 2 realizations for `grid`.

In each of these directories, the files are:

| file name     | file content  | comments      |
| ------------- | ------------- | ------------- |
| `realization/projected_probes_maps_v11dmb.h5`   | HDF5 store with baryonified probe maps for 4 redshift bins for lensing, clustering and intrinsic alignment probes | the HDF5 file has the following structure: `probe/sample`|                            
| `realization/projected_probes_maps_v11dmo.h5`     | Same as above, but with no baryonification | same as above |                          
| `realization/shell_permutations_index.h5`               | HDF5 store with information about the shell selection for the shell permutation scheme | contains datsets:  <br /> `shell_groups`: list of shell groups taken from different simulations   <br /> `perms_info`: information which simulation to use for each shell group and whether to apply rotations or flips (see below for description of this table)|                
| `realization/probe_weights.h5`                             | HDF store with probe projection kernels, single value for shell mean redshift | datasets are organized as `probe/sample` | 



Additional notes:

* The lensing map ``kg`` is noise free. To use it, subtract the mean and add shape noise.
* The intrinsic alignment map ``ia`` is an NLA model with $A_{IA}=1$. To use it, scale according to desired $A_{IA}$ and add to the lensing convergence map.
* The clustering map ``dg`` is an overdensity map $n_g = (\delta - \bar \delta)/(\bar \delta)$. To use it, multiply by a bias function and add Poisson noise $\delta_g = Poisson( \bar N(1+b \delta_g) ) $

TODO:
--------

* Create website for API documentation 
* Add description of the config file
* Add description of output files
* Add description of the permutation table


