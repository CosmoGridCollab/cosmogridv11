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

    1.1 Create a table that contains input parameters for baryonification, specified in `config_v11.yaml`:      
    ``python cosmogridv1.apps.run_paramtables baryon_params --config=config_v11.yaml --help``
    The resulting output file will be `CosmoGridV11_bary_{tag}_metainfo.h5`, where `{tag}` is the entry in the config under `baryonification/tag:`.
    After the file is created, make sure to include it in the config under `paths/metainfo_bary:`, for later use.

    1.2 Displace shells using the halocone:      
    ``python cosmogridv1.apps.run_haloops baryonify_shells  --config=config_v11.yaml  --help``     

3)  Create projected probe maps      

    2.1 Create shell permutation table, including perturbations to n(z), according to the config:      
    ``python cosmogridv1.apps.run_paramtables shell_permutations --config=config_v11desy3.yaml``
    The resulting output file will be `metainfo_perms.npy`, in the output directory.
    After the file is created, make sure to include it in the config under `paths/redshift_perturbation_list:`, for later use.
    
    2.2 Create projected probe maps:      
     ``python cosmogridv1.apps.run_probemaps --config=config_v11desy3.yaml --help``      

The output files will contain full-sky probe maps, without noise, masks, and systematics.
The intrinsic alignment map can be added to the convergence map with an appropriate amplitude.
Galaxy overdensity maps can be multiplied by a bias parameter.
Downstream processig for applying survey masks, noise, systematics, etc, is not part of this pipeline and left for the user.


## CosmoGridV11: output files


### The baronification parameter table, `CosmoGridV1_bary_{tag}_metainfo.h5`

This table is extending the `CosmoGridV1_metainfo.h5` table to include baryonified simulations.
See documentation on http://www.cosmogrid.ai/data_docs/.
The contents and structure are similar.

| dataset | content |
| ------------- | ------------- | 
| `parameters/all` | baryonification parameters for all unique parameter sets, extending the cosmology metainfo parameters, number of rows corresponds to the number of *baryonified* parameter sets. Datasets `grid`, `fiducial` and `benchmark` contain the same but split into dataset types. | 
| `simulations/all` | baryonificatied simulations for all unique simulations sets, extending the cosmology metainfo parameters, number of rows corresponds to the number of *baryonified* simulations. Same as above for other types. |
| `shell_info` | shell info for *baryonified* parameters, matching the paths in `parameters`.  |


### The profiled halos catalog, `profiled_halos_v11.h5`

The halo catalog is created using snapshots, and then replicated to cover the entire lightcone. 
Therefore some halos occur at multiple locations on the sky.

| dataset | content |
| ------------- | ------------- | 
| `shell{:03d}/halo_pos` | 3D positions of halos, number of rows corresponds to the total number of halos in the lightcone. Match with `halo_props` with `uid`. | 
| `shell{:03d}/halo_props` | 3D positions of halos, number of rows corresponds to the total number of *unique* halos in the lightcone, no replication. Match with `halo_pos` by `ID`. | 


#### Content of `halo_pos`

| field | content |
| ------------- | ------------- | 
| `uid` | Unique halo id |
| `x`, `y`, `z` | 3D position in Mpc/h |
| `shell_id` | Id of the shell the halo belongs to |
| `halo_buffer` | If it is a halo belonging to a neighbour shell, +1/-1 for halo from a lower and higher redshift shells, respectively |

#### Content of `halo_props`

| field | content |
| ------------- | ------------- | 
| `ID` | Unique halo id |
| `m_200c` | NFW mass in 200c units M_solar/h |
| `r_200c` | NFW radius in 200c units Mpc/h |
| `c_200c` | NFW concentration |



### The baryonified shell shell files

These files contain both the unbaryonified and baryonified maps.
See function `utils_maps.load_v11_shells` for instructions how to load this file and get the baryonified map.

| dataset | content |
| ------------- | ------------- | 
| `nobaryon_shells/shell{:03d}` | unbaryonified shells at the specified baryonification nside |
| `diff_shell_inds/shell{:03d}` | indices of pixels that are different in the baryonified shells  |
| `diff_shell_vals/shell{:03d}` | values of pixels that are different in the baryonified shells  |


## The projected maps files

Files in `simulation_set/cosmology/permutation/projected_probes_maps_v11dmb.h5` contain the projected maps. Here the tag `v11dmb` was used.
The power spectra for all probes and samples are measured (`cell_ccl`). 
PyCCL theory was calculated (`cell_ccl`).
Additionally, a Synfast map was created with the theory Cell and its power spectrum was measured (`cell_map_ccl`).
This allows to compare the Cell CosmoGrid with PyCCL accounting for the pixelization.

| dataset | content |
| ------------- | ------------- | 
| `kernel/probe/sample` | Projection kernel for a given probe and sample. Array of size Nx2, where N is the number of entries in the n(z) provided in config. Columns are `z`, `kernel value`. |
| `map/probe/sample` | Projected map for a given probe and sample. Vector corresponding to the given nside.  |
| `nz/sample` | Redshift distribution for each sample, same array dimensions as in the kernel. |
| `perms_info` | Permutation specifications for these maps. See http://www.cosmogrid.ai/data_docs/. |
| `shell_groups/{i:d}` | A list of shells that created each group `i`, containing the rows from `shell_info`. |
| `cell/probe/sample` | Power spectra of the maps. The columns are: `ell`, `cell_ccl`, `cell_map_ccl`, `cell_map`. See description above. |




## CosmoGridV11: LSST-DESC Y1 Trial


Probe maps that can be used for making forecasts for LSST-DESC-Y1 are stored in `CosmoGrid/lsstdescy1trial` and condain: full sky projected weak lensing, intrinsic alignment, and galaxy clustering maps at nside=1024 for a LSST Y1.
The redshift bins used are [here](https://github.com/LSSTDESC/forecasting/tree/main/updated_forecasts/datafiles/z_bins).
This data is described in [Kacprzak et al. 2022](https://arxiv.org/abs/2209.04662).

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


