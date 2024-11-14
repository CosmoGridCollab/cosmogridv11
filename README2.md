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

    1.1 Create parameters for baryonification:
    ``python cosmogridv1/apps/run_paramtables.py baryon_params --help``     

    1.2 Run halocone creation and halo profiling:      
    ``python cosmogridv1/apps/run_haloops.py profile_halos --help``     

    1.3 Displace shells using the halocone:      
    ``python cosmogridv1/apps/run_haloops.py displace_shells --help``     

3)  Create projected probe maps      

    2.1 Create shell permutation table, including perturbations to n(z):      
    ``python cosmogridv1/apps/run_paramtables.py shell_permutations --help``      

    2.3 Create projected probe maps:      
     ``python cosmogridv1/apps/run_project_shells.py --help``      

The output files will contain full-sky probe maps, without noise, masks, and systematics.
The intrinsic alignment map can be added to the convergence map with an appropriate amplitude.
Galaxy overdensity maps can be multiplied by a bias parameter.
Downstream processig for applying survey masks, noise, systematics, etc, is not part of this pipeline and left for the user.
For usage of each app, use the help command, example ``cosmogridv1/apps/run_baryonification.py --help``.


## CosmoGridV11: baryonification output files

## CosmoGridV11: LSST-DESC Y1 Trial


TODO:
--------

* New baryonification parameters interface
* Re-run with new codes
* Write docs for new codes
