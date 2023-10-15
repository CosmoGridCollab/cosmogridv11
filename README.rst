============
CosmoGridV11
============

.. image:: https://img.shields.io/pypi/v/cosmogridv11.svg
        :target: https://pypi.python.org/pypi/cosmogridv11

.. image:: https://img.shields.io/travis/tomaszkacprzak/cosmogridv11.svg
        :target: https://travis-ci.com/tomaszkacprzak/cosmogridv11

.. image:: https://readthedocs.org/projects/cosmogridv11/badge/?version=latest
        :target: https://cosmogridv11.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

 


Codes for CosmoGridV1.1 simulations.

* Free software: GNU General Public License v3
* Documentation: www.cosmogrid.ai 
* Paper: Kacprzak et al. 2023 https://arxiv.org/abs/2209.04662


Installation
------------

For now, please use:

``python setup.py install``



Usage
-----

There is a number of apps that can be used to accomplish consecutive stages of processing, from baryonification to projected probe maps.

1)  Baryonification      

    1.1 Run halocone creation and halo profiling:  ``cosmogridv1/apps/run_baryonification.py profile_halos``     

    1.2 Displace shells using the halocone: ``cosmogridv1/apps/run_baryonification.py displace_shells``     

2)  Create projected probe maps      

    2.1 Create shell permutation table, including perturbations to n(z): ``cosmogridv1/apps/run_create_permutation_table.py``      

    2.2 Create probe kernels: ``cosmogridv1/apps/run_create_probe_kernels.py``      

    2.3 Create projected probe maps: ``cosmogridv1/apps/run_project_shells.py``      

The output files will contain full-sky probe maps, without noise, masks, and systematics.
The intrinsic alignment map can be added to the convergence map with an appropriate amplitude.
Galaxy overdensity maps can be multiplied by a bias parameter.
Downstream processig for applying survey masks, noise, systematics, etc, is not part of this pipeline and left for the user.
For usage of each app, use the help command, example ``cosmogridv1/apps/run_baryonification.py --help``.





Features
--------

* Create website for API documentation 
* Add description of the config file
* Add description of output files
* Add description of the permutation table


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
