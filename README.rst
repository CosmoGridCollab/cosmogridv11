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


Installation
------------

For now, please use:

`python setup.py install`



Usage
-----

There is a number of apps that can be used to accomplish consecutive stages of processing, from baryonification to projected probe maps.

1. Baryonification      
   - Run halocone creation and halo profiling:  `cosmogridv1/apps/run_baryonification.py profile_halos`     
   - Displace shells using the halocone: `/cosmogridv1/apps/run_baryonification.py displace_shells`     

2. Create projected probe maps      
   - Create shell permutation table, including perturbations to n(z): `cosmogridv1/apps/run_create_permutation_table.py`      
   - Create probe kernels: `cosmogridv1/apps/run_create_probe_kernels.py`      
   - Create projected probe maps: `cosmogridv1/apps/run_project_shells.py`      





Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage