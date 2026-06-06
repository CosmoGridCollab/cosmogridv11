.. highlight:: shell

============
Installation
============


Stable release
--------------

To install CosmoGridV11, run this command in your terminal:

.. code-block:: console

    $ pip install cosmogridv11

This is the preferred method to install CosmoGridV11, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for CosmoGridV11 can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/tomaszkacprzak/cosmogridv11

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/tomaszkacprzak/cosmogridv11/tarball/master

Once you have a copy of the source, install it with a modern Python package installer:

.. code-block:: console

    $ uv pip install .

For editable development installs, use:

.. code-block:: console

    $ uv pip install -e .

You can also install from the local checkout with ``pip``:

.. code-block:: console

    $ python -m pip install .

The legacy ``python setup.py install`` workflow is no longer used.


.. _Github repo: https://github.com/tomaszkacprzak/cosmogridv11
.. _tarball: https://github.com/tomaszkacprzak/cosmogridv11/tarball/master
