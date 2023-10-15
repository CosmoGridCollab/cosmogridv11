#! /usr/bin/env python

# Copyright (C) 2022 ETH Zurich,
# Institute for Particle Physics and Astrophysics
# Author: Silvan Fischbacher & Tomasz Kacprzak
# From: https://cosmo-gitlab.phys.ethz.ch/cosmo_public/redshift_tools/-/blob/master/src/redshift_tools/manipulate.py

import numpy as np
from scipy.optimize import bisect
from scipy.interpolate import interp1d


def resample(xp, x, y, interp_kind="linear"):
    """
    Resample a distribution with a given set of bins.

    :param xp: new bins
    :param x: old bins
    :param y: old distribution
    :param interp_kind: interpolation method
    :return: resampled distribution
    """
    yp = interp1d(x=xp, y=y, kind=interp_kind, fill_value="extrapolate")(x)
    yp = np.clip(yp, a_min=0, a_max=np.inf)
    yp = yp / np.sum(yp)
    return yp


def stretch_bin(z, nz, s):
    """
    Stretch a redshift distribution by a given factor.

    :param z: redshift array
    :param nz: redshift distribution n(z)
    :param s: stretch factor
    :return: stretched redshift distribution n_s(z)
    """
    if s == 1:
        return nz
    nz = nz / np.sum(nz)
    mean_z = np.sum(z * nz)
    z_prime = z - mean_z
    z_prime = z_prime / s

    def func_bisect(delta_z):
        nz_prime = resample(xp=z, x=z_prime + mean_z + delta_z, y=nz)
        mean_z_prime = np.sum(nz_prime * z)
        return mean_z_prime - mean_z

    delta_z = bisect(f=func_bisect, a=-0.1, b=0.1)
    nz_prime = resample(xp=z, x=z_prime + mean_z + delta_z, y=nz)
    return nz_prime


def stretch(NZ, sigma_z, normalize=True):
    """
    Stretch a set of redshift distributions by a given set of stretch factors.

    :param NZ: list of redshift distributions
    :param sigma_z: list of stretch factors
    :param normalize: normalize the stretched distributions
    :return: list of stretched redshift distributions
    """
    NZ_new = []
    for i, nz in enumerate(NZ):
        nz_new = nz.copy()
        nz_new[:, 1] = stretch_bin(nz[:, 0], nz[:, 1], sigma_z[i])
        NZ_new.append(nz_new)
    if normalize:
        NZ_new = normalize_bins(NZ_new)
    return NZ_new


def overlap(bin1, bin2):
    """
    Compute the overlap between two redshift distributions.

    :param bin1: first redshift distribution
    :param bin2: second redshift distribution
    :return: overlap
    """
    z1 = bin1[:, 0]
    z2 = bin2[:, 0]
    z_interp = np.linspace(min(min(z1), min(z2)), max(max(z1), max(z2)), 1000)
    nz1 = np.interp(z_interp, z1, bin1[:, 1])
    nz2 = np.interp(z_interp, z2, bin2[:, 1])
    norm1 = np.trapz(nz1, z_interp)
    norm2 = np.trapz(nz2, z_interp)
    nz1 /= norm1
    nz2 /= norm2
    return np.trapz(np.minimum(nz1, nz2), z_interp)


def overlap_all_bins(nz):
    """
    Compute the overlap between all redshift distributions.

    :param nz: list of redshift distributions
    :return: overlap matrix
    """
    o = []
    for i in range(len(nz)):
        for j in range(i, len(nz)):
            o.append(overlap(nz[i], nz[j]))
    return np.array(o)


def shift(nz_fid, delta_z):
    """
    Shift a set of redshift distributions by a given set of shift factors.

    :param nz_fid: list of redshift distributions
    :param delta_z: list of shift factors
    :return: list of shifted redshift distributions
    """
    nz_new = []
    for i, d in enumerate(delta_z):
        nz_i = np.zeros_like(nz_fid[i])
        z = nz_fid[i][:, 0]
        n = nz_fid[i][:, 1]

        z_new = z + d
        n_new = np.interp(z, z_new, n)
        nz_i[:, 0] = nz_fid[i][:, 0]
        nz_i[:, 1] = n_new
        nz_new.append(nz_i)
    return nz_new


def compute_n_tot(nz):
    """
    Compute the total redshift distribution from a set of redshift distributions.

    :param nz: list of redshift distributions
    :return: total redshift distribution
    """
    n = nz[0][:, 1].copy()
    for i in range(1, len(nz)):
        n += nz[i][:, 1]
    return n


def normalize_bins(nz):
    """
    Normalize a set of redshift distributions.

    :param nz: list of redshift distributions
    :return: list of normalized redshift distributions
    """
    n_tot = compute_n_tot(nz)
    z = nz[0][:, 0]
    norm = np.trapz(n_tot, z)
    for NZ in nz:
        NZ[:, 1] /= norm
    return nz

def taper(nz, factor=10):

    for nz_ in nz:
        nz_[:,1] *= np.tanh(nz_[:,0]*factor)

    return nz


def shift_and_stretch(nz, z_bias, z_sigma, normalize=True):
    """
    Shift and stretch a set of redshift distributions.
    Add tapering to make n(z) go to 0 at z=0.

    :param nz: list of redshift distributions
    :param z_bias: list of shift factors
    :param z_sigma: list of stretch factors
    :return: list of shifted and stretched redshift distributions
    """
    nz_stretched = stretch(nz, z_sigma, normalize)
    nz_stretched_shifted = shift(nz_stretched, z_bias)
    nz_stretched_shifted_tapered = taper(nz_stretched_shifted)
    return nz_stretched_shifted_tapered

def sample_redshift_perturb_params(offset=0, n_samples=1, mu=[0,0], sigma=[0.01, 0.02]):

    import chaospy

    rule='halton'

    assert len(mu)==len(sigma), "mu and sigma should have the same length"

    if len(mu) == 1:

        samples = np.atleast_2d(chaospy.TruncNormal(mu=mu[0], sigma=sigma[0]).sample(offset+n_samples, rule=rule)).T
    
    else:

        norm = [chaospy.TruncNormal(mu=mu[i], sigma=sigma[i], lower=-3*sigma[i], upper=3*sigma[i]) for i in range(len(mu))]
        samples = np.atleast_2d(chaospy.J(*norm).sample(offset+i, rule=rule).T)

    return samples[[-1]]
    
