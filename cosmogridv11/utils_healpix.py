import healpy as hp
import numpy as np

def mirror_pix(pix, nside, lr=False):
    """
    mirror a patch

    :param pix: patch pixel indices
    :param nside: healpy nside
    :param lr: left to right (instead of up down) mirroring
    """
    theta, phi = hp.pix2ang(ipix=pix, nside=nside)
    if lr:
        phi = 2*np.pi - phi
    else:
        theta = np.pi - theta
    new_pix = hp.ang2pix(theta=theta, phi=phi, nside=nside)

    # we make sure that no pixel appears twice that should not
    assert len(set(new_pix)) == len(set(pix))

    return new_pix

def rotate_pix(pix, nside, n_rot=1):
    """
    rotate a patch by 90 degrees
    
    :param pix: patch pixel indices
    :param n_rot: number of rotations
    :param nside: healpy nside
    """
    theta, phi = hp.pix2ang(ipix=pix, nside=nside)
    phi = (phi + n_rot*np.pi/2) % (2*np.pi)
    new_pix = hp.ang2pix(theta=theta, phi=phi, nside=nside)

    # we make sure that no pixel appears twice that should not
    assert len(set(new_pix)) == len(set(pix))

    return new_pix