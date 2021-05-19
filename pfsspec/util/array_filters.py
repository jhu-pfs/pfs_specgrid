import numpy as np
from scipy.interpolate import interp1d, interpn
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator, CubicSpline

from pfsspec.data.arraygrid import ArrayGrid

# def get_value_padded(self, name, interpolation='ijk', s=None, fill_holes=False, filter=np.nanmean):
#     """Returns a slice of the grid and pads with a single item in every direction using linearNd extrapolation.

#     Extrapolation is done either in grid coordinates or in axis coordinates

#     Args:
#         name (str): Name of value array
#         s (slice, optional): Slice to apply to value array. Defaults to None.
#         interpolation: Whether to extrapolate based on array indices ('ijk', default)
#             or axis coordinates ('xyz').
#         **kwargs: Values of axis coordinates. Only exact values are supported. For
#             missing direction, full, padded slices will be returned.
#     """

#     # If slicing is turned on, these functions will automatically return the sliced
#     # value array and the sliced (and squeezed) axes.
#     orig_axes = self.get_axes(squeeze=False)
#     orig_value = self.get_value(name, s=s, squeeze=False)

#     if fill_holes and self.has_value_index(name):
#         mask = self.get_value_index(name)
#         orig_value = orig_value.copy()
#         orig_value[~mask] = np.nan
#         orig_value = fill_holes_filter(orig_value, filter=filter)

#     padded_value, padded_axes = pad_array(orig_axes, orig_value, interpolation=interpolation)
            
#     return padded_value, padded_axes

def fill_holes_interpnd(axes, value, mask, interpolation='ijk'):
    # Replace the masked values inside the convex hull with linear interpolation

    orig_xi = ArrayGrid.get_grid_points(axes, padding=False, interpolation=interpolation)

    oijk = []
    for p in orig_xi:
        if orig_xi[p].shape[0] > 1:
            oijk.append(orig_xi[p])

    oijk = np.stack(np.meshgrid(*oijk, indexing='ij'), axis=-1)
    oijk = oijk.reshape((-1, oijk.shape[-1]))

    oval = value.reshape((-1, value.shape[-1]))

    pijk = oijk[~(mask.flatten())]
    oijk = oijk[mask.flatten()]
    oval = oval[mask.flatten()]

    ip = LinearNDInterpolator(oijk, oval)
    fill_value = ip(pijk)

    fill_value = value.copy()
    fill_value = fill_value.reshape((-1, fill_value.shape[-1]))
    fill_value[~(mask.flatten())] = ip(pijk)
    fill_value = fill_value.reshape(value.shape)

    return fill_value

def fill_holes_filter(value, mask_function=np.isnan, fill_filter=np.nanmean, value_filter=np.nanmin):
    # Replaces all nan values by averaging out direct neighbors. Requires
    # array of ndim = 2 or larger. Averaging is done over values in last
    # dimension.
    
    pp = value.copy()

    # Fill in large continuous masked regions
    mask = mask_function(pp)
    # Squeeze array because binary_erosion doesn't work with thin arrays
    mask = ndimage.binary_erosion(mask.squeeze())
    mask = mask.reshape(pp.shape)
    # Fill in large regions using value_filter
    pp[mask] = value_filter(pp)

    last_count = pp.size
    while True:
        mask = mask_function(pp)
        count = np.sum(mask)
        # Stop if all filled in or cannot fill in anything
        if count == 0:
            break
        elif count == last_count:
            break
        last_count = count
        pp[mask] = ndimage.generic_filter(pp, fill_filter, size=3, mode='constant', cval=np.nan)[mask]

    return pp


def pad_array(orig_axes, orig_value, mask=None, size=1, interpolation='ijk'):
    # Depending on the interpolation method, the original axes are converted from
    # actual values to index values. The padded axes will have the original values
    # extrapolated linearly.
    # Mask should be true where valid values are. The padded mask will be the same.

    padded_axes = ArrayGrid.pad_axes(orig_axes, size=size)

    orig_xi = ArrayGrid.get_grid_points(orig_axes, padding=False, interpolation=interpolation)
    padded_xi = ArrayGrid.get_grid_points(padded_axes, padding=True, interpolation=interpolation)

    # Pad original slice with phantom cells
    # We a do a bit of extra work here because we interpolate the entire new slice, not just
    # the edges.
    # TODO: in theory, this method could be used to fill in the holes in the middle but
    #       in practice this doesn't seem to work, so do more tests
    oijk = []
    pijk = []
    padding = []
    padded_shape = []
    for p in orig_xi:
        if orig_xi[p].shape[0] > 1:
            oijk.append(orig_xi[p])
            pijk.append(padded_xi[p])
            padding.append((size, size))
            padded_shape.append(padded_xi[p].shape[0])
        else:
            padding.append((0, 0))
            padded_shape.append(1)

    padding = tuple(padding)
    padded_shape = tuple(padded_shape)

    # TODO: now we interpolate to the entire grid, although it would be
    #       enough to do it for the edges only and use the original values
    #       inside
    pijk = np.stack(np.meshgrid(*pijk, indexing='ij'), axis=-1)

    # fill_value=None : extrapolate
    ip = RegularGridInterpolator(oijk, orig_value, method='linear', bounds_error=False, fill_value=None)
    padded_value = ip(pijk)
    padded_value = np.reshape(padded_value, padded_shape + (padded_value.shape[-1],))
    
    # Fill in the middle from the original
    mm = np.isnan(padded_value)
    padded_value[mm] = np.pad(orig_value, padding + ((0, 0),), mode='constant', constant_values=np.nan)[mm]

    # Create a new mask. The middle should be the original mask and the padded region
    # is masked out if it is _not_ nan
    if mask is not None:
        padded_mask = np.pad(mask, padding, mode='constant', constant_values=False)
        padded_mask[~padded_mask] |= ~np.any(mm, axis=-1)[~padded_mask]
    else:
        padded_mask = ~np.any(mm, axis=-1)

    return padded_value, padded_axes, padded_mask


def anisotropic_diffusion(img, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=1):
    r"""
    Edge-preserving, XD Anisotropic diffusion.


    Parameters
    ----------
    img : array_like
        Input image (will be cast to numpy.float).
    niter : integer
        Number of iterations.
    kappa : integer
        Conduction coefficient, e.g. 20-100. ``kappa`` controls conduction
        as a function of the gradient. If ``kappa`` is low small intensity
        gradients are able to block conduction and hence diffusion across
        steep edges. A large value reduces the influence of intensity gradients
        on conduction.
    gamma : float
        Controls the speed of diffusion. Pick a value :math:`<= .25` for stability.
    voxelspacing : tuple of floats or array_like
        The distance between adjacent pixels in all img.ndim directions
    option : {1, 2, 3}
        Whether to use the Perona Malik diffusion equation No. 1 or No. 2,
        or Tukey's biweight function.
        Equation 1 favours high contrast edges over low contrast ones, while
        equation 2 favours wide regions over smaller ones. See [1]_ for details.
        Equation 3 preserves sharper boundaries than previous formulations and
        improves the automatic stopping of the diffusion. See [2]_ for details.

    Returns
    -------
    anisotropic_diffusion : ndarray
        Diffused image.

    Notes
    -----
    Original MATLAB code by Peter Kovesi,
    School of Computer Science & Software Engineering,
    The University of Western Australia,
    pk @ csse uwa edu au,
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal,
    Department of Pharmacology,
    University of Oxford,
    <alistair.muldal@pharm.ox.ac.uk>

    Adapted to arbitrary dimensionality and added to the MedPy library Oskar Maier,
    Institute for Medical Informatics,
    Universitaet Luebeck,
    <oskar.maier@googlemail.com>

    June 2000  original version. -
    March 2002 corrected diffusion eqn No 2. -
    July 2012 translated to Python -
    August 2013 incorporated into MedPy, arbitrary dimensionality -

    References
    ----------
    .. [1] P. Perona and J. Malik.
       Scale-space and edge detection using ansotropic diffusion.
       IEEE Transactions on Pattern Analysis and Machine Intelligence,
       12(7):629-639, July 1990.
    .. [2] M.J. Black, G. Sapiro, D. Marimont, D. Heeger
       Robust anisotropic diffusion.
       IEEE Transactions on Image Processing,
       7(3):421-432, March 1998.
    """
    # define conduction gradients functions
    if option == 1:
        def condgradient(delta, spacing):
            return np.exp(-(delta/kappa)**2.)/float(spacing)
    elif option == 2:
        def condgradient(delta, spacing):
            return 1./(1.+(delta/kappa)**2.)/float(spacing)
    elif option == 3:
        kappa_s = kappa * (2**0.5)

        def condgradient(delta, spacing):
            top = 0.5*((1.-(delta/kappa_s)**2.)**2.)/float(spacing)
            return np.where(np.abs(delta) <= kappa_s, top, 0)

    # initialize output array
    out = np.array(img, dtype=np.float32, copy=True)

    # set default voxel spacing if not supplied
    if voxelspacing is None:
        voxelspacing = tuple([1.] * img.ndim)

    # initialize some internal variables
    deltas = [np.zeros_like(out) for _ in range(out.ndim)]

    for _ in range(niter):

        # calculate the diffs
        for i in range(out.ndim):
            slicer = tuple([slice(None, -1) if j == i else slice(None) for j in range(out.ndim)])
            deltas[i][slicer] = np.diff(out, axis=i)

        # update matrices
        matrices = [condgradient(delta, spacing) * delta for delta, spacing in zip(deltas, voxelspacing)]

        # subtract a copy that has been shifted ('Up/North/West' in 3D case) by one
        # pixel. Don't as questions. just do it. trust me.
        for i in range(out.ndim):
            slicer = tuple([slice(1, None) if j == i else slice(None) for j in range(out.ndim)])
            matrices[i][slicer] = np.diff(matrices[i], axis=i)

        # update the image
        out += gamma * (np.sum(matrices, axis=0))

    return out