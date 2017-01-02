import numpy as np
from skimage.exposure import equalize_hist
from skimage.filters.rank import median
from skimage.measure import regionprops
from skimage.morphology import disk
from skimage.segmentation import felzenszwalb
from skimage.transform import rescale
from scipy.ndimage import binary_fill_holes
from scipy.misc import imresize


def breast_segment(im, scale_factor=0.25, threshold=3900, felzenzwalb_scale=0.15):
    """
    Fully automated breast segmentation in mammographies.
    https://github.com/olieidel/breast_segment

    :param im: Image
    :param scale_factor: Scale Factor
    :param threshold: Threshold
    :param felzenzwalb_scale: Felzenzwalb Scale

    :return: (im_mask, bbox) where im_mask is the segmentation mask and
    bbox is the bounding box (rectangular) of the segmentation.

    """

    # set threshold to remove artifacts around edges
    im_thres = im.copy()
    im_thres[im_thres > threshold] = 0

    # determine breast side
    col_sums_split = np.array_split(np.sum(im_thres, axis=0), 2)
    left_col_sum = np.sum(col_sums_split[0])
    right_col_sum = np.sum(col_sums_split[1])

    if left_col_sum > right_col_sum:
        breast_side = 'l'
    else:
        breast_side = 'r'

    # rescale and filter aggressively, normalize
    im_small = rescale(im_thres, scale_factor)
    im_small_filt = median(im_small, disk(50))
    # this might not be helping, actually sometimes it is
    im_small_filt = equalize_hist(im_small_filt)

    # run mr. felzenzwalb
    segments = felzenszwalb(im_small_filt, scale=felzenzwalb_scale)
    segments += 1  # otherwise, labels() would ignore segment with segment=0


    props = regionprops(segments)

    # Sort Props by area, descending
    props_sorted = sorted(props, key=lambda x: x.area, reverse=True)

    expected_bg_index = 0
    bg_index = expected_bg_index

    bg_region = props_sorted[bg_index]
    minr, minc, maxr, maxc = bg_region.bbox
    filled_mask = bg_region.filled_image

    im_small_fill = np.zeros((im_small_filt.shape[0]+2, im_small_filt.shape[1]+1), dtype=int)

    if breast_side == 'l':
        # breast expected to be on left side,
        # pad on right and bottom side
        im_small_fill[minr+1:maxr+1, minc:maxc] = filled_mask
        im_small_fill[0, :] = 1  # top
        im_small_fill[-1, :] = 1  # bottom
        im_small_fill[:, -1] = 1  # right
    elif breast_side == 'r':
        # breast expected to be on right side,
        # pad on left and bottom side
        im_small_fill[minr+1:maxr+1, minc+1:maxc+1] = filled_mask  # shift mask to right side
        im_small_fill[0, :] = 1  # top
        im_small_fill[-1, :] = 1  # bottom
        im_small_fill[:, 0] = 1  # left

    im_small_fill = binary_fill_holes(im_small_fill)

    im_small_mask = im_small_fill[1:-1, :-1] if breast_side == 'l' \
                  else im_small_fill[1:-1, 1:]

    # rescale mask
    im_mask = imresize(im_small_mask, im.shape).astype(bool)

    # invert!
    im_mask = ~im_mask

    # determine side of breast in mask and compare
    col_sums_split = np.array_split(np.sum(im_mask, axis=0), 2)
    left_col_sum = np.sum(col_sums_split[0])
    right_col_sum = np.sum(col_sums_split[1])

    if left_col_sum > right_col_sum:
        breast_side_mask = 'l'
    else:
        breast_side_mask = 'r'

    if breast_side_mask != breast_side:
        # breast mask is not on expected side
        # we might have segmented bg instead of breast
        # so invert again
        print('breast and mask side mismatch. inverting!')
        im_mask = ~im_mask

    # exclude thresholded area (artifacts) in mask, too
    im_mask[im > threshold] = False

    # fill holes again, just in case there was a high-intensity region
    # in the breast
    im_mask = binary_fill_holes(im_mask)

    # if no region found, abort early and return mask of complete image
    if im_mask.ravel().sum() == 0:
        all_mask = np.ones_like(im).astype(bool)
        bbox = (0, 0, im.shape[0], im.shape[1])
        print('Couldn\'t find any segment')
        return all_mask, bbox

    # get bbox
    minr = np.argwhere(im_mask.any(axis=1)).ravel()[0]
    maxr = np.argwhere(im_mask.any(axis=1)).ravel()[-1]
    minc = np.argwhere(im_mask.any(axis=0)).ravel()[0]
    maxc = np.argwhere(im_mask.any(axis=0)).ravel()[-1]

    bbox = (minr, minc, maxr, maxc)

    return im_mask, bbox
