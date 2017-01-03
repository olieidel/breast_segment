# breast_segment

### Fully automated Breast Segmentation on Mammographies

But wait, let's look at some pictures first!

<img src="https://github.com/olieidel/breast_segment/raw/master/images/jpeg/original.jpg" width="24%" alt="Original Image" />
<img src="https://github.com/olieidel/breast_segment/raw/master/images/jpeg/mask.jpg" width="24%" alt="Computed Segmentation Mask" />
<img src="https://github.com/olieidel/breast_segment/raw/master/images/jpeg/bbox.jpg" width="24%" alt="Computed Bounding Box" />
<img src="https://github.com/olieidel/breast_segment/raw/master/images/jpeg/overlay.jpg" width="24%" alt="Overlay Visualization" />

`breast_segment` takes a Mammography image and detects the largest, connected region (usually the breast).
It outputs a segmentation mask and the coordinates of the rectangular bounding box.

Now, the images are easily explained:
- First image: Input Image
- Second image: Segmentation mask (computed, boolean)
- Third image: Bounding Box (computed)
- Fourth image: Overlay Visalization of the Segmentation

### Usage

`breast_segment` should be available on your python path.
You could achieve this like so:
```python
import sys
sys.path.append('PATH_TO_BREAST_SEGMENT')
```
Then, import and use. Let's say that we've loaded an image into the variable `im`.
It should be a NumPy Array.
```python
from breast_segment import breast_segment
mask, bbox = breast_segment(im)
```
Done! You may want to visualize it to check if it worked. Let's
use [matplotlib]:
```python
from matplotlib import pyplot as plt
%matplotlib inline
```
Visualize the original image (you probably know how to do that):
```python
f, ax = plt.subplots(1, figsize=(12, 12)) # adjust the figure size
# set the correct window and color map. yours may differ.
# radiologists like gray, not understandably
ax.imshow(im, vmin=0, vmax=4096, cmap='gray')
```
Have a look at the segmentation mask:
```python
f, ax = plt.subplots(1, figsize=(12, 12))
ax.imshow(mask, vmin=0, vmax=1, cmap='inferno') # use inferno colormap for dramatisation
```
Create an overlay visualization like in the fourth image above:
```python
f = plt.figure(figsize=(12, 12))
ax = plt.subplot(111)
ax.imshow(im, vmin=0, vmax=4096, cmap='gray')
ax.imshow(mask, alpha=.3, cmap='inferno') # alpha controls the transparency
```
Done!

Showing the bounding box like in the third image above is a little more code but
also easy. Check out the complete example (see below) for the full story!

If you want to tweak some parameters, look no further:

```python
mask, bbox = breast_segment(image, scale_factor=0.25, threshold=3900, felzenzwalb_scale=0.15)
```

**Input Parameters**

| Parameter | Type | Explanation |
| --------- | ---- | ----------- |
| image     | NumPy Array, two-dimensional | The Input Mammography to segment. |
| scale_factor | float | Downscaling factor. The segmentation will be computed on this smaller image. Default: 0.25 (25% of original size) |
| threshold | integer | The maximum cut-off for noisy (scanned) Mammography images. Values above are treated as noise and ignored. Default: 3900 (values > 3900 will be set to 0) |
| felzenzwalb_scale | float | Scale Parameter for Felzenzwalb's Algorithm. Check out the [respctive skimage docs] for more information. Default: 0.15 (tested on DDSM) |

**Output Parameters**

| Parameter | Type | Explanation |
| --------- | ---- | ----------- |
| mask      | NumPy Array, two-dimensional, boolean | The computed segmentation mask. `true` (1) values for breast, `false` for everything else. |
| bbox      | 4-element Tuple | The coordinates of the rectangular bounding box of the segmentation. The tuple consists of `(min_row, min_col, max_row, max_col)` where min_row is the starting row and max_row the ending row of the bounding box. Same for min_col and max_col, only in columns (you guessed it). |

### Complete Example

The above images were generated in an IPython Notebook which resides `examples/` ([ddsm_examples.ipynb]).
Check out the full notebook for a complete walkthrough!

The Mammography image is from the [Digital Database for Screening Mammography] (DDSM).
To my knowledge, it's one of the biggest public mammography databases. However, you will
need some more tools to open and process the weird `LJPEG` format and their associated
metadata, which I conveniently provide:

- [ljpeg3] for opening the LJPEGs
- [ddsmtools] for processing of the DDSM Metadata

### Details

`breast_segment` is basically a simple application of the [Felzenzwalb Algorithm in skimage]
with some tweaking of parameters and thresholding for noise-reduction. Additionally,
Felzenzwalb's Algorithm is only applied to a downscaled version of the image to reduce
computation time.

### Edge cases and Fails

Sometimes, `breast_segment` wrongly detects the background as "breast" and creates an inverted
segmentation. There is a slightly hacky mechanism to detect that, in which case it simply selects
the second largest region (instead of the largest) as breast.

In other cases, it simply fails dramatically. Tweaking the threshold and Felzenzwalb Scale to
some optimum values for the given Mammography images, dependent e.g. on the type of scanner,
can help in some cases.

### Background

I had been trying to apply Neural Networks to Mammographies by training them to differentiate
between bening and malign images of the DDSM database (spoiler: it didn't work). In trying to
improve the preprocessing (reducing the noise), I developed this automatic segmentation.
Feel free to [contact me] if you want to hear the full story!

As I'm currently moving away from Machine Learning, I decided to open-source it to spare you
some time and nerves. Good luck!

### Questions?

Feel free to file an issue on this Repo or contact me ([Oliver Eidel]), I'm happy to help!

## License

MIT

<!-- Internal Links -->
[ddsm_examples.ipynb]: https://github.com/olieidel/breast_segment/blob/master/examples/ddsm_examples.ipynb

<!-- Github Links -->
[Oliver Eidel]: http://www.eidel.io
[@olieidel]: http://www.eidel.io

<!-- Repos -->
[ljpeg3]: https://github.com/olieidel/ljpeg3
[ddsmtools]: https://github.com/olieidel/ddsmtools

<!-- Other dependencies -->
[matplotlib]: http://matplotlib.org/users/installing.html

<!-- General -->
[Digital Database for Screening Mammography]: http://marathon.csee.usf.edu/Mammography/Database.html
[DDSM]: http://marathon.csee.usf.edu/Mammography/Database.html
[respctive skimage docs]: http://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb
[Felzenzwalb Algorithm in skimage]: http://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb
