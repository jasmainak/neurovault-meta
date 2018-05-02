from nilearn import datasets

# The code below is to include thresholded images.

# With non thresholded images it is simpler

from nilearn.datasets import neurovault

img_terms = neurovault.basic_image_terms().copy()

img_terms = neurovault.basic_image_terms()

del img_terms['is_thresholded']

d = datasets.fetch_neurovault(max_images=None, image_terms=img_terms)
