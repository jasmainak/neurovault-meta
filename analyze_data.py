import scipy

import pandas as pd
import nibabel as nib

from nilearn import datasets
from nilearn.datasets import fetch_neurovault_ids
from nilearn import plotting
from nilearn.image import new_img_like, load_img, math_img

# These ids were determined by querying neurovault like this:

# from nilearn.datasets import fetch_neurovault, neurovault

# nv_data = fetch_neurovault(
#     max_images=7,
#     cognitive_paradigm_cogatlas=neurovault.Contains('stop signal'),
#     contrast_definition=neurovault.Contains('succ', 'stop', 'go'),
#     map_type='T map')

# print([meta['id'] for meta in nv_data['images_meta']])

# 1. feature extraction / metadata
# 2. dimensionality reduction

nv_data = datasets.fetch_neurovault(max_images=None, mode='offline')

images = nv_data['images']
images_meta = nv_data['images_meta']
collections = nv_data['collections_meta']

img = nib.load(images[0])
data = img.get_data()

metadata = pd.DataFrame(images_meta)
metadata.to_csv('metadata.csv', encoding='utf-8')
