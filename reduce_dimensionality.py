import pandas as pd
import nibabel as nib

from nilearn import datasets
from nilearn.input_data import NiftiMapsMasker

from utils import read_resampled_img

data_dir = '/home/mainak/Desktop/neurovault/'

# TODOs
#
# 1. feature extraction / metadata
# 2. t maps to z maps etc.
# 3. weighting of different collections
#    (some are subject level, some are group level)
# 4. weighting features -- e.g., cog atlas terms
# 5. dirty_cat for labels using ngram similarities (SimilarityEncoder)
# 6. Weighting based on collections
# 7. Distribution of #images per collection
# 8. Average over all images?
# 9. Nearest neighbor to go from images to labels
# 10. Get encoder maps for the most frequently occuring words
# 11. Normalize by frequency of different terms in baseline
# 12. Precompute baseline

# get data / list of files
nv_data = datasets.fetch_neurovault(max_images=None, mode='offline',
                                    data_dir=data_dir)

images = nv_data['images']
images_meta = nv_data['images_meta']
collections = nv_data['collections_meta']

# export metadata to pandas
metadata = pd.DataFrame(images_meta)

# read Arthur Mensch's parcellations
maps = nib.load('components_512.nii.gz')

# reduce dimensionality
imgs = []
for ii, image in enumerate(images):
    print('Resampling image %d' % ii)
    imgs.append(read_resampled_img(image))

masker = NiftiMapsMasker(maps)
X = masker.fit_transform(imgs)
