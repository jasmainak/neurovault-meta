import pandas as pd
import nibabel as nib

from nilearn import datasets
from nilearn.input_data import NiftiMapsMasker
from nilearn.image import resample_to_img

# TODOs
#
# 1. feature extraction / metadata
# 2. t maps to z maps etc.
# 3. weighting of different collections
#    (some are subject level, some are group level)
# 4. weighting features -- e.g., cog atlas terms
# 5. dirty_cat for labels using ngram similarities (SimilarityEncoder)
# 6. Encoding problem = terms to maps. Decoding problem = maps to terms

# get data / list of files
nv_data = datasets.fetch_neurovault(max_images=100)

images = nv_data['images']
images_meta = nv_data['images_meta']
collections = nv_data['collections_meta']

# export metadata to pandas
metadata = pd.DataFrame(images_meta)

# read arthur's parcellations
maps = nib.load('components_512.nii.gz')

# find images with same affine
imgs = []
for ii, image in enumerate(images):
    print('Resampling image %d' % ii)
    img = nib.load(image)
    img = resample_to_img(img, maps)
    imgs.append(img)

img_shape = img.get_data().shape

masker = NiftiMapsMasker(maps)
X = masker.fit_transform(imgs)
