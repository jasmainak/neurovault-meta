import os
import os.path as op

from nilearn.image import resample_to_img
from nilearn import datasets

import nibabel as nib

data_dir = '/home/mainak/Desktop/neurovault/'
base_dir = 'neurovault_resampled'
nv_data = datasets.fetch_neurovault(max_images=None, mode='offline',
                                    data_dir=data_dir)

images = nv_data['images']

if not op.exists(base_dir):
    os.mkdir(base_dir)

target_img = nib.load(images[0])
for ii, image in enumerate(images):
    collection, name = image.split('/')[-2:]
    fname = op.join(base_dir, collection, name)
    print('Resampling image %d' % ii)
    if op.exists(fname):
        continue

    if not op.exists(op.join(base_dir, collection)):
        os.mkdir(op.join(base_dir, collection))
    img = nib.load(image)
    img = resample_to_img(img, target_img)
    nib.save(img, fname)
