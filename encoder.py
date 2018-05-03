import os.path as op

import nibabel as nib
import numpy as np
import pandas as pd

from nilearn import datasets
from nilearn.plotting import plot_stat_map

from mne.externals.six import string_types

# get data / list of files
nv_data = datasets.fetch_neurovault(max_images=None,
                                    mode='offline')

# just average maps containing a certain term

images = nv_data['images']
images_meta = nv_data['images_meta']
collections = nv_data['collections_meta']

# export metadata to pandas
metadata_df = pd.DataFrame(images_meta)


def func(x):
    if isinstance(x, string_types):
        return 'motor' in x
    else:
        return False


metadata_df['is_motor'] = metadata_df.applymap(func).any(axis=1)
motor_idxs = np.where(
    np.all(np.array([metadata_df['is_motor'].values,
                    (metadata_df['map_type'] == 'Z map').values]),
           axis=0))[0]
other_idxs = np.where(
    np.all(np.array([np.invert(metadata_df['is_motor'].values),
                    (metadata_df['map_type'] == 'Z map').values]),
           axis=0))[0]

other_idxs = np.random.choice(other_idxs, len(motor_idxs))

images_motor = [images[idx] for idx in motor_idxs]
images_other = [images[idx] for idx in other_idxs]


def read_resampled_img(img_fname):
    collection, name = img_fname.split('/')[-2:]
    fname = op.join('neurovault_resampled',
                    collection, name)
    img = nib.load(fname)
    return img


def average_maps(img_fnames, target_img):
    avg = np.zeros_like(target_img.dataobj)
    for ii, image_fname in enumerate(img_fnames):
        img = read_resampled_img(image_fname)
        avg += img.get_data()
        print('Resampling image %d (max = %f)'
              % (ii, img.get_data().max()))
    avg /= len(img_fnames)
    return avg


target_img = nib.load(images[0])
avg_motor = average_maps(images_motor, target_img)
avg_other = average_maps(images_other, target_img)
contrast = nib.Nifti1Image(avg_motor - avg_other, target_img.affine)

plot_stat_map(contrast, vmax=100)
