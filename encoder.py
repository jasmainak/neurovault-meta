import matplotlib.pyplot as plt

import nibabel as nib
import numpy as np
import pandas as pd

from nilearn import datasets
from nilearn import surface
from nilearn.plotting import (plot_stat_map, plot_glass_brain,
                              plot_surf_stat_map)
from mne.externals.six import string_types

from utils import read_resampled_img

# data_dir = None for default home directory
data_dir = '/home/mainak/Desktop/neurovault/'
search_term = 'memory'

# get data / list of files
nv_data = datasets.fetch_neurovault(max_images=None,
                                    mode='offline', data_dir=data_dir)

# just average maps containing a certain term

images = nv_data['images']
images_meta = nv_data['images_meta']
collections = nv_data['collections_meta']

# export metadata to pandas
metadata_df = pd.DataFrame(images_meta)


def func(x):
    if isinstance(x, string_types):
        return search_term in x
    else:
        return False


metadata_df['is_effect'] = metadata_df.applymap(func).any(axis=1)
effect_idxs = np.where(
    np.all(np.array([metadata_df['is_effect'].values,
                    (metadata_df['map_type'] == 'Z map').values]),
           axis=0))[0]
other_idxs = np.where(
    np.all(np.array([np.invert(metadata_df['is_effect'].values),
                    (metadata_df['map_type'] == 'Z map').values]),
           axis=0))[0]

other_idxs = np.random.choice(other_idxs, len(effect_idxs))

images_effect = [images[idx] for idx in effect_idxs]
images_other = [images[idx] for idx in other_idxs]


def average_maps(img_fnames, target_img):
    avg = np.zeros_like(target_img.get_data())
    n_images = 0
    for ii, image_fname in enumerate(img_fnames):
        img = read_resampled_img(image_fname)
        try:
            data = img.get_data()
        except IOError:
            print('Skipping')
            continue
        print('Image %d (max = %f)'
              % (ii, img.get_data().max()))
        if not np.any(np.isnan(data / data.std())):
            avg += data / data.std()
            n_images += 1
        else:
            print('Skipping ...')
            continue
    avg /= n_images
    return avg


target_img = nib.load(images[0])
avg_effect = average_maps(images_effect, target_img)
avg_other = average_maps(images_other, target_img)

contrast_img = nib.Nifti1Image(avg_effect - avg_other, target_img.affine)
avg_img = nib.Nifti1Image(avg_effect, target_img.affine)
plot_glass_brain(avg_img, plot_abs=False)

# surface
fsaverage = datasets.fetch_surf_fsaverage5()
texture = surface.vol_to_surf(contrast_img, fsaverage.pial_right)
plot_surf_stat_map(fsaverage.infl_right, texture, symmetric_cbar=True)
plt.show()
