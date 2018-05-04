import matplotlib.pyplot as plt

import pandas as pd
import nibabel as nib
import numpy as np

from nilearn import datasets
from nilearn import surface
from nilearn.plotting import (plot_stat_map, plot_glass_brain,
                              plot_surf_stat_map)
from mne.externals.six import string_types
from mne.utils import ProgressBar

from utils import read_resampled_img

# data_dir = None for default home directory
data_dir = '/home/mainak/Desktop/neurovault/'
search_term = 'motor'

# export metadata to pandas
print('Loading metadata')
metadata_df = pd.read_csv('metadata.csv', low_memory=False)
images = metadata_df.images
print('Done')


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


def average_maps(img_fnames, target_img, desc):
    avg = np.zeros_like(target_img.get_data())
    n_images = 0
    print('')
    for ii, image_fname in enumerate(ProgressBar(
            img_fnames, mesg=desc, spinner=True)):
        collection, name = image_fname.split('/')[-2:]
        img = read_resampled_img(image_fname)
        try:
            data = img.get_data()
        except IOError:
            continue
        # print('Image %d (max = %f)'
        #       % (ii, img.get_data().max()))
        if not np.any(np.isnan(data / data.std())):
            avg += data / data.std()
            n_images += 1
        else:
            continue
    avg /= n_images
    return avg


def get_collection_counts(img_fnames):
    counts = dict()
    for ii, image in enumerate(img_fnames):
        collection, name = image.split('/')[-2:]
        counts[collection] = counts.get(collection, 0) + 1
    return counts


# counts = get_collection_counts(images)

target_img = nib.load(images[0])
desc = 'Average maps for %s' % search_term
avg_effect = average_maps(images_effect, target_img, desc)
desc = 'Average other maps'
avg_other = average_maps(images_other, target_img, desc)

contrast_img = nib.Nifti1Image(avg_effect - avg_other, target_img.affine)
avg_img = nib.Nifti1Image(avg_effect, target_img.affine)
fig = plot_glass_brain(contrast_img, plot_abs=False)
fig.savefig('figures/encoder_%s.png' % search_term)

# surface
fsaverage = datasets.fetch_surf_fsaverage5()
texture = surface.vol_to_surf(contrast_img, fsaverage.pial_right)
plot_surf_stat_map(fsaverage.infl_right, texture, symmetric_cbar=True)
plt.show()
