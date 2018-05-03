import nibabel as nib
import numpy as np
import pandas as pd

from nilearn import datasets
from nilearn.image import resample_to_img
from nilearn.plotting import plot_glass_brain

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

motor_idxs = np.where(metadata_df['is_motor'])[0]
other_idxs = np.where(np.invert(metadata_df['is_motor']))[0]

other_idxs = np.random.choice(other_idxs, len(motor_idxs))

images_motor = [images[idx] for idx in motor_idxs]
images_other = [images[idx] for idx in other_idxs]


def average_maps(img_fnames, target_img):
    avg = np.zeros_like(target_img.dataobj)
    for ii, image_fname in enumerate(img_fnames):
        print('Resampling image %d' % ii)
        img = nib.load(image_fname)
        img = resample_to_img(img, target_img)
        avg += img.dataobj
    avg /= len(img_fnames)
    return avg


target_img = nib.load(images[0])
avg_motor = average_maps(images_motor, target_img)
avg_other = average_maps(images_other, target_img)
contrast = nib.Nifti1Image(avg_motor - avg_other, target_img.affine)

plot_glass_brain(contrast, plot_abs=True, colorbar=True)
