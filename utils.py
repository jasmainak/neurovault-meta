import nibabel as nib
import os.path as op


def read_resampled_img(img_fname):
    collection, name = img_fname.split('/')[-2:]
    fname = op.join('neurovault_resampled',
                    collection, name)
    img = nib.load(fname)
    return img
