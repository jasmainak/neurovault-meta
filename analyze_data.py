import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib

from nilearn import datasets
from nilearn.decomposition import DictLearning
from nilearn import plotting

# TODOs
#
# 1. feature extraction / metadata
# 2. dimensionality reduction

# get data / list of files
nv_data = datasets.fetch_neurovault(max_images=100, mode='offline')

images = nv_data['images']
images_meta = nv_data['images_meta']
collections = nv_data['collections_meta']

# export metadata to pandas
metadata = pd.DataFrame(images_meta)

# look at the metadata
if False:
    metadata.to_csv('metadata.csv', encoding='utf-8')

if False:
    dict_learn = DictLearning(n_components=5, smoothing_fwhm=6.,
                              memory="nilearn_cache", memory_level=2,
                              random_state=0)
    dict_learn.fit(images)
    components_img = dict_learn.components_img_

# plot statistical maps
if False:
    img = nib.load(images[0])
    data = img.get_data()
    plotting.plot_glass_brain(img)
    plt.show()
