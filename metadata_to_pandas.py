import pandas as pd
from nilearn import datasets

data_dir = '/home/mainak/Desktop/neurovault/'

nv_data = datasets.fetch_neurovault(max_images=None, mode='offline',
                                    data_dir=data_dir)

images = nv_data['images']
images_meta = nv_data['images_meta']
collections = nv_data['collections_meta']

# export metadata to pandas
metadata_df = pd.DataFrame(images_meta)
metadata_df['images'] = images

metadata_df.to_csv('metadata.csv', encoding='utf-8')
