import pandas as pd

from nilearn import datasets
from sklearn.feature_extraction.text import CountVectorizer

# get data / list of files
nv_data = datasets.fetch_neurovault(max_images=None, mode='offline')

images = nv_data['images']
images_meta = nv_data['images_meta']
collections = nv_data['collections_meta']

# export metadata to pandas
metadata_df = pd.DataFrame(images_meta)
metadata_df['name'] = metadata_df['name'].apply(lambda x: x.replace('_', ' '))

vectorizer = CountVectorizer()
y = vectorizer.fit_transform(metadata_df.name)

# look at the metadata
if True:
    metadata_df.to_csv('metadata.csv', encoding='utf-8')
