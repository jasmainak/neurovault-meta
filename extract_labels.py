import pandas as pd

from nilearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from dirty_cat import SimilarityEncoder

# get data / list of files
nv_data = datasets.fetch_neurovault(max_images=None, mode='offline')

images = nv_data['images']
images_meta = nv_data['images_meta']
collections = nv_data['collections_meta']

# export metadata to pandas
metadata_df = pd.DataFrame(images_meta)
metadata_df['name'] = metadata_df['name'].apply(lambda x: x.replace('_', ' '))

se = SimilarityEncoder(similarity='ngram', handle_unknown='ignore')
vectorizer = CountVectorizer()

y = se.fit_transform(metadata_df.name) # XXX: need more features than 1 ...

sdfdf
y = vectorizer.fit_transform(metadata_df.name)

# look at the metadata
if True:
    metadata_df.to_csv('metadata.csv', encoding='utf-8')
