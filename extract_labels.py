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
# metadata_df['name'] = metadata_df['name'].apply(lambda x: x.replace('_', ' '))

string_columns = metadata_df.select_dtypes(include=['object']).columns
# [(col, metadata_df[col].map(type).unique()) for col in string_columns]
string_columns = string_columns.tolist()
string_columns.remove('data')
metadata_df['all'] = metadata_df[string_columns].apply(
    lambda row: row.str.cat(sep=' '), axis=1)

vectorizer = CountVectorizer()

corpus = [metadata_df.iloc[ii]['all'] for ii in range(metadata_df.shape[0])]
bag_of_words = vectorizer.fit_transform(corpus)
sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx])
              for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
frequent_words = [w[0] for w in words_freq]

print(' '.join(frequent_words[:100]))

sdfdf

se = SimilarityEncoder(similarity='ngram', handle_unknown='ignore')
y = se.fit_transform(metadata_df.name) # XXX: need more features than 1 ...

sdfdf

# look at the metadata
if True:
    metadata_df.to_csv('metadata.csv', encoding='utf-8')
