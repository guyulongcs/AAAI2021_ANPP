import pickle
import pandas as pd
import os
import sys

#parameters
arr = sys.argv
assert (len(arr) == 1+6)
folder_src, file_review, file_meta, folder_dst, pkl_review, pkl_meta = arr[1], arr[2], arr[3], arr[4], arr[5], arr[6]

print "\n1_convert_pd.py start..."
#read file to df
def to_df(file_path):
  with open(file_path, 'r') as fin:
    df = {}
    i = 0
    for line in fin:
      df[i] = eval(line)
      i += 1
    df = pd.DataFrame.from_dict(df, orient='index')
    return df


#files
file_review_path = os.path.join(folder_src, file_review)
file_meta_path = os.path.join(folder_src, file_meta)

file_pkl_review = os.path.join(folder_dst, pkl_review)
file_pkl_meta = os.path.join(folder_dst, pkl_meta)

#process reviews
reviews_df = to_df(file_review_path)
with open(file_pkl_review, 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

#process meta
meta_df = to_df(file_meta_path)
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
meta_df = meta_df.reset_index(drop=True)
with open(file_pkl_meta, 'wb') as f:
    pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)

print "\n1_convert_pd.py end!"
