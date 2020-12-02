import random
import pickle
import numpy as np
import os
import sys
import collections
import pandas as pd

sys.path.append("../..")

from Config import *
from utils.Time import *

print "\nremap_id.py start..."

#parameters
arr = sys.argv
assert (len(arr) == 1+6)
folder_src, file_pkl_review, file_pkl_meta, folder_dst, file_join, pkl_remap = arr[1], arr[2], arr[3], arr[4], arr[5], arr[6]




#review: user, item, timestamp
file_pkl_review = os.path.join(folder_src, file_pkl_review)
print("file_pkl_review:", file_pkl_review)
with open(file_pkl_review, 'rb') as f:
    user_df = pickle.load(f)
    user_df = user_df[['user', 'item', 'timestamp']]

#meta: item, categories, title, brand
file_pkl_meta = os.path.join(folder_src, file_pkl_meta)
print("file_pkl_meta:", file_pkl_meta)
with open(file_pkl_meta, 'rb') as f:
    meta_df = pickle.load(f)
    meta_df = meta_df[['item', 'categories', 'title', 'brand']]

#join
join_df = user_df.join(meta_df.set_index('item'), on='item')
join_df = join_df.sort_values(['user', 'timestamp'])
join_df['datetime'] = join_df['timestamp']
file_join = os.path.join(folder_src, file_join)
print("file_join:", file_join)
join_df.to_csv(file_join, columns=['user', 'item', 'timestamp', 'datetime', 'categories', 'title'], index=False)

#map
#m: item name to int(1,2,...,n)
#key: item name list
#df: item int list
def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(1, len(key)+1)))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key
#
# #review: user, item, timestamp
# with open(file_pkl_review, 'rb') as f:
#     user_df = pickle.load(f)
#     user_df = user_df[['reviewerID', 'asin', 'unixReviewTime']]
#     user_df.rename(columns={'reviewerID':'user', 'asin':'item', 'unixReviewTime': 'timestamp'}, inplace = True)

#meta: item, categories, title
# with open(file_pkl_meta, 'rb') as f:
#     meta_df = pickle.load(f)
#     meta_df = meta_df[['asin', 'categories', 'title']]
#     meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])
#     meta_df.rename(columns={'asin': 'item'}, inplace=True)
#
# #join
# join_df = user_df.join(meta_df.set_index('item'), on='item')
# join_df = join_df.sort_values(['user', 'timestamp'])
# join_df['datetime'] = join_df['timestamp'].map(lambda x: Time.timestamp_to_datetime(x))
# join_df.to_csv(file_join, columns=['user', 'item', 'timestamp', 'datetime', 'categories', 'title'], index=False)
#


#map
#m: item name to int(1,2,...,n)
# #key: item name list
# #df: item int list
# def build_map(df, col_name):
#     key = sorted(df[col_name].unique().tolist())
#     m = dict(zip(key, range(1, len(key)+1)))
#     df[col_name] = df[col_name].map(lambda x: m[x])
#     return m, key

item_map, item_key = build_map(meta_df, 'item')
cate_map, cate_key = build_map(meta_df, 'categories')
user_map, user_key = build_map(user_df, 'user')

user_count, item_count, cate_count, example_count =\
    len(user_map), len(item_map), len(cate_map), user_df.shape[0]


#file desc
file_desc = os.path.join(folder_src, Config.file_dataset_desc)
with open(file_desc, 'w') as f:
    descStr = "user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d\n" % \
              (user_count, item_count, cate_count, example_count)
    print(descStr)
    f.write(descStr)

    #time gap(Day)
    list_gap = []
    for user_id, h in join_df.groupby('user'):
        h_t = h['datetime'].tolist()
        for i in range(1, len(h_t)):
            gap = Time.get_time_diff_days_str_str(h_t[i - 1], h_t[i])
            list_gap.append(gap)
    gap_counter = collections.Counter(list_gap).most_common()
    f.write("\ntime_gap_counter (Day):\n")
    for k in gap_counter:
        line = "%d\t%d\n" % (k[0], k[1])
        f.write(line)

meta_df = meta_df.sort_values('item')
meta_df = meta_df.reset_index(drop=True)
user_df['item'] = user_df['item'].map(lambda x: item_map[x])
user_df = user_df.sort_values(['user', 'timestamp'])
user_df = user_df.reset_index(drop=True)
user_df = user_df[['user', 'item', 'timestamp']]

cate_list = [meta_df['categories'][i] for i in range(len(item_map))]
cate_list = np.array(cate_list, dtype=np.int32)

#dict_item_cate: item, cate
dict_item_cate = meta_df.set_index('item').to_dict()['categories']

#write file remap
file_pkl_remap = os.path.join(folder_dst, pkl_remap)
with open(file_pkl_remap, 'wb') as f:
    #user_df: user, item, timestamp [id format]
    pickle.dump(user_df, f, pickle.HIGHEST_PROTOCOL)
    #catelist: category list         [id format]
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count, example_count),
              f, pickle.HIGHEST_PROTOCOL)
    #item_map, cate_map, user_map:   name, id
    pickle.dump((item_map, cate_map, user_map), f, pickle.HIGHEST_PROTOCOL)
    #dict_item_cate
    pickle.dump(dict_item_cate, f, pickle.HIGHEST_PROTOCOL)
print "remap_id.py end!"