import numpy as np
import pandas as pd
import os
import sys
import pickle

file_user_log = 'user_log_format1.csv'
file_user_purchase = "user_purchase.csv"

file_item_info = "item.csv"

min_user_count = 5
min_item_count = 5

purchase_action_type = 2

arr = sys.argv
folder_dataset = arr[1]
file_pkl_review = arr[2]
file_pkl_meta = arr[3]


year = '2014'

def norm_timestamp(t):
	s = str(t)
	y = year
	if(len(s) < 4):
		s = "0" + s
	m = s[0:2]
	d = s[-2:]
	return "-".join([y, m, d])

def filt_purchase(folder, file_src, file_dst):
	print("filt_purchase...")
	file_src = os.path.join(folder, file_src)
	file_dst = os.path.join(folder, file_dst)
	df = pd.read_csv(file_src)

	df = df[df['action_type'] == 2]
	df.rename(columns={'user_id': 'user', 'item_id': 'item', 'cat_id': 'categories', 'brand_id': 'brand', 'time_stamp': 'timestamp'}, inplace=True)
	df = df[['user', 'item', 'timestamp', 'categories', 'brand']]
	df['timestamp'] = df['timestamp'].map(lambda x: norm_timestamp(x))

	df = df.sort_values(['user', 'timestamp'])
	df.to_csv(file_dst, index=False)
	print("filt_purchase done!")

def norm_order_data(folder, file_src,  file_pkl_review, file_pkl_meta):
	print("norm_order_data...")
	file_src = os.path.join(folder, file_src)


	df = pd.read_csv(file_src)
	df['title'] = ""

	#filt speical data
	df = df[df['timestamp'] < '2014-11-01']

	#filt
	df = df.groupby('item').filter(lambda x: len(x) >= min_item_count)
	df = df.groupby('user').filter(lambda x: len(x) >= min_user_count)

	user_df = df[['user', 'item', 'timestamp']].drop_duplicates()
	meta_df =  df[['item', 'categories', 'title', 'brand']].drop_duplicates()

	file_pkl_review = os.path.join(folder, file_pkl_review)
	with open(file_pkl_review, 'wb') as f:
		pickle.dump(user_df, f, pickle.HIGHEST_PROTOCOL)


	file_pkl_meta = os.path.join(folder, file_pkl_meta)
	with open(file_pkl_meta, 'wb') as f:
		pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)

	print("norm_order_data done!")



def main():
	#filt_purchase(folder_dataset, file_user_log, file_user_purchase)
	norm_order_data(folder_dataset, file_user_purchase, file_pkl_review, file_pkl_meta)

main()