import numpy as np

class DataInput:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def next(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    u, i, cate, y, sl = [], [], [], [], []
    for t in ts:
      [user_id, (hist_item, cur_item), (hist_cate, cur_cate), (hist_time, cur_time_ts), label] = t
      u.append(user_id)
      i.append(cur_item)
      cate.append(cur_cate)
      y.append(label)
      sl.append(len(hist_item))
    max_sl = max(sl)

    hist_i = np.zeros([len(ts), max_sl], np.int64)
    hist_c = np.zeros([len(ts), max_sl], np.int64)
    hist_t = np.zeros([len(ts), max_sl], np.float32)

    k = 0
    for t in ts:
      [user_id, (hist_item, cur_item), (hist_cate, cur_cate), (hist_time, cur_time_ts), label] = t
      for l in range(len(hist_item)):
        hist_i[k][l] = hist_item[l]
        hist_c[k][l] = hist_cate[l]
        hist_t[k][l] = hist_time[l]
      k += 1

    return self.i, (u, i, cate, y, hist_i, hist_c, hist_t, sl)

class DataInputTest:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def next(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1


    max_negN_item = 0
    max_negN_cate = 0
    u, i, cate, sl = [], [], [], []

    for t in ts:
      [user_id, (hist_item, cur_item, neg_list_N_item_test), (hist_cate, cur_cate, neg_list_N_cate_test), (hist_time, cur_time_ts)] = t
      u.append(user_id)
      i.append(cur_item)
      cate.append(cur_cate)
      sl.append(len(hist_item))
      max_negN_item = max(max_negN_item, len(neg_list_N_item_test))
      max_negN_cate = max(max_negN_cate, len(neg_list_N_cate_test))

    max_sl = max(sl)

    hist_i = np.zeros([len(ts), max_sl], np.int64)
    hist_c = np.zeros([len(ts), max_sl], np.float32)
    hist_t = np.zeros([len(ts), max_sl], np.float32)
    negN_item = np.zeros([len(ts), max_negN_item], np.int32)
    negN_cate = np.zeros([len(ts), max_negN_cate], np.int32)

    k = 0
    for t in ts:
      [user_id, (hist_item, cur_item, neg_list_N_item_test), (hist_cate, cur_cate, neg_list_N_cate_test),
       (hist_time, cur_time_ts)] = t
      for l in range(len(hist_item)):
        hist_i[k][l] = hist_item[l]
        hist_c[k][l] = hist_cate[l]
        hist_t[k][l] = hist_time[l]

      negN_item[k] = np.array(neg_list_N_item_test)
      negN_cate[k] = np.array(neg_list_N_cate_test)
      k += 1

    return self.i, (u, i, cate, hist_i, hist_c, hist_t, sl, negN_item, negN_cate)
