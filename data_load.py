import random
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.utils.rnn as rnn_utils
import kaldiio

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    seq, label = zip(*batch)
    seq_length = [len(x) for x in label]
    data = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)
    # label_stack = []
    label = rnn_utils.pad_sequence(label, batch_first=True, padding_value=0)
    # return data, torch.tensor(label_stack), seq_length
    return data, label, seq_length


def collate_fn_atten(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    seq, labels, seq_length = zip(*batch)
    data = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels)
    return data, labels, seq_length


class RawFeatures(data.Dataset):
    def __init__(self, txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        self.feature_list = [i.split()[0] for i in lines]
        self.label_list = [i.split()[1] for i in lines]
        self.seq_len_list = [i.split()[2].strip() for i in lines]

    def __getitem__(self, index):
        feature_path = self.feature_list[index]
        feature = torch.from_numpy(np.load(feature_path, allow_pickle=True))
        label = int(self.label_list[index])
        seq_len = int(self.seq_len_list[index])
        return feature, label, seq_len

    def __len__(self):
        return len(self.label_list)

class RawFeatures2(data.Dataset):
    def __init__(self, txt_path, train_feats):
        with open(train_feats, 'r') as f:
            lines = f.readlines()
        self.feat_dict = {}
        for line in lines:
            temp = line.split()
            self.feat_dict[temp[0]] = temp[1]

        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.feature_list = [i.split()[0] for i in lines]
            self.label_list = [i.split()[1] for i in lines]
            self.seq_len_list = [i.split()[2].strip() for i in lines]

    def __getitem__(self, index):
        # feature_path = self.feature_list[index]
        feature_path = self.feat_dict[self.feature_list[index]]
        feature = torch.from_numpy(kaldiio.load_mat(feature_path))

        # feature = torch.from_numpy(np.load(feature_path, allow_pickle=True))
        label = int(self.label_list[index])
        seq_len = int(self.seq_len_list[index])
        return feature, label, seq_len

    def __len__(self):
        return len(self.label_list)

class RawFeaturesCurriculumLearning(data.Dataset):
    # 更改比例
    def __init__(self, txt_path, train_feats, scale):
        print(txt_path)
        print(train_feats)
        with open(train_feats, 'r') as f:
            lines = f.readlines()
        # 通过uttName来存储特征
        self.feat_dict = {}
        for line in lines:
            temp = line.split()
            self.feat_dict[temp[0]] = temp[1].replace('\n', '')
            
        # 做一下数据的选择
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        length = len(lines)
        # 将整个数组乱序
        random.shuffle(lines)
        # feature_list就是name
        self.feature_list = [i.split()[0] for i in lines]
        # label_list就是label标签
        self.label_list = [i.split()[1] for i in lines]
        # 特征的长度
        self.seq_len_list = [i.split()[2].strip() for i in lines]
        
        l = int(len(self.label_list)/5)
        if len(scale) != 5:
            print("Error, the number of data set not equal to 5")
        else:
            arr = []
            for i in scale:
                arr.append(int(i * l))
            temp_feature_list = []
            temp_label_list = []
            temp_seq_len_list = []
            number = [0, 0, 0, 0, 0]
            tail_symbol = ["_clean", "_20_snrs", "_15_snrs", "_10_snrs", "_5_snrs"]
            for i in range(length):
                flag = 0
                for j in range(1, len(tail_symbol)):
                    if self.feature_list[i].endswith(tail_symbol[j]):
                        if arr[j] > number[j]:
                            temp_feature_list.append(self.feature_list[i])
                            temp_label_list.append(self.label_list[i])
                            temp_seq_len_list.append(self.seq_len_list[i])
                            number[j] += 1
                        flag = 1
                        break
                # 代表着clean的数据集
                if flag == 0:
                    # print(self.feature_list[i])
                    if arr[0] > number[0]:
                        temp_feature_list.append(self.feature_list[i])
                        temp_label_list.append(self.label_list[i])
                        temp_seq_len_list.append(self.seq_len_list[i])
                        number[0] += 1
                flag = 0
            self.feature_list = temp_feature_list
            self.label_list = temp_label_list
            self.seq_len_list = temp_seq_len_list
            print(number)
            print(arr)
    def __getitem__(self, index):
        # feature_path = self.feature_list[index]
        feature_path = self.feat_dict[self.feature_list[index]]
        feature = torch.from_numpy(kaldiio.load_mat(feature_path))
        label = int(self.label_list[index])
        seq_len = int(self.seq_len_list[index])
        return feature, label, seq_len

    def __len__(self):
        return len(self.label_list)

def get_atten_mask(seq_lens, batch_size):
    max_len = seq_lens[0]
    atten_mask = torch.ones([batch_size, max_len, max_len])
    for i in range(batch_size):
        length = seq_lens[i]
        atten_mask[i, :length, :length] = 0
    return atten_mask.bool()

def get_atten_mask_frame(seq_lens, batch_size):
    max_len = seq_lens[0]
    atten_mask = torch.ones([batch_size, max_len*20, max_len*20])
    for i in range(batch_size):
        length = seq_lens[i]*20
        atten_mask[i, :length, :length] = 0
    return atten_mask.bool()

def get_atten_mask_student(seq_lens, batch_size, mask_type='fix', win_len=15):
  max_len = seq_lens[0]
  atten_mask = torch.ones([batch_size, max_len, max_len])
  if mask_type == 'fix':
    for i in range(batch_size):
      atten_mask[i, 0:win_len, 0:win_len] = 0
  elif mask_type == 'random':
    for i in range(batch_size):
      seq_len = seq_lens[i]
      if seq_len>win_len:
          rest_len = seq_len - win_len
          start = random.randint(0, rest_len)
          end = start + win_len
          atten_mask[i, start:end, start:end] = 0
      else:
          atten_mask[i, :seq_len, :seq_len] = 0
  return atten_mask.bool()


def std_mask(seq_lens, batchsize, dim):
    max_len = seq_lens[0]
    weight_unbaised = torch.tensor(seq_lens) / (torch.tensor(seq_lens) - 1)
    atten_mask = torch.ones([batchsize, max_len, dim])
    for i in range(batchsize):
        length = seq_lens[i]
        atten_mask[i, length:, :] = 1e-9
    return atten_mask, weight_unbaised

def mean_mask(seq_lens, batchsize, dim):
    max_len = seq_lens[0]
    weight_unbaised = seq_lens[0] / torch.tensor(seq_lens)
    atten_mask = torch.ones([batchsize, max_len, dim])
    for i in range(batchsize):
        length = seq_lens[i]
        atten_mask[i, :length, :] = 0
    return atten_mask.bool(), weight_unbaised

def layer_mask(seq_lens, batchsize, dim):
    max_len = seq_lens[0]*20
    atten_mask = torch.ones([batchsize, max_len, dim])
    for i in range(batchsize):
        length=seq_lens[i]
        atten_mask[i, :length, :] = 0
    return atten_mask.bool()

def se_mask(seq_lens, batchsize):
    max_len = seq_lens[0]
    weight_unbaised = seq_lens[0] / torch.tensor(seq_lens)
    atten_mask = torch.ones([batchsize, max_len*20])
    for i in range(batchsize):
        length=seq_lens[i]*20
        atten_mask[i, :length] = 0
    return atten_mask.bool()


