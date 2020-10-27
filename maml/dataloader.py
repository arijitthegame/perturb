import torch
from torch.utils.data import Dataset
from numpy.random import choice as npc
import numpy as np
import random
import helpfuntions

class perturbdataloader(Dataset):

    def __init__(self, dataset , ways = 10, support_shots = 5, query_shots = 15, shuffle = True):
        super(perturbdataloader, self).__init__()
        np.random.seed(0)
        self.dataset = dataset
        self.num_classes = len(self.dataset)
        self.ways = ways
        self.support_shots = support_shots
        self.query_shots  = query_shots
        self.shuffle = shuffle

    def __len__(self):
        return 12800000000000

    def __getitem__(self, index):
        select = random.sample(range(self.num_classes), self.ways)
        self.datas = helpfuntions.make_datas(self.dataset,select)

        inputs_support, inputs_query, target_support, target_query = helpfuntions.sample_once(self.datas, self.support_shots,
                                                                                              self.query_shots, shuffle=self.shuffle)
        #batch = {}
        #batch["support"] = [inputs_support, target_support.reshape(-1,1)]
        #batch['query'] = [inputs_query, target_query.reshape(-1,1)]
        return torch.from_numpy(inputs_support).float(), torch.from_numpy(inputs_query).float(), \
               torch.from_numpy(target_support).long(), torch.from_numpy(target_query).long()

class perturbdataloader_test(Dataset):

    def __init__(self, test_support, test_target_support, support_shots = 5):
        super(perturbdataloader_test, self).__init__()
        np.random.seed(0)
        self.dataset = test_support
        self.target = test_target_support
        self.item = list(range(len(self.target)))
        random.shuffle(self.item)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return torch.tensor(self.dataset[self.item[index]]).float(),\
               torch.tensor(self.target[self.item[index]]).long()