import torch
from torch.utils.data import Dataset
from numpy.random import choice as npc
import numpy as np
import random


class Trainset(Dataset):

    def __init__(self, anndata, sgRNA_list):
        super(Trainset, self).__init__()
        np.random.seed(0)
        #self.transform = transform
        self.sgRNA_list = sgRNA_list
        self.datas, _ = make_total_data(anndata, sgRNA_list)
        self.num_classes = len(sgRNA_list) + 1

    def __len__(self):
        return 1280000000

    def __getitem__(self, index):
        label = None
        cell1 = None
        cell2 = None
        # get cell from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            cell1 = random.choice(self.datas[idx1])
            cell2 = random.choice(self.datas[idx1])
        # get cell from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            cell1 = random.choice(self.datas[idx1])
            cell2 = random.choice(self.datas[idx2])

        cell1 = torch.from_numpy(cell1).float()
        cell2 = torch.from_numpy(cell2).float()
        return cell1, cell2, torch.from_numpy(np.array([label], dtype=np.float32))

class Testset(Dataset):

    def __init__(self, anndata, sgRNA_list, times=200, way=20):
        np.random.seed(1)
        super(Testset, self).__init__()
        #self.transform = transform
        self.times = times
        self.way = way
        self.cell1 = None
        self.idx1 = None
        _ ,self.datas  = make_total_data(anndata, sgRNA_list)
        self.num_classes = len(sgRNA_list) + 1

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate cell pair from same class
        if idx == 0:
            self.idx1 = random.randint(0, self.num_classes - 1)
            self.cell1 = random.choice(self.datas[self.idx1])
            cell2 = random.choice(self.datas[self.idx1])
        # generate cell pair from different class
        else:
            idx2 = random.randint(0, self.num_classes - 1)
            while self.idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            cell2 = random.choice(self.datas[idx2])


        cell1 = torch.from_numpy(self.cell1).float()
        cell2 = torch.from_numpy(cell2).float()
        return cell1, cell2




def delete_zero(X):
    return X[~np.all(X == 0, axis=1)]

def make_one_perturbed_data(anndata, idx = None, sgRNA_list=[]):

    if idx != None and idx < len(sgRNA_list) :
        sgRNA = sgRNA_list[idx]
        X_is_pertrubed = (anndata.obs[sgRNA] != 0).values.reshape(-1, 1)
        X_not_perturbed = (anndata.obs[sgRNA] == 0).values.reshape(-1, 1)
        X_train = (anndata.obs['batch'] == "0").to_numpy(dtype=int).reshape(-1, 1)
        X_test = (anndata.obs['batch'] == "1").to_numpy(dtype=int).reshape(-1, 1)

        X_pertrubed_train = delete_zero(X_is_pertrubed * anndata.obsm['X_pca'] * X_train)
        X_pertrubed_test = delete_zero(X_is_pertrubed * anndata.obsm['X_pca'] * X_test)
        X_npertrubed_train = delete_zero(X_not_perturbed * anndata.obsm['X_pca'] * X_train)
        X_npertrubed_test = delete_zero(X_not_perturbed * anndata.obsm['X_pca'] * X_test)

        return X_pertrubed_train, X_pertrubed_test, X_npertrubed_train, X_npertrubed_test

    if idx == None and sgRNA_list != []:
        X_is_pertrubed = (anndata.obs[sgRNA_list].sum(axis=1) == 0).values.reshape(-1, 1)
        X_not_perturbed = (anndata.obs[sgRNA_list].sum(axis=1) != 0).values.reshape(-1, 1)
        X_train = (anndata.obs['batch'] == "0").to_numpy(dtype=int).reshape(-1, 1)
        X_test = (anndata.obs['batch'] == "1").to_numpy(dtype=int).reshape(-1, 1)

        X_pertrubed_train = delete_zero(X_is_pertrubed * anndata.obsm['X_pca'] * X_train)
        X_pertrubed_test = delete_zero(X_is_pertrubed * anndata.obsm['X_pca'] * X_test)
        X_npertrubed_train = delete_zero(X_not_perturbed * anndata.obsm['X_pca'] * X_train)
        X_npertrubed_test = delete_zero(X_not_perturbed * anndata.obsm['X_pca'] * X_test)

        return X_pertrubed_train, X_pertrubed_test, X_npertrubed_train, X_npertrubed_test

    else:
        print('No sgRNA found')


def make_total_data(anndata, sgRNA_list = []):

    if sgRNA_list != []:
        X_train = {}
        X_test = {}

        for idx in range(len(sgRNA_list)):
            X_train[idx],X_test[idx],_ ,_ = make_one_perturbed_data(anndata, idx, sgRNA_list)

        X_train[len(sgRNA_list)],X_test[len(sgRNA_list)],_ ,_ = make_one_perturbed_data(anndata, None, sgRNA_list)

        return X_train,X_test
    else:
        print('No sgRNA found')





