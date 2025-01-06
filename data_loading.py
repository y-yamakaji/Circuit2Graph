"""Module providing spliting train and test dataset for pytorch geometric"""
import os
from pathlib import Path
import random
import shutil
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

from params import args
from util import fix_seed


class LoadData():
    """
    Split the graphs into train and test and output train_dataset
    and test_dataset that can be treated by pytorch geometric.

    Attributes
    ----------
    args : argparse
        params.py

    Methods
    -------
    '''
    split_condition(self)
        Defines the conditions for splitting the graph into train and
        test and saves them in .npy format in the save folder.

    split_dataset(self)
        Based on the .npy format in the saved folder, copy the graph
        for training to "07_changeToNodeNum_train" and "08_changeToEdgeNum_train"
        and copy the graph for testing to "07_changeToNodeNum_test" and
        "08_changeToEdgeNum_test.

    train_dataset(self)
        Combine the graphs in "07_changeToNodeNum_train" and "08_changeToEdgeNum_train"
        into a single data (data_list_test) for faster operation on the GPU.

    test_dataset(self)
        Combines the graphs in "07_changeToNodeNum_test" and "08_changeToEdgeNum_test"
        into one data (data_list_test) for faster operation on the GPU.

    visualize_split_data(self)
        For verification: Visualize the number of circuits of each type.
    '''

    """

    def __init__(self, arg):
        self.args = arg
        self.save_dir05 = Path(self.args.dataset) / "05_changeToNodeNum"
        self.save_dir06 = Path(self.args.dataset) / "06_changeToEdgeNum"

        self.node_train = Path(self.args.dataset) / "07_changeToNodeNum_train"
        self.node_test = Path(self.args.dataset) / "07_changeToNodeNum_test"
        self.edge_train = Path(self.args.dataset) / "08_changeToEdgeNum_train"
        self.edge_test = Path(self.args.dataset) / "08_changeToEdgeNum_test"

        self.node_train.mkdir(parents=True, exist_ok=True)
        self.node_test.mkdir(parents=True, exist_ok=True)
        self.edge_train.mkdir(parents=True, exist_ok=True)
        self.edge_test.mkdir(parents=True, exist_ok=True)

    def split_condition(self):
        '''
        Defines the conditions for splitting the graph into train and
        test and saves them in .npy format in the save folder.
        '''
        fix_seed(args.seed)

        node_files = [str(file) for file in self.save_dir05.glob("*")]
        num_files = len(node_files)
        files_test = random.sample(node_files, int(num_files * self.args.train_test))
        files_train = []
        for file in node_files:
            if file not in files_test:
                files_train.append(file)

        if not Path(self.args.dataset + os.sep + "train_file_names.npy").is_file():
            fname = []
            for file in files_train:
                fname.append(file.split(os.sep)[-1])
                np.save(self.args.dataset + os.sep + "train_file_names.npy", fname)

        if not Path(self.args.dataset + os.sep + "test_file_names.npy").is_file():
            fname = []
            for file in files_test:
                fname.append(file.split(os.sep)[-1])
                np.save(self.args.dataset + os.sep + "test_file_names.npy", fname)

    def split_dataset(self):
        '''
        Based on the .npy format in the saved folder, copy the graph
        for training to "07_changeToNodeNum_train" and "08_changeToEdgeNum_train"
        and copy the graph for testing to "07_changeToNodeNum_test" and
        "08_changeToEdgeNum_test.
        '''
        files_train = np.load(Path(self.args.dataset) / "train_file_names.npy")
        files_test = np.load(Path(self.args.dataset) / "test_file_names.npy")

        files_train = [f.replace("ltspice_dataset", args.dataset) for f in files_train]
        files_test = [f.replace("ltspice_dataset", args.dataset) for f in files_test]
       
        file01 = self.save_dir05.name
        file02 = self.save_dir06.name
        file03 = self.node_train.name
        file04 = self.edge_train.name
        file05 = self.node_test.name
        file06 = self.edge_test.name

        for file in files_train:
            file = str(self.save_dir05) + os.sep + file
            shutil.copyfile(file, file.replace(file01, file03))
            gfile = file.replace(file01, file02)
            shutil.copyfile(gfile, gfile.replace(file02, file04))

        for file in files_test:
            file = str(self.save_dir05) + os.sep + file
            shutil.copyfile(file, file.replace(file01, file05))
            gfile = file.replace(file01, file02)
            shutil.copyfile(gfile, gfile.replace(file02, file06))

    def train_dataset(self):
        '''
        Combine the graphs in "07_changeToNodeNum_train" and "08_changeToEdgeNum_train"
        into a single data (data_list_test) for faster operation on the GPU.
        '''
        print(self.node_train)
        node_files = [str(file) for file in self.node_train.glob("*")]
        
        data_list_train = []
        for _, file in enumerate(node_files):
            x_data = torch.tensor(np.load(file), dtype=torch.float)
            y_data = file.split(os.sep)[-1].split("_")[0]

            file_load = np.load(file.replace(str(self.node_train), str(self.edge_train)))  # load edges
            edge = torch.tensor(file_load, dtype=torch.long)

            torch_data = Data(x=x_data, edge_index=edge.t().contiguous(), y=int(y_data))
            data_list_train.append(torch_data)
        return data_list_train

    def test_dataset(self):
        '''
        Combines the graphs in "07_changeToNodeNum_test" and "08_changeToEdgeNum_test"
        into one data (data_list_test) for faster operation on the GPU.
        '''
        node_files = [str(file) for file in self.node_test.glob("*")]

        data_list_test = []
        for _, file in enumerate(node_files):
            x_data = torch.tensor(np.load(file), dtype=torch.float)
            y_data = file.split(os.sep)[-1].split("_")[0]

            file_load = np.load(file.replace(str(self.node_test), str(self.edge_test)))  # load edges
            edge = torch.tensor(file_load, dtype=torch.long)

            torch_data = Data(x=x_data, edge_index=edge.t().contiguous(), y=int(y_data))
            data_list_test.append(torch_data)
        return data_list_test

    def visualize_split_data(self):
        '''
        For verification: Visualize the number of circuits of each type.
        '''
        files = [str(file) for file in self.node_train.glob("*")]
        num = []
        for file in files:
            num.append(int(file.split(os.sep)[-1].split("_")[0]))
        count = Counter(num)
        print(count)
        plt.bar(count.keys(), count.values())


if __name__ == '__main__':
    load = LoadData(args)
    load.split_condition()
    load.split_dataset()
    # load.visualize_split_data()
    data_train = load.train_dataset()
