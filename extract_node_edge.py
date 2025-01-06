"""Module providing extracting nodes and edges from netlists"""
from pathlib import Path
import re
import pickle
import itertools
from collections import Counter
import numpy as np

from params import args


class Netlist2Graph():
    """
    Create netlist from LTspice schematic. Assign circuit type information
    for each netlist and save it in a folder.

    Attributes
    ----------
    args : argparse
        params.py

    Methods
    -------
    extract_node(self)
        Extracts nodes (= circuit components) from the netlist and saves it
        as a pickle format in the "save folder"->03_extractNode folder.

    extract_edge(self)
        Extracts edges (= wirings) from the netlist and saves it as a pickle
        format in the "save folder"->04_extractEdge folder. In particular, add
        the wiring to the list(zero_edge) to convert the ground from edges to a node.

    extract_parts_symbol(self)
        Extracts the circuit components used in all netlists and saves
        them in the list(type_of_symbol).

    replace_node_num(self)
        Convert the list(type_of_symbol) as an index of the circuit components and
        the nodes into numbers and save them in the "save folder"->05_changeToNodeNum.

    replace_edge_num(self)
        Based on the number of  "05_changeToNodeNum", convert an edge into an edge
        consisting of a combination (tuple) of two numbers, and save it in
        "save folder"->06_changeToEdgeNum.

    visualize_graph(self, num=0)
        For verification: Graphicalize the edges created by method(replace_edge_num()).

    check_node_num(self)
        For verification: count the number of nodes for each component type.

    check_edge_num(self)
        For verification: count the number of edges for each component type.
    """
    def __init__(self, arg) -> None:
        self.args = arg
        self.save_dir02 = Path(self.args.dataset) / "02_circuitConstant"
        self.save_dir03 = Path(self.args.dataset) / "03_extractNode"
        self.save_dir04 = Path(self.args.dataset) / "04_extractEdge"
        self.save_dir05 = Path(self.args.dataset) / "05_changeToNodeNum"
        self.save_dir06 = Path(self.args.dataset) / "06_changeToEdgeNum"
        self.type_of_symbol = []  # All symbols used in LTspice

        self.save_dir03.mkdir(parents=True, exist_ok=True)
        self.save_dir04.mkdir(parents=True, exist_ok=True)
        self.save_dir05.mkdir(parents=True, exist_ok=True)
        self.save_dir06.mkdir(parents=True, exist_ok=True)

    def extract_node(self):
        """
        Extracts nodes (= circuit components) from the netlist and saves it
        as a pickle format in the "save folder"->03_extractNode folder.
        """
        nodes = ["0"]  # add GND as "0"
        files = [str(file) for file in self.save_dir02.glob("*")]

        for file in files:
            with open(file, "rb") as ufp:
                data = pickle.load(ufp)

            node = ["0"]  # adding GND node "0"
            for each_data in data:
                d_split = each_data.split(" ")
                if d_split[0][0] in {"R", "L", "C"}:
                    node_entry = f"{d_split[0]}:{d_split[3]}"
                    node.append(node_entry)
                    nodes.append(node_entry)
                else:
                    node.append(d_split[0])
                    nodes.append(d_split[0])

            save_path = Path(self.save_dir03) / Path(file).name
            with open(save_path, "wb") as fpack:
                pickle.dump(node, fpack)

    def extract_edge(self):
        """
        Extracts edges (= wirings) from the netlist and saves it as a pickle
        format in the "save folder"->04_extractEdge folder. In particular, add
        the wiring to the list(zero_edge) to convert the ground from edges to a node.
        """
        files = list(self.save_dir02.glob("*"))
        for file in files:
            with open(file, "rb") as ufp:  # Unpickling
                data = pickle.load(ufp)

            extract_edge_name = []
            for each_data in data:
                d_split = each_data.split(" ")
                extract_edge_name.extend(d_split[1:])

            c = Counter(extract_edge_name)
            extract_edge_name = list(set(item for item in extract_edge_name if c[item] > 1 and item != '0'))

            edges = []
            for edge in extract_edge_name:
                output = [d_split[0] for each_data in data if edge in (d_split := each_data.split(" "))]
                edges.extend(itertools.combinations(set(output), 2))

            zero_edge = [("0", d_split[0]) for each_data in data if "0" in (d_split := each_data.split(" "))]

            save_path = Path(self.save_dir04) / Path(file).name
            with open(save_path, "wb") as fpack:
                pickle.dump(edges + zero_edge, fpack)

    def extract_parts_symbol(self):
        """
        Extracts the circuit components used in all netlists and saves
        them in the list(type_of_symbol).
        """
        symbol = []
        files = list(self.save_dir03.glob("*"))

        for file in files:
            with open(file, "rb") as ufp:
                data = pickle.load(ufp)
                symbol.extend(each_d[0] for each_d in data)

        self.type_of_symbol = sorted(set(symbol))

    def replace_node_num(self):
        """
        Convert the list(type_of_symbol) as an index of the circuit components and
        the nodes into numbers and save them in the "save folder"->05_changeToNodeNum.
        """
        files = list(self.save_dir03.glob("*"))

        for file in files:
            with open(file, "rb") as ufp:
                data = pickle.load(ufp)

            node_one_hot = []
            for each_d in data:
                node_list = [0] * len(self.type_of_symbol)
                index = self.type_of_symbol.index(each_d[0])

                if each_d[0] in {"R", "L", "C"}:
                    d_split = each_d.split(":")
                    node_list[index] = float(d_split[1])
                elif each_d[0] == "X":
                    node_list[index] = 1 - 0.1 * float(each_d[-1])
                else:
                    node_list[index] = 1
                    
                node_one_hot.append(node_list)

            save_path = Path(self.save_dir05) / Path(file).name
            np.save(save_path, node_one_hot)

    def replace_edge_num(self):
        """
        Based on the number of  "05_changeToNodeNum", convert an edge into an edge
        consisting of a combination (tuple) of two numbers, and save it in
        "save folder"->06_changeToEdgeNum.
        """
        files = [str(file) for file in self.save_dir04.glob("*")]
        for file in files:
            with open(file, "rb") as ufp:
                edges = pickle.load(ufp)

            # load node value of created by the method(replace_node_num())
            with open(file.replace("04_extractEdge", "03_extractNode"), "rb") as nfp:
                data_node = pickle.load(nfp)

            # remove decimal value from node  ex) 'C3:0.26086956521739124' -> 'C3'
            data_node = [re.sub(r'(:[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', '', d) for d in data_node]

            out = []
            for edge in edges:
                if edge[0][0] in ['A', 'D', 'E', 'G', 'I', 'J', 'K', 'M', 'Q', 'S', 'T', 'X']:
                    first = data_node.index(edge[0].split(":")[0])
                else:
                    first = data_node.index(edge[0])

                if edge[1][0] in ['A', 'D', 'E', 'G', 'I', 'J', 'K', 'M', 'Q', 'S', 'T', 'X']:
                    second = data_node.index(edge[1].split(":")[0])
                else:
                    second = data_node.index(edge[1])

                taple = (first, second)
                out.append(taple)

            save_path = Path(self.save_dir06) / Path(file).name
            np.save(save_path, out)

    def visualize_graph(self, num=0):
        """
        For verification: Graphicalize the edges created by method(replace_edge_num()).
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        files = [str(file) for file in self.save_dir04.glob("*")]
        file = files[num]

        with open(file, "rb") as ufp:
            edge = pickle.load(ufp)

        graph_nx = nx.Graph()
        fig = plt.figure()
        fig_axes = fig.add_axes([0, 0, 1, 1])
        graph_nx.add_edges_from(edge)
        nx.draw(graph_nx, ax=fig_axes, with_labels=True, node_color="lightblue")
        plt.savefig("graph.png")

    def check_node_num(self):
        """
        For verification: count the number of nodes for each component type.
        """
        files = [str(file) for file in self.save_dir05.glob("*")]

        target_num = ["00", "01", "02", "03", "04", "05", "06"]
        target_name = ["ADC",
                       "Comparator",
                       "Filter Products",
                       "Opamps",
                       "Power Products",
                       "Reference",
                       "Switches"]
        print("\n average nodes of each category:")
        for i, target in enumerate(target_num):
            sum_nodes = 0
            files = list(Path(self.save_dir05).glob(f"{target}*"))
            for file in files:
                data = np.load(file)
                sum_nodes += len(data)
            print(f"{target_name[i]} :{round(sum_nodes/len(files), 2)}")

        sum_nodes_all = 0
        for file in files:
            data = np.load(file)
            sum_nodes_all += len(data)
        print(f"<average nodes>:{sum_nodes_all/len(files)}", end="\n\n")

    def check_edge_num(self):
        """
        For verification: count the number of edges for each component type.
        """
        files = [str(file) for file in self.save_dir06.glob("*")]

        target_num = ["00", "01", "02", "03", "04", "05", "06"]
        target_name = ["ADC",
                       "Comparator",
                       "Filter Products",
                       "Opamps",
                       "Power Products",
                       "Reference",
                       "Switches"]
        print("\n average edges of each category:")
        for i, target in enumerate(target_num):
            sum_edges = 0
            files = list(Path(self.save_dir06).glob(f"{target}*"))
            for file in files:
                data = np.load(file)
                sum_edges += len(data)
            print(f"{target_name[i]} :{round(sum_edges/len(files), 2)}")

        sum_edges_all = 0
        for file in files:
            data = np.load(file)
            sum_edges_all += len(data)
        print(f"<average edges>:{sum_edges_all/len(files)}", end="\n\n")

    def __repr__(self):
        return "type_of_symbol: " + ', '.join(self.type_of_symbol)


if __name__ == '__main__':
    graph = Netlist2Graph(args)
    print(graph)
    graph.extract_node()
    graph.extract_edge()
    # graph.visualize_graph(num=1000)
    graph.extract_parts_symbol()
    graph.replace_node_num()
    graph.replace_edge_num()
    graph.check_node_num()
    graph.check_edge_num()
