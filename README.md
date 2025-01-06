# Circuit2Graph
This source code is an implementation of graph classification in a graph neural network using pytorch(CPU/GPU), by transforming the analog/digital circuits described in the papers into a graph network.

Yusuke Yamakaji, Hayaru Shouno and Kunihiko Fukushima, [Circuit2Graph: Circuits With Graph Neural Networks](https://ieeexplore.ieee.org/document/10494518) in (IEEE Access vol. 12, pp. 51818-51827, 2024)

Yusuke Yamakaji, Hayaru Shouno and Kunihiko Fukushima, [Circuit2Graph: Diodes as Asymmetric Directional Nodes](https://ieeexplore.ieee.org/document/10752500) in (IEEE Access, vol. 12, pp. 168963-168974, 2024)

Yusuke Yamakaji, Hayaru Shouno and Kunihiko Fukushima, [Equivalent Circuit for Single/Three Phase Magnetic Coupling With Graph Neural Networks](https://ieeexplore.ieee.org/document/10731705) in (IEEE Transactions on Power Electronics, vol. 40, no. 2, pp. 3313-3325, Feb. 2025)

## Environment Setup
Assume that LTspice XVII (https://www.analog.com/jp/resources/design-tools-and-calculators/ltspice-simulator.html) is installed in the "C:\LTspice" directory. If the installation path is different, please modify the "default=os.path.join('C:', 'LTspice')" in `params.py`. Additionally, LTspice XVII may have files like `examples.zip` and `lib.zip` in the "C:\LTspice" directory. Please unzip these files beforehand so that "C:\LTspice\examples" and "C:\LTspice\lib" are accessible.

## For GPU environment
1. Install the NVIDIA driver and CUDA Toolkit according to your environment. In this explanation, the version of the CUDA Toolkit is 11.8.

2. Install the required libraries:
    ```sh
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    pip install numpy torch_geometric
    (optional) pip install matplotlib networkx 
    ```

This code has been tested using the following versions of the library:
- numpy: 2.2.1
- torch: 2.5.1
- torch_geometric: 2.6.1
- (optional)matplotlib: 3.10.0
- (optional)networkx: 3.4.2

## For CPU environment
Compared to a GPU environment, the calculation time is slower, but the code is designed to allow training and inference to be carried out in a CPU environment.

1. Install the required libraries:
```sh
pip install numpy torch torch_geometric
(optional) pip install matplotlib networkx 
```

### Execution
When you run this code for the first time, you need to run the batch file `_make_netlist.bat` created by the `make_net_batch` function in `extract_netlist.py` in order to extract netlists from sample circuits of LTspice.

To execute, run the following scripts in order:
```sh
python extract_netlist.py
python extract_node_edge.py
python data_loading.py
python train.py
```