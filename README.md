# Circuit2Graph
This repository contains the source code for implementing graph classification in a graph neural network using PyTorch (CPU/GPU). The code transforms analog/digital circuits described in the papers into a graph network.

## Publications
- Yusuke Yamakaji, Hayaru Shouno, and Kunihiko Fukushima, [Circuit2Graph: Circuits With Graph Neural Networks](https://ieeexplore.ieee.org/document/10494518), IEEE Access, vol. 12, pp. 51818-51827, 2024.
- Yusuke Yamakaji, Hayaru Shouno, and Kunihiko Fukushima, [Circuit2Graph: Diodes as Asymmetric Directional Nodes](https://ieeexplore.ieee.org/document/10752500), IEEE Access, vol. 12, pp. 168963-168974, 2024.
- Yusuke Yamakaji, Hayaru Shouno, and Kunihiko Fukushima, [Equivalent Circuit for Single/Three Phase Magnetic Coupling With Graph Neural Networks](https://ieeexplore.ieee.org/document/10731705), IEEE Transactions on Power Electronics, vol. 40, no. 2, pp. 3313-3325, Feb. 2025.

## Environment Setup
Assume that LTspice XVII  is installed in the "C:\LTspice" directory. If the installation path is different, please modify the "default=os.path.join('C:', 'LTspice')" in `params.py`. Additionally, LTspice XVII may have files like `examples.zip` and `lib.zip` in the "C:\LTspice" directory. Please unzip these files beforehand so that "C:\LTspice\examples" and "C:\LTspice\lib" are accessible.

Ensure that [LTspice XVII](https://www.analog.com/jp/resources/design-tools-and-calculators/ltspice-simulator.html) is installed in the `C:\LTspice` directory. If the installation path is different, modify the `default=os.path.join('C:', 'LTspice')` in `params.py`. Additionally, unzip `examples.zip` and `lib.zip` in the `C:\LTspice` directory so that `C:\LTspice\examples` and `C:\LTspice\lib` are accessible.


## For GPU environment
1. Install the NVIDIA driver and CUDA Toolkit according to your environment. This guide uses CUDA Toolkit version 11.8.

2. Install the required libraries:
    ```sh
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    pip install numpy torch_geometric
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
    (optional) pip install matplotlib networkx 
    ```
    
This code has been tested with the following library versions:
- numpy: 2.2.1
- torch: 2.5.1
- torch_geometric: 2.6.1
- torch_scatter: 2.1.2
- matplotlib: 3.10.0 (optional)
- networkx: 3.4.2 (optional)

## For CPU environment
The calculation time is slower compared to a GPU environment, but the code supports training and inference on a CPU.

1. Install the required libraries:
    ```sh
    pip install numpy torch torch_geometric
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
    # Optional
    pip install matplotlib networkx 
    ```

### Execution
When running this code for the first time, execute the batch file `_make_netlist.bat` created by the `make_net_batch` function in `extract_netlist.py` to extract netlists from sample circuits of LTspice.

To execute, run the following scripts in order:
```sh
python extract_netlist.py --semiconductor star --mutual equiv
python extract_node_edge.py --semiconductor star --mutual equiv
python data_loading.py --semiconductor star --mutual equiv --train_test 0.3
python train.py --semiconductor star --mutual equiv --seed 12 --epoch 30000 --batch 5000
```

## Hyperparameters
The hyperparameters can be adjusted in ``params.py``:
- --semiconductor: Defines how semiconductors are treated. Options are single, star, or complete.
  1. "single": Semiconductors are treated as single nodes.
  2. "star": Semiconductors are treated as a star graph.
  3. "complete": Semiconductors are treated as a complete graph.

- --mutual: Defines how the coupling coefficient of magnetic coupling is handled. Options are equiv or none.
  1. "none": The coupling coefficient is not held.
  2. "equiv": The coupling coefficient is held by transforming it into an equivalent circuit.
