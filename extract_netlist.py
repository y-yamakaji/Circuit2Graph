"""Module providing extracting netlist from LTspice folder"""
from pathlib import Path
import re
import pickle

from util import parts_order, parts_normalize, searchL, mutualTwoPhase, newNet_twoPhase, mutualThreePhase, newNet_threePhase, natsorted
from params import args


class LTspice2Netlist():
    """
    Create netlist from LTspice schematic. Assign circuit type information
    for each netlist and save it in a folder.

    Attributes
    ----------
    args : argparse
        params.py

    Methods
    -------
    extract_file_name(self):.
      Extracts the LTspice circuit part format (.asy) from the LTspice parts
      folder and saves the file name in the save folder.

    make_net_batch(self):
      Generates a batch to extract the netlist (.net) from the LTspice circuit
      format (.asc) in the save folder.

    extract_netlist(self):.
      Match the netlist(.net) that matches the circuit part format(.asy) and
      generate a netlist with the φile names with the part numbers assigned and
      the simulation information removed.

    check_extract_netlist(self, folder_num=1, num=0):
      Check the generated netlist.

    unify_description(self):
      Unify the description of the same component, since they are described
      in different ways. ex) current source: I, Is, Ib -> I

    circuit_constant(self):.
      Convert a circuit constant from a true number to a logarithmic netlist.
    """
    def __init__(self, arg) -> None:
        self.args = arg
        dataset_path = Path(self.args.dataset)
        dataset_path.mkdir(parents=True, exist_ok=True)  # save folder

        self.netlistfolder = Path(self.args.ltspice) / 'examples' / 'jigs'
        self.ascfiles = list(self.netlistfolder.glob('*.asc'))
        self.netfiles = list(self.netlistfolder.glob('*.net'))

        # Not used due to small number of data:
        # removeFiles = ['SpecialFunctions', 'Contrib', 'DAC', 'Digital', 'Misc', 'Optos']
        # SpecialFunctions are excluded because some of them have a common structure with other types.
        usefiles = ['ADC',
                    'Comparators',
                    'FilterProducts',
                    'OpAmps',
                    'PowerProducts',
                    'References',
                    'Switches']
        
        self.savefolder01 = dataset_path / "01_extractNetlist"
        self.savefolder02 = dataset_path / "02_circuitConstant"
        self.partsfolder = [Path(self.args.ltspice) / 'lib' / 'sym' / p for p in usefiles]
        self.jigsfolder = self.netlistfolder

        self.savefolder01.mkdir(parents=True, exist_ok=True)
        self.savefolder02.mkdir(parents=True, exist_ok=True)

    def extract_file_name(self) -> None:
        """
        Extracts the LTspice circuit part format (.asy) from the LTspice parts
        folder and saves the file name in the save folder.
        """

        for parts in self.partsfolder:
            asyfile = list(Path(parts).glob('*.asy'))
            name = Path(parts).name + ".txt"
            file_path = Path(self.args.dataset) / name
            with file_path.open("w", encoding='utf-8') as file:
                for asy in asyfile:
                    file.write(asy.stem + ",")

    def make_net_batch(self) -> None:
        """
        Generates a batch to extract the netlist (.net) from the LTspice circuit
        format (.asc) in the save folder.
        """
        batch_file_path = Path(self.args.dataset) / "_make_netlist.bat"
        with batch_file_path.open("w", encoding='utf-8') as file:
            for asc in self.ascfiles:
                asc_name = Path(asc).stem
                if asc_name not in self.netfiles:
                    file.write(f'"{self.args.ltspice}XVIIx64.exe" -netlist "{asc}"\n')
        

    def extract_netlist(self) -> None:
        """
        Match the netlist(.net) that matches the circuit part format(.asy) and
        generate a netlist with the φile names with the part numbers assigned and
        the simulation information removed.
        """
        parts = list(Path(self.args.dataset).glob("*.txt"))

        for n_index, part in enumerate(parts):
            with open(part, "r", encoding='utf-8') as file:
                data = file.readline()  # AD4000,AD4001,AD4002,AD4003,AD4004,...

            for fname in data.split(","):
                fremove = fname.replace("LTC", "").replace("LTM", "").replace("LTZ", "").replace("LT", "").replace("RH", "")

                fpath = None
                if (self.jigsfolder / (fname + ".net")).is_file():
                    fpath = self.jigsfolder / (fname + ".net")
                elif (self.jigsfolder / (fremove + ".net")).is_file():
                    fpath = self.jigsfolder / (fremove + ".net")

                if fpath:
                    try:
                        with open(fpath, "r", encoding='utf-8', errors='ignore') as net_file:
                            net_data = net_file.readlines()

                        target_data = []
                        for each_data in net_data:
                            d_split = each_data.strip()

                            # remove like ".tran 0 100m" in netlist
                            if re.match(r'^[A-Z]', d_split):
                                target_data.append(d_split)

                        if fname:
                            name = f"{str(n_index).zfill(2)}_{fname}"
                            save_path = self.savefolder01 / name
                            with open(save_path, "wb") as fpack:
                                pickle.dump(target_data, fpack)

                    except FileNotFoundError:
                        print(f"FileNotFoundError: {fremove}")

    def check_extract_netlist(self, folder_num=1, num=0) -> None:
        """
        Check the generated netlists.
        """
        if folder_num == 1:
            files = list(self.savefolder01.glob("*"))
        elif folder_num == 2:
            files = list(self.savefolder02.glob("*"))
        else:
            print("folder_num is 1 or 2")
            return

        if num < len(files):
            file = files[num]
            with open(file, "rb") as fpack:
                data = pickle.load(fpack)
            print(data)
        else:
            print(f"Invalid file index: {num}. There are only {len(files)} files in the folder.")

    
    def unify_description(self) -> None:
        """
        Unify the description of the same component, since they are described
        in different ways. ex) current source: I, Is, Ib -> I
        """
        # ex) Unify MQ1 or M1 with Q1
        files = list(self.savefolder01.glob("*"))

        patterns = {
            'M': re.compile(r'(MQ\d+)|(M\d+)'),
            'Q': re.compile(r'(Q\d+)'),
            'B': re.compile(r'(B\S+)'),
            'S': re.compile(r'(S\S+)'),
            'I': re.compile(r'(I\S+)')
        }

        for file in files:
            with open(file, "rb") as fpack:
                data = pickle.load(fpack)

            output = []
            counters = {key: 1 for key in patterns.keys()}

            for each_data in data:
                d_split = each_data.split(" ")
                for key, pattern in patterns.items():
                    if pattern.search(d_split[0]):
                        d_split[0] = f"{key}{counters[key]}"
                        counters[key] += 1
                        break
                output.append(" ".join(d_split))

            if self.args.mutual == "none":
                with open(file, "wb") as fpack:
                    pickle.dump(output, fpack)

            elif self.args.mutual == "equiv":
                saveData = self._process_equiv(output)
                with open(file, "wb") as fpack:
                    pickle.dump(saveData, fpack)

    def _process_equiv(self, output):
        saveData = []
        listed_L = []

        for each_line in output:
            if re.compile('[K]').search(each_line[0]):
                d = each_line.split(" ")
                if len(d) == 4:  # single phase
                    saveData.extend(self._process_two_phase(d, output, listed_L))
                elif len(d) == 5:  # three phase
                    saveData.extend(self._process_three_phase(d, output, listed_L))

        for each_line in output:
            if re.compile('[L]').search(each_line[0]):
                d = each_line.split(" ")
                if d[0] not in listed_L:
                    saveData.append(each_line)
            elif re.compile('[ABCDEFGHIJMNOPQRSTUVWXYZ]').search(each_line[0]):
                saveData.append(each_line)

        return saveData

    def _process_two_phase(self, d, output, listed_L):
        L_list = searchL(targetList=[d[1], d[2]], data=output)
        listed_L.extend([d[1], d[2]])
        z1, z2, z3 = mutualTwoPhase(L_list, K=d[3])
        return newNet_twoPhase(output, z1, z2, z3)

    def _process_three_phase(self, d, output, listed_L):
        L_list = searchL(targetList=[d[1], d[2], d[3]], data=output)
        listed_L.extend([d[1], d[2], d[3]])
        z1, z2, z3, z12, z23, z13 = mutualThreePhase(L_list, K1=d[4], K2=d[4], K3=d[4])
        return newNet_threePhase(output, z1, z2, z3, z12, z23, z13)

    def circuit_constant(self) -> None:
        files = list(self.savefolder01.glob("*"))

        pattern1 = re.compile('[CFHJLNOPRUWYZ]')
        pattern2 = re.compile('[ABDMVTX]')
        decimal_pattern = re.compile(r'(?:\d+\.?\d*|\.\d+)')

        for file in files:
            with open(file, "rb") as fpack:
                data = pickle.load(fpack)

            save_data = []
            for each_line in data:
                if pattern1.search(each_line[0]):
                    d_split = each_line.split(" ")

                    if (decimal_pattern.search(d_split[3]) and 
                        "{" not in d_split[3] and "exp" not in d_split[3]):
                        
                        parts_data = f"{d_split[0]} {d_split[1]} {d_split[2]}"
                        constant_data = str(parts_normalize(*parts_order(d_split)))
                        save_data.append(f"{parts_data} {constant_data}")

                elif pattern2.search(each_line[0]):
                    d_split = each_line.split(" ")
                    out = " ".join(dsp for dsp in d_split if "=" not in dsp)
                    save_data.append(out.strip())

                else:
                    save_data.append(each_line)

            if self.args.semiconductor == "single":
                self._save_to_file(file.name, save_data)

            elif self.args.semiconductor == "star":
                sdata = self._process_star_semiconductor(save_data)
                self._save_to_file(file.name, sdata)

            elif self.args.semiconductor == "complete":
                sdata = self._process_complete_semiconductor(save_data)
                self._save_to_file(file.name, sdata)


    def _save_to_file(self, filename: str, data: list) -> None:
        save_path = self.savefolder02 / filename
        with open(save_path, "wb") as fpack:
            pickle.dump(data, fpack)

    def _process_star_semiconductor(self, save_data: list) -> list:
        sdata = []
        targetData_except_XU = []
        pattern_RLC = re.compile(r'[RLC]')
        pattern_AJMQSTX = re.compile(r'[AJMQSTX]')
        pattern_MP = re.compile(r'MP_\d{1,}')

        for line in save_data:
            parts = line.split()
            if pattern_RLC.search(line[0]):
                sdata.append(' '.join(parts[:4]))
            elif pattern_AJMQSTX.search(parts[0][0]):
                parts = [p for p in parts if '=' not in p]
                add_nodes = []
                for i, p in enumerate(parts[1:-1]):
                    if not pattern_MP.search(p):
                        sdata.append(f"{parts[0]}:{i} {p} {parts[0]}:{i}-0")
                        add_nodes.append(f"{parts[0]}:{i}-0")
                sdata.append(f"{parts[0]} {' '.join(natsorted(set(add_nodes)))}")
            else:
                targetData_except_XU.append(' '.join(parts[:3]))

        sdata = natsorted(set(sdata))
        sdata.extend(targetData_except_XU)
        return sdata

    def _process_complete_semiconductor(self, save_data: list) -> list:
        sdata = []
        node_pattern = re.compile(r'[AEIJMQSTX]')
        mp_pattern = re.compile(r'MP_\d{1,}')

        for line in save_data:
            parts = line.split()
            if node_pattern.search(parts[0][0]):
                parts = [p for p in parts if '=' not in p]
                if len(parts) > 3:
                    end = -2 if parts[-1] == parts[-2] else -1
                    for i, p1 in enumerate(parts[1:end]):
                        if not mp_pattern.search(p1):
                            for j, p2 in enumerate(parts[1:end]):
                                if not mp_pattern.search(p2) and i != j:
                                    a, b = min(i, j), max(i, j)
                                    sdata.append(f"{parts[0]}:{i} {p1} {parts[0]}:{a}-{b}")
            else:
                sdata.append(line)

        return sdata

    def __repr__(self):
        output1 = "number of asc files: "+str(len(self.ascfiles))
        output2 = ", number of net files: "+str(len(self.netfiles))
        output3 = "\nRegistered Components:\n " + "\n ".join(str(p) for p in self.partsfolder)
        return output1 + output2 + output3

if __name__ == '__main__':
    LTnet = LTspice2Netlist(args)
    print(LTnet)
    LTnet.make_net_batch()
    LTnet.extract_file_name()
    LTnet.extract_netlist()
    LTnet.unify_description()
    LTnet.check_extract_netlist(folder_num=1, num=0)
    LTnet.circuit_constant()
    LTnet.check_extract_netlist(folder_num=2, num=0)