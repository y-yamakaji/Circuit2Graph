"""Module provide various useful methods"""
import math
import re
import random
from typing import List, Tuple, Optional

import numpy as np
import torch


def fix_seed(seed=1234) -> None:
    '''
    For Fix random numbers
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    
def parts_order(parts: List[str]) -> Tuple[str, str]:
    component = parts[0][0]
    value_str = re.sub(r"[^\d]*$", "", parts[3])
    
    unit_multipliers = {
        'p': 1e-12,
        'n': 1e-9,
        'u': 1e-6,
        'm': 1e-3,
        'k': 1e3,
        'K': 1e3,
        'M': 1e6,
        'G': 1e9
    }
    
    try:
        value = float(value_str)
        is_negative = value < 0
        value = abs(value)
        
        unit = next((s for s in parts[3] if s in unit_multipliers), '')
        multiplier = unit_multipliers.get(unit, 1)
        exp = math.log10(value * multiplier)
        
        if value == 0 or parts[3] in {"0", "0.0", ".0"}:
            exp = 0.00001
        
        if is_negative:
            return (component, f"minus:{exp}")
        else:
            return (component, str(exp))
    except ValueError:
        return ('G', '1')
        
def parts_order_linear(string: str) -> float:
    s = re.sub(r"[^\d]*$", "", string)
    unit_multipliers = {
        'p': 1e-12,
        'n': 1e-9,
        'u': 1e-6,
        'm': 1e-3,
        'k': 1e3,
        'K': 1e3,
        'M': 1e6,
        'G': 1e9
    }
    if string in {"0", "0.0", ".0"}:
        return 0
    for unit, multiplier in unit_multipliers.items():
        if unit in string:
            return float(s) * multiplier
    return float(s)
    
def parts_order_linear2(string: str) -> float:
    string = string.replace("{", "").replace("}", "")
    d = string.split("*")
    if d == 1:
        return parts_order_linear(string)
    else:
        out = 0
        for i in range(0, len(d)):
            if i == 0:
                out = parts_order_linear(d[0])
            else:
                out *= parts_order_linear(d[i])
        return out
    
def searchL(targetList: List[str], data: List[str]) -> List[str]:
    compiled_patterns = [re.compile(re.escape(target) + "$") for target in targetList]
    
    result_list = []
    for each_line in data:
        d = each_line.split(" ")
        for pattern in compiled_patterns:
            if pattern.search(d[0]):
                result_list.append(each_line)
                break
    return result_list


def mutualTwoPhase(targetList: List[str], K: float = 0.999) -> List[float]:
    assert len(targetList) == 2, "targetList must contain exactly two elements"
    K = 0.9999 if float(K) == 1.0 else K
    L1 = parts_order_linear2(targetList[0].split(" ")[3])
    L2 = parts_order_linear2(targetList[1].split(" ")[3])
    M = float(K) * math.sqrt(L1 * L2)
    numerator = L1 * L2 - M**2
    z1 = numerator / (L2 - M)
    z2 = numerator / (2 * M)
    z3 = numerator / (L1 - M)
    return [z1, z2, z3]


def mutualThreePhase(targetList: List[str], K1: float = 0.999, K2: float = 0.999, K3: float = 0.999) -> List[float]:
    assert len(targetList) == 3, "targetList must contain exactly three elements"
    K1 = 0.9999 if float(K1) == 1.0 else K1
    K2 = 0.9999 if float(K2) == 1.0 else K2
    K3 = 0.9999 if float(K3) == 1.0 else K3

    L1 = parts_order_linear2(targetList[0].split(" ")[3])
    L2 = parts_order_linear2(targetList[1].split(" ")[3])
    L3 = parts_order_linear2(targetList[2].split(" ")[3])
    M12 = float(K1) * math.sqrt(L1 * L2)
    M23 = float(K2) * math.sqrt(L2 * L3)
    M13 = float(K3) * math.sqrt(L1 * L3)
    numerator = -L1 * L2 * L3 + L1 * M23**2 + L2 * M13**2 + L3 * M12**2 - 2 * M12 * M23 * M13
    z1 = numerator / (-L2 * L3 + L2 * M13 + L3 * M12 - M12 * M23 - M13 * M23 + M23**2)
    z2 = numerator / (-L1 * L3 + L1 * M23 + L3 * M12 - M12 * M13 - M13 * M23 + M13**2)
    z3 = numerator / (-L1 * L2 + L1 * M23 + L2 * M13 - M12 * M13 - M12 * M23 + M12**2)
    z12 = numerator / (2 * (-L3 * M12 + M13 * M23))
    z23 = numerator / (2 * (-L2 * M13 + M12 * M23))
    z13 = numerator / (2 * (-L1 * M23 + M12 * M13))
    return [z1, z2, z3, z12, z23, z13]

def newNet_twoPhase(targetList: List[str], z1: float, z2: float, z3: float) -> List[str]:
    d1 = targetList[0].split(" ")
    d2 = targetList[1].split(" ")
    combined_label = f"L0{d1[0].replace('L', '')}{d2[0].replace('L', '')}"
    L1 = f"{d1[0]} {d1[1]} {d1[2]} {z1}"
    L2 = f"{d2[0]} {d2[1]} {d2[2]} {z3}"
    L3 = f"{combined_label} {d1[1]} {d2[1]} {z2}"
    L4 = f"{combined_label} {d2[1]} {d1[1]} {z2}"
    return [L1, L2, L3, L4]

def newNet_threePhase(targetList: List[str], z1: float, z2: float, z3: float, z12: float, z23: float, z13: float) -> List[str]:
    d1 = targetList[0].split(" ")
    d2 = targetList[1].split(" ")
    d3 = targetList[2].split(" ")
    combined_label_12 = f"L0{d1[0].replace('L', '')}{d2[0].replace('L', '')}"
    combined_label_23 = f"L0{d2[0].replace('L', '')}{d3[0].replace('L', '')}"
    combined_label_13 = f"L0{d1[0].replace('L', '')}{d3[0].replace('L', '')}"
    L1 = f"{d1[0]} {d1[1]} {d1[2]} {z1}"
    L2 = f"{d2[0]} {d2[1]} {d2[2]} {z2}"
    L3 = f"{d3[0]} {d3[1]} {d3[2]} {z3}"
    L4 = f"{combined_label_12} {d1[1]} {d2[1]} {z12}"
    L5 = f"{combined_label_12} {d2[1]} {d1[1]} {z12}"
    L6 = f"{combined_label_23} {d2[1]} {d3[1]} {z23}"
    L7 = f"{combined_label_23} {d3[1]} {d2[1]} {z23}"
    L8 = f"{combined_label_13} {d1[1]} {d3[1]} {z13}"
    L9 = f"{combined_label_13} {d3[1]} {d1[1]} {z13}"

    return [L1, L2, L3, L4, L5, L6, L7, L8, L9]


def normalize_value(value: str, offset: float) -> float:
    if value.startswith("minus:"):
        return -(float(value.replace("minus:", "")) * 0.1 + offset)
    else:
        return float(value) * 0.1 + offset

def parts_normalize(component: str, value: str) -> float:
    offsets = {
        'L': 0.2,
        'C': 1.4,
        'R': 0.4
    }
    offset = offsets.get(component, 0.1)
    return normalize_value(value, offset)
 
def natural_sort_key(item: str) -> List[str]:
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split('([0-9]+)', item)]

def natsorted(items: List[str]) -> List[str]:
    return sorted(items, key=natural_sort_key)

def scatter_max(src: torch.Tensor, 
                index: torch.Tensor, 
                dim: int = -1, 
                out: Optional[torch.Tensor] = None, 
                dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_max(src, index, dim, out, dim_size)

def accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
