import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from fxpmath import Fxp


def inject_error(x, p):
    y = x.clone()
    for i in range(len(x)):
        y[i] = torch.tensor(inject_error_C(x[i].numpy(), p))
    return y, None


def downscale(X):
    for i, x in enumerate(X):
        down = Fxp(x.item(), True, 8, 4)
        X[i] = down
    return X


def N_bits(X):
    X = X.astype(np.int8)
    return X


def inject_error_C(x, p):
    faultValue = Fxp(x, True, 32, 16)
    b = faultValue.bin()
    b = list(map(list, b))
    Index = np.random.randint(0, int(1 / p), (len(x), 32))
    b = list(map(lambda sublist: list(map(int, sublist)), b))
    b = np.array(b).reshape(len(x), 32)
    b[Index == 0] = 1 - b[Index == 0]
    b = list(map(lambda sublist: "".join(map(str, sublist)), b))
    b = list(map(lambda s: f"0b{s}", b))
    faultValue = Fxp(b, True, 32, 16).get_val()
    return faultValue.reshape(len(x))


def inject_error_N(x, p):
    faultValue = Fxp(x.item(), True, 32, 16)
    b = faultValue.bin()
    b = list(b)
    for s in range(32):
        if random.randint(0, int(1 / p)) == 0:
            b[s] = "0" if b[s] == "1" else "1"
    b = "".join(b)
    faultValue = Fxp(f"0b{b}", True, 32, 16).get_val()
    return faultValue


def error_injection_checksum(
    checksum: np.ndarray, p: float, mini: float, maxi: float, N_tmr: int = 3
) -> np.ndarray:
    """
    Inject errors into checksum array and return the clipped median.

    Parameters:
    - checksum (np.ndarray): Input array of checksum values.
    - p (float): Probability of error injection.
    - mini (float): Minimum value for clipping.
    - maxi (float): Maximum value for clipping.

    Returns:
    - np.ndarray: Resulting array after injecting errors and computing the clipped median.
    """
    m1 = np.zeros((len(checksum), N_tmr))
    for i in range(N_tmr):
        m1[:, i] = inject_error_C(checksum, p)
    m = np.median(m1, axis=-1)
    return np.clip(m, mini, maxi)


def error_injection_clusters(
    D: Dict[str, Dict[str, np.ndarray]], p: float, N_tmr: int = 3, do_clip: bool = True
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Inject errors into a dictionary of arrays and return the modified dictionary.

    Parameters:
    - D (Dict[str, Dict[str, np.ndarray]]): Input dictionary with keys representing elements and values containing 's' and 'm' arrays.
    - p (float): Probability of error injection.
    - N_tmr (int, optional): Number of iterations for error injection. Default is 3.
    - do_clip (bool, optional): Whether to clip the modified arrays. Default is True.

    Returns:
    - Dict[str, Dict[str, np.ndarray]]: Resulting dictionary after injecting errors and optionally clipping the arrays.
    """
    for k in D.keys():
        element = D[k]
        S = element["s"].astype(np.float32)
        M = element["m"].astype(np.float32)

        min_S = S.min()
        min_M = M.min()
        max_S = S.max()
        max_M = M.max()
        m = np.zeros((len(M), 3))
        for i in range(N_tmr):
            m[:, i] = inject_error_C(M, p)

        m = np.median(m, axis=-1)
        M = np.clip(m, min_M, max_M)

        s = np.zeros((len(S), 3))
        for i in range(N_tmr):
            s[:, i] = inject_error_C(S, p)

        s = np.median(s, axis=-1)
        S = np.clip(s, min_S, max_S)

        if do_clip:
            S = np.clip(S, min_S, max_S)
            M = np.clip(M, min_M, max_M)

        element["s"] = S
        element["m"] = M
        D[k] = element

    return D


def look(i: str, keys: List[Tuple]) -> Optional[Tuple]:
    """
    Find the first tuple in a list of keys that contains a specified substring.

    Parameters:
    - i (str): Substring to search for within the list of keys.
    - keys (List[Tuple]): List of tuples to search through.

    Returns:
    - Optional[Tuple]: The first tuple in the list that contains the specified substring 'i',
      or None if no matching tuple is found.
    """
    for j in keys:
        if i in j:
            return j
    return None
