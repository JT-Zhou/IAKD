from ._base import Vanilla,Distiller
from .KD import KD
from .AT import AT
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT import PKT
from .SP import SP
from .VID import VID
from .IAKD import IAKD
from .DKD import DKD
from .KD import KD
from .WSL import WSL
from .ReviewKD import ReviewKD
from .AN import AN

distiller_dict = {
    "D":Distiller,
    "NONE": Vanilla,
    "KD": KD,
    "AT": AT,
    "OFD": OFD,
    "RKD": RKD,
    "FITNET": FitNet,
    "KDSVD": KDSVD,
    "CRD": CRD,
    "NST": NST,
    "PKT": PKT,
    "SP": SP,
    "VID": VID,
    "IAKD": IAKD,
    "DKD": DKD,
    "KD":KD,
    "WSL":WSL,
    "REVIEWKD":ReviewKD,
    "AN":AN,
}
