from ._base import Vanilla
from .KD import KD
from .AT import AT
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT_n import PKT
from .SP import SP
from .VID_IA import VID
from .IAKD import IAKD
from .DKD import DKD
from .KD_n import KD
from .WSL import WSL
from .ICKD import ICKD
from .ReviewKD1 import ReviewKD
from .AN import AN
from .SimKD import SimKD

distiller_dict = {
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
    "ICKD":ICKD,
    "REVIEWKD":ReviewKD,
    "AN":AN,
    "SimKD":SimKD
}
