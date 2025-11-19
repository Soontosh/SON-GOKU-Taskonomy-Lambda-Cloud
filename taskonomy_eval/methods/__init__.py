from .base import METHOD_REGISTRY, MultiTaskMethod, register_method

from .son_goku_method import SonGokuMethod
from .gradnorm_method import GradNormMethod
from .mgda_method import MGDAMethod
from .pcgrad_method import PCGradMethod
from .cagrad_method import CAGradMethod
from .sel_update_method import SelectiveUpdateMethod
from .adatask_method import AdaTaskMethod
from .nashmtl_method import NashMTLMethod
from .fairgrad_method import FairGradMethod
from .famo_method import FAMOMethod 
from .son_goku_gradnorm_method import SonGokuGradNormWarmStartMethod
from .son_goku_adatask_method import SonGokuAdaTaskMethod 
from .son_goku_pcgrad_method import SonGokuPCGradMethod

METHOD_REGISTRY = {
    "son_goku": SonGokuMethod,
    "gradnorm": GradNormMethod,
    "son_goku_gradnorm": SonGokuGradNormWarmStartMethod,
    "son_goku_adatask": SonGokuAdaTaskMethod,
    "son_goku_pcgrad": SonGokuPCGradMethod,
    "mgda": MGDAMethod,
    "pcgrad": PCGradMethod,
    "cagrad": CAGradMethod,
    "adatask": AdaTaskMethod,
    "sel_update": SelectiveUpdateMethod,
    "nashmtl": NashMTLMethod,
    "fairgrad": FairGradMethod,
    "famo": FAMOMethod,
}