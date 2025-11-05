from .base import METHOD_REGISTRY, MultiTaskMethod, register_method

from .son_goku_method import SonGokuMethod
from .gradnorm_method import GradNormMethod
from .mgda_method import MGDAMethod  # if you already added MGDA
from .pcgrad_method import PCGradMethod  # newly added PCGrad
from .cagrad_method import CAGradMethod  # if you have CAGrad implemented
from .sel_update_method import SelectiveUpdateMethod
from adatask_method import AdaTaskMethod

METHOD_REGISTRY = {
    "son_goku": SonGokuMethod,
    "gradnorm": GradNormMethod,
    "mgda": MGDAMethod,
    "pcgrad": PCGradMethod,
    "cagrad": CAGradMethod,
    "adatask": AdaTaskMethod,
    "sel_update": SelectiveUpdateMethod,
}