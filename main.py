
from configuration import config
from datasets import *
from methods.er_baseline import ER
from methods.rainbow_memory import RM
from methods.ewcpp import EWCpp
from methods.lwf import LwF
from methods.mvp import MVP
from methods.GACL import GACL
from methods.muse import MUSE
from methods.inflora import InfLoRA
from methods.SLDA import SLDA

# torch.backends.cudnn.enabled = False

methods = {
    "gacl"            : GACL,
    "muse"            : MUSE,
    "inflora"         : InfLoRA,
    "er"              : ER, 
    "rm"              : RM,
    "lwf"             : LwF,
    "ewc++"           : EWCpp,
    "mvp"             : MVP,
    "SLDA"            : SLDA,
}

def main():
    args = config.base_parser()
    print(args)
    trainer = methods[args.mode](**vars(args))

    trainer.run()

if __name__ == "__main__":
    main()