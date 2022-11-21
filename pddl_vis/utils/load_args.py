import argparse
import pytorch_lightning as pl
from solo.utils.checkpointer import Checkpointer
from solo.utils.auto_resumer import AutoResumer
from solo.methods import METHODS

try:
    from solo.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True



def load_args():
    '''
    Taken from parse_args_pretrain() in solo.setup
    Needed to alter to not load in dataset/augmentation args
    '''
    parser = argparse.ArgumentParser()

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # add method-specific arguments
    parser.add_argument("--method", type=str)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add model specific args
    parser = METHODS[temp_args.method].add_model_specific_args(parser)

    # add pddl domain file and problem file args
    parser.add_argument("--domain", type=str)
    parser.add_argument("--problem", type=str)

    parser.add_argument("--train_samples", type=int)
    parser.add_argument("--val_samples", type=int)
    parser.add_argument("--test_samples", type=int)

    parser.add_argument("--img_h", type=int)
    parser.add_argument("--img_w", type=int)

    parser.add_argument("--trace_len", type=int, default=0)

    # add auto checkpoint/umap args
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--auto_umap", action="store_true")
    parser.add_argument("--auto_resume", action="store_true")
    temp_args, _ = parser.parse_known_args()

    # optionally add checkpointer and AutoUMAP args
    if temp_args.save_checkpoint:
        parser = Checkpointer.add_checkpointer_args(parser)

    if _umap_available and temp_args.auto_umap:
        parser = AutoUMAP.add_auto_umap_args(parser)

    if temp_args.auto_resume:
        parser = AutoResumer.add_autoresumer_args(parser)
    
    # parse args
    args = parser.parse_args()

    # cifar = True adjusts the backbone to handle lower resolution images (i.e. 32x32 instead of 224x224)
    args.backbone_args = {"cifar" : True, "zero_init_residual" : args.zero_init_residual}

    args.extra_optimizer_args = {}
    if args.optimizer == "sgd":
        args.extra_optimizer_args["momentum"] = 0.9
    elif args.optimizer == "lars":
        args.extra_optimizer_args["momentum"] = 0.9
        args.extra_optimizer_args["eta"] = args.eta_lars
        args.extra_optimizer_args["clip_lars_lr"] = args.grad_clip_lars
        args.extra_optimizer_args["exclude_bias_n_norm"] = args.exclude_bias_n_norm_lars
    elif args.optimizer == "adamw":
        args.extra_optimizer_args["betas"] = [args.adamw_beta1, args.adamw_beta2]

    # Seems like these are the numbers of samples of each class (num_small_crops + num_large_crops)
    # Not sure what the difference is between a small and large crop, they do treat them differently
    # I think large_crops are the first 'num_large_crops' images given, rest are small
    # I.e. large_crop_images = images[:num_large_crops]
    args.num_large_crops = 2
    args.num_small_crops = 0

    return args