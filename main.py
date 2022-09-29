import os
from pddl_vis.utils import load_args
from pddl_vis.dataset import PDDLDataset, prepare_dataloader, get_domain

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from solo.utils.misc import make_contiguous
from solo.methods import METHODS
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer

from experiments.val_vis_test import predict_vis_trace



def main():

    # Closely resembles main_pretrain.py from solo-learn

    args = load_args(n_states=20)

    model = METHODS[args.method](**args.__dict__)
    make_contiguous(model)

    # Add these to args
    vis_args = {
        'square_width': 50,
        'div_width': 1,
        'door_width': 6,
        'key_size': 15,
        'robot_size': 17,
    }

    train_samples=10
    val_samples=3
    test_samples=3

    generator, vis = get_domain(
        domain_file=args.domain_file, 
        problem_file=args.problem_file, 
        vis_args=vis_args
    )

    train_data = PDDLDataset(generator, vis, n_samples=train_samples, img_size=(24,24))
    val_data = PDDLDataset(generator, vis, n_samples=val_samples, img_size=(24,24), train=False)
    test_data = PDDLDataset(generator, vis, n_samples=test_samples, img_size=(24,24), train=False)

    train_loader, val_loader, test_loader = prepare_dataloader(
        train_dataset=train_data, 
        val_dataset=val_data, 
        test_dataset=test_data, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    ckpt_path, wandb_run_id = None, None
    if args.auto_resume and args.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(args.checkpoint_dir, args.method),
            max_hours=args.auto_resumer_max_hours,
        )
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(args)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            ckpt_path = resume_from_checkpoint
    elif args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        del args.resume_from_checkpoint

    callbacks = []

    if args.save_checkpoint:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, args.method),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)
    
    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name,
            project=args.project,
            offline=args.offline,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        enable_checkpointing=False,
        strategy=DDPStrategy(find_unused_parameters=False)
        if args.strategy == "ddp"
        else args.strategy,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    for batch in test_loader:
        predict_vis_trace(batch, generator, model)



if __name__ == '__main__':

    main()