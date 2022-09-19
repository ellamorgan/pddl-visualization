import os
from pddl_vis.utils import load_args
from pddl_vis.pddl import PDDLDataset, prepare_dataloader, GridVisualizer

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from solo.utils.misc import make_contiguous
from solo.methods import METHODS
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer

from macq.generate.pddl import StateEnumerator


def main():

    # Closely resembles main_pretrain.py from solo-learn

    args = load_args(n_states=20)

    model = METHODS[args.method](**args.__dict__)
    make_contiguous(model)

    # This should be a function that gets the right visualization depending on the domain
    generator = StateEnumerator(dom=args.domain_file, prob=args.problem_file)
    vis = GridVisualizer(generator, square_width=50, div_width=1, door_width=6, key_size=15, robot_size=17)
    dataset = PDDLDataset(generator, vis.visualize_state, n_samples=5, img_size=(24,24))
    train_loader = prepare_dataloader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

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

    trainer.fit(model, train_loader, ckpt_path=ckpt_path)


if __name__ == '__main__':

    # Doesn't run yet!
    main()