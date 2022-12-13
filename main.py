import os
from pddl_vis.utils import load_args, clustering_test
from pddl_vis.dataset import PDDLDataset, prepare_dataloader, get_domain, visualize_traces
from pddl_vis.aligning import bnb_align, bnb_neighbours_align, greedy_align, get_graph_and_traces, get_predictions

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from solo.utils.misc import make_contiguous
from solo.methods import METHODS
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer

import numpy as np
import wandb


def main():

    # Closely resembles main_pretrain.py from solo-learn

    seed = 0

    args = load_args()

    domain_file = 'data/pddl/' + args.domain + '/' + args.domain + '.pddl'
    problem_file = 'data/pddl/' + args.domain + '/problems/' + args.problem + '.pddl'

    visualizer, n_states = get_domain(
        domain = args.domain,
        domain_file=domain_file, 
        problem_file=problem_file
    )
    args.num_classes = n_states

    #model = METHODS[args.method].load_from_checkpoint("trained_models/swav/solo-learn/2ev6njzg/checkpoints/epoch=1-step=1566.ckpt", **args.__dict__)
    model = METHODS[args.method](**args.__dict__)
    make_contiguous(model)

    train_data = PDDLDataset(visualizer, n_samples=args.train_samples, img_size=(args.img_h, args.img_w), seed=seed)
    val_data = PDDLDataset(visualizer, n_samples=args.val_samples, img_size=(args.img_h, args.img_w), train=False, seed=seed)
    test_data = PDDLDataset(visualizer, n_samples=args.test_samples, img_size=(args.img_h, args.img_w), train=False, seed=seed)

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
        default_root_dir='trained_models',
        strategy=DDPStrategy(find_unused_parameters=False)
        if args.strategy == "ddp"
        else args.strategy,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    embeddings = []
    labels = []

    for batch in test_loader:
        x, l = batch
        out = model(x)
        embeddings += list(out['feats'].detach().numpy())
        labels += list(l.detach().numpy())
    
    embeddings = np.array(embeddings)    # (data_size, 512)  
    labels = np.array(labels)            # (data_size)

    homogeneity, completeness, v_measure = clustering_test(embeddings, labels, n_states)


    n_traces = 100
    trace_len = 10

    # Get one long trace to break into trace_len sizes
    state_graph, traces, states = get_graph_and_traces(
        domain_file, 
        problem_file, 
        n_traces=1, 
        trace_len=n_traces * trace_len
    )

    # (1, trace_len * n_traces, *img_shape)
    data = visualize_traces(traces, vis=visualizer.visualize_state, img_size=(args.img_h, args.img_w))

    preds, logits = get_predictions(model, data[0], batch_size=args.batch_size)

    # (n_traces, trace_len, n_states)
    preds = preds.reshape((n_traces, trace_len, *preds.shape[1:]))
    logits = logits.reshape((n_traces, trace_len, *logits.shape[1:]))
    states = states.reshape((n_traces, trace_len))


    top_1_accuracy = 100 * np.sum(preds[:, :, 0] == np.array(states)) / states.size

    greedy_accuracy, top_1_in_graph, greedy_in_graph = greedy_align(
        state_graph, 
        states, 
        preds, 
        logits, 
        top_n=int(n_states / 10)
    )

    bnb_accuracy = bnb_align(
        state_graph, 
        states, 
        preds, 
        logits, 
        top_n=5
    )

    bnb_n_accuracy = bnb_neighbours_align(
        state_graph, 
        states, 
        preds, 
        logits, 
        top_n=int(n_states / 10)
    )


    wandb.log({
        'top_1_accuracy'  : top_1_accuracy, 
        'greedy_accuracy' : greedy_accuracy, 
        'bnb_accuracy'    : bnb_accuracy,
        'bnb_n_accuracy'  : bnb_n_accuracy,
        'top_1_in_graph'  : top_1_in_graph, 
        'greedy_in_graph' : greedy_in_graph,
        'homogeneity'     : homogeneity, 
        'completeness'    : completeness, 
        'v_measure'       : v_measure
    })


if __name__ == '__main__':

    main()