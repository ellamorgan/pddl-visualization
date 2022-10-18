python3 main.py \
    --domain_file data/pddl/grid.pddl \
    --problem_file data/pddl/grid_data.pddl \
    --backbone resnet18 \
    --max_epochs 5 \
    --devices 1 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm_lars \
    --scheduler warmup_cosine \
    --scheduler_interval epoch \
    --lr 0.6 \
    --min_lr 0.0006 \
    --classifier_lr 0.1 \
    --weight_decay 1e-6 \
    --batch_size 5 \
    --num_workers 4 \
    --name swav \
    --project solo-learn \
    --wandb \
    --save_checkpoint \
    --auto_resume \
    --method swav \
    --proj_hidden_dim 2048 \
    --queue_size 3840 \
    --proj_output_dim 128 \
    --num_prototypes 3000 \
    --epoch_queue_starts 50 \
    --freeze_prototypes_epochs 2