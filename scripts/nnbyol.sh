python3 main.py \
    --domain_file data/pddl/grid/grid.pddl \
    --problem_file data/pddl/grid/problems/grid1.pddl \
    --backbone resnet18 \
    --max_epochs 30 \
    --devices 1 \
    --accelerator gpu \
    --precision 16 \
    --optimizer lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm_lars \
    --scheduler warmup_cosine \
    --lr 1.0 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 256 \
    --num_workers 4 \
    --name nnbyol-$1 \
    --project solo-learn \
    --wandb \
    --method nnbyol \
    --proj_output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --momentum_classifier