python3 main.py \
    --domain $1 \
    --problem $2 \
    --train_samples 20 \
    --val_samples 5 \
    --test_samples 5 \
    --img_h 48 \
    --img_w 48 \
    --backbone resnet18 \
    --max_epochs $3 \
    --devices 1 \
    --accelerator gpu \
    --precision 16 \
    --num_workers 4 \
    --optimizer lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm_lars \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size $4 \
    --name barlow-$1-$2-$3-$4 \
    --project solo-learn \
    --wandb \
    --method barlow_twins \
    --proj_hidden_dim 2048 \
    --proj_output_dim 2048 \
    --scale_loss 0.1 \
    --save_checkpoint
