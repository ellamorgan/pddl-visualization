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
    --optimizer lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --scheduler warmup_cosine \
    --lr 0.6 \
    --min_lr 0.0006 \
    --warmup_start_lr 0.0 \
    --warmup_epochs 11 \
    --classifier_lr 0.1 \
    --weight_decay 1e-6 \
    --batch_size $4 \
    --num_workers 4 \
    --name deepclusterv2-$1-$2-$3-$4 \
    --project solo-learn \
    --wandb \
    --method deepclusterv2 \
    --proj_hidden_dim 2048 \
    --proj_output_dim 128 \
    --num_prototypes 3000 3000 3000 \
    --save_checkpoint
