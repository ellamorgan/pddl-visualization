python3 main.py \
    --domain $1 \
    --problem $2 \
    --train_samples 20 \
    --val_samples 5 \
    --test_samples 5 \
    --img_h 48 \
    --img_w 48 \
    --backbone resnet18 \
    --max_epochs 5 \
    --devices 1 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.05 \
    --classifier_lr 0.1 \
    --weight_decay 1e-4 \
    --batch_size 256 \
    --num_workers 4 \
    --name ressl-$1-$2 \
    --project solo-learn \
    --wandb \
    --save_checkpoint \
    --auto_resume \
    --method ressl \
    --proj_output_dim 256 \
    --proj_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --momentum_classifier
