python pcnn_train.py \
--batch_size 1 \
--sample_batch_size 16 \
--sampling_interval 10 \
--save_interval 5 \
--dataset cpen455_complex_embeddings \
--nr_resnet 1 \
--nr_filters 1 \
--nr_logistic_mix 1 \
--lr_decay 0.999995 \
--max_epochs 500 \
--en_wandb True \
--tag "large model" \