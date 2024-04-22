python pcnn_train.py \
--batch_size 16 \
--sample_batch_size 16 \
--sampling_interval 100 \
--save_interval 5 \
--dataset cpen455_multi_complex_embeddings_3 \
--nr_resnet 4 \
--nr_filters 100 \
--nr_logistic_mix 8 \
--lr_decay 0.999995 \
--max_epochs 50 \
--en_wandb True \
--tag "large model" \