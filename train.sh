python train.py \
  --log_path data/tinyzero_qwen3_log \
  --lora_config lora_config.json \
  --model_path /mnt/data/litechainrl/models/Qwen3-8B/ \
  --lora_path data/tinyzero_qwen3_lora \
  --epsilon 0.2 \
  --beta 0.001 \
  --ds_config ds_config.json \
  --master_port 14514 \
  --trainer_gpu 7 \
  --sampling_batch 8 \
  --vllm_gpu 6 \
  --gpu_memory_utilization 0.4 \
  --base_seed 114514 \
  --max_model_len 2048 \
  --max_token_per_turn 2048 \
  --update_batch 512
# --qlora \
