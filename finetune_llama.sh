num_machines=1
num_processes=$((num_machines * 8))
machine_rank=0

accelerate launch \
	--config_file sft.yaml \
	--num_processes $num_processes \
	--num_machines $num_machines \
	--machine_rank $machine_rank \
	--deepspeed_multinode_launcher standard finetune_llama.py \
	--model_name_or_path fnlp/llama \
	--output_dir ./ckpts/llama \
	--log_dir ./train_logs/llama \
	--n_epochs 2 \
	--train_bsz_per_gpu 1 \
	--eval_bsz_per_gpu 1 \
	--learning_rate 0.000015 \
	--eval_step 200 \
	--save_step 2000 