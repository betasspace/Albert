python run_classifier.py \
	--arch albert_large \
	--albert_config_path configs/albert_config_large.json \
	--bert_dir pretrain/pytorch/albert_large_zh \
	--train_batch_size 24 \
	--num_train_epochs 10 \
	--do_train 
