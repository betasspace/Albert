python convert_albert_tf_checkpoint_to_pytorch.py \
	--tf_checkpoint_path ~/tmp/albert_large_zh/ \
	--bert_config_file configs/albert_config_large.json \
	--pytorch_dump_path pretrain/pytorch/albert_large_zh/pytorch_model.bin

