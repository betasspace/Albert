#/************************************************************************************
#***
#***	Copyright 2019 Dell(18588220928@163.com), All Rights Reserved.
#***
#***	File Author: Dell, 2019-10-18 13:36:25
#***
#************************************************************************************/
#
#! /bin/sh

usage()
{
	echo "Usage: $0 [options] commands"
	echo "Options:"
	echo "  --base          Convert base model"
	echo "  --large         Convert large model"
	echo "  --xlarge        Convert xlarge model"
	exit 1
}

base_model()
{
	OUTPUT_DIR=pretrain/pytorch/${MODEL}

	python run_classifier.py \
		--arch albert_base \
		--albert_config_path ${OUTPUT_DIR}/albert_config_base.json \
		--bert_dir ${OUTPUT_DIR} \
		--train_batch_size 24 \
		--num_train_epochs 10 \
		--do_train 
}

large_model()
{
	OUTPUT_DIR=pretrain/pytorch/albert_large_zh

	python run_classifier.py \
		--arch albert_large \
		--albert_config_path ${OUTPUT_DIR}/albert_config_large.json \
		--bert_dir ${OUTPUT_DIR} \
		--train_batch_size 24 \
		--num_train_epochs 10 \
		--do_train 
}

xlarge_model()
{
	OUTPUT_DIR=pretrain/pytorch/albert_xlarge_zh

	python run_classifier.py \
		--arch albert_xlarge \
		--albert_config_path ${OUTPUT_DIR}/albert_config_xlarge.json \
		--bert_dir ${OUTPUT_DIR} \
		--train_batch_size 24 \
		--num_train_epochs 10 \
		--do_train 
}

[ "$*" = "" ] && usage

case $1 in
--base)
	base_model
	;;
--large)
	large_model
	;;
--xlarge)
	xlarge_model
	;;
*)
	usage
	;;
esac


