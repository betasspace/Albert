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
	INPUT_DIR=~/tmp/albert_base_zh
	OUTPUT_DIR=pretrain/pytorch/albert_base_zh

	mkdir -p ${OUTPUT_DIR}

	python convert_albert_tf_checkpoint_to_pytorch.py \
		--tf_checkpoint_path ${INPUT_DIR} \
		--bert_config_file ${INPUT_DIR}/albert_config_base.json \
		--pytorch_dump_path ${OUTPUT_DIR}/pytorch_model.bin

	cp -v ${INPUT_DIR}/albert_config_base.json ${OUTPUT_DIR}
}

large_model()
{
	INPUT_DIR=~/tmp/albert_large_zh
	OUTPUT_DIR=pretrain/pytorch/albert_large_zh

	mkdir -p ${OUTPUT_DIR}

	python convert_albert_tf_checkpoint_to_pytorch.py \
		--tf_checkpoint_path ${INPUT_DIR} \
		--bert_config_file ${INPUT_DIR}/albert_config_large.json \
		--pytorch_dump_path ${OUTPUT_DIR}/pytorch_model.bin

	cp -v ${INPUT_DIR}/albert_config_large.json ${OUTPUT_DIR}
}

xlarge_model()
{
	INPUT_DIR=~/tmp/albert_xlarge_zh
	OUTPUT_DIR=pretrain/pytorch/albert_xlarge_zh

	mkdir -p ${OUTPUT_DIR}

	python convert_albert_tf_checkpoint_to_pytorch.py \
		--tf_checkpoint_path ${INPUT_DIR} \
		--bert_config_file ${INPUT_DIR}/albert_config_xlarge.json \
		--pytorch_dump_path ${OUTPUT_DIR}/pytorch_model.bin

	cp -v ${INPUT_DIR}/albert_config_xlarge.json ${OUTPUT_DIR}
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

