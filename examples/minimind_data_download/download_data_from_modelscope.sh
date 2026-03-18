dataset_name=jingyaogong/minimind_dataset
dir=/data/qldu/wyang/workspace/exp/minimind/dataset
cache_dir=/data/qldu/.cache/huggingface/datasets/

# if 'modelscope' command is not found, please run the following command to install it first: 'pip install modelscope'
modelscope download --dataset "${dataset_name}" pretrain_hq.jsonl --local_dir "${dir}"