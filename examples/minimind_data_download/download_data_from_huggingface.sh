
dataset_name=jingyaogong/minimind_dataset
dir=/data/qldu/wyang/workspace/exp/minimind/dataset
cache_dir=/data/qldu/.cache/huggingface/datasets/

huggingface-cli download --repo-type dataset --resume-download "${dataset_name}" pretrain_hq.jsonl --local-dir "${dir}"