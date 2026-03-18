from datasets import load_dataset
import json
import os

# 注意：这里加载的是 en.noclean 版本
dataset = load_dataset(
    "json",
    data_files={"train": "https://huggingface.co/datasets/allenai/c4/resolve/main/en.noclean/c4-train.07167-of-07168.json.gz"},
    split="train",
    streaming=True,
)

output_file = "./examples/data_preprocess/raw_data/c4_noclean_sample.jsonl"
if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))
print(f"正在提取‘原始’数据到 {output_file}...")

with open(output_file, "w", encoding="utf-8") as f:
    for i, example in enumerate(dataset.take(1000)):
        # 只保留 text 和 url，方便对比
        f.write(json.dumps({"text": example["text"], "url": example["url"]}, ensure_ascii=False) + "\n")
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} 条...")

print("数据下载完成！")