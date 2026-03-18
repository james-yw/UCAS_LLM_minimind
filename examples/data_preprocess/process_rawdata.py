from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.filters import GopherQualityFilter, LanguageFilter, C4QualityFilter
from datatrove.pipeline.dedup import MinhashDedupSignature, MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupFilter
from datatrove.pipeline.dedup.minhash import MinhashConfig
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig
import os

# 路径设置
INPUT_FILE = "c4_noclean_sample.jsonl"
BASE_OUTPUT = "./examples/data_preprocess/processed_data"

# 确保输出目录存在
os.makedirs(BASE_OUTPUT, exist_ok=True)

# 1. 基础过滤流 (清理脏数据)
filter_pipeline = [
    JsonlReader(data_folder="./examples/data_preprocess/raw_data", glob_pattern=INPUT_FILE),
    LanguageFilter(languages=["en"], language_threshold=0.8), # 确保是英文
    GopherQualityFilter(),                 # 剔除低质量页面
    C4QualityFilter(),                     # 进一步精炼
    PIIFormatter(),                        # 脱敏处理
    JsonlWriter(f"{BASE_OUTPUT}/filtered_data")
]

# 2. 去重配置 (针对小样本微调)
minhash_cfg = MinhashConfig(
    hash_config=HashConfig(hash_fc="sha1", precision=64),
    num_buckets=5, 
    hashes_per_bucket=10,
    n_grams=5
)

def run_pipeline():
    # 步骤 A: 基础清洗
    print("--- 正在进行基础清洗与脱敏 ---")
    LocalPipelineExecutor(pipeline=filter_pipeline, tasks=1).run()

    # 步骤 B: 生成去重签名
    print("--- 正在生成 MinHash 签名 ---")
    LocalPipelineExecutor(pipeline=[
        JsonlReader(f"{BASE_OUTPUT}/filtered_data"),
        MinhashDedupSignature(output_folder=f"{BASE_OUTPUT}/signatures", config=minhash_cfg)
    ], tasks=1).run()

    # 步骤 C: 聚类并过滤
    print("--- 正在识别并移除重复项 ---")
    LocalPipelineExecutor(pipeline=[
        MinhashDedupBuckets(input_folder=f"{BASE_OUTPUT}/signatures", output_folder=f"{BASE_OUTPUT}/buckets", config=minhash_cfg),
    ], tasks=5).run()
    
    LocalPipelineExecutor(pipeline=[
        MinhashDedupCluster(input_folder=f"{BASE_OUTPUT}/buckets", output_folder=f"{BASE_OUTPUT}/remove_ids", config=minhash_cfg),
    ], tasks=1).run()

    # 最终输出
    print("--- 导出最终数据集 ---")
    LocalPipelineExecutor(pipeline=[
        JsonlReader(f"{BASE_OUTPUT}/filtered_data"),
        MinhashDedupFilter(input_folder=f"{BASE_OUTPUT}/remove_ids"),
        JsonlWriter(f"{BASE_OUTPUT}/final_output")
    ], tasks=1).run()

if __name__ == "__main__":
    run_pipeline()