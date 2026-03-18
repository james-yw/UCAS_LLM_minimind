import os
import sys
import json

__package__ = "trainer"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import init_model


DATA_PATH = "./dataset/pretrain_hq.jsonl"
SAVE_PATH = "./examples/data_tokenize/tokenized_data/tokenized_pretrain_hq.jsonl"
MAX_SAMPLES = 1000

def main():
    # 1. 初始化 tokenizer
    lm_config = MiniMindConfig(
        hidden_size=512,
        num_hidden_layers=8,
        use_moe=False,
    )
    _, tokenizer = init_model(
        lm_config,
        from_weight="none",
        tokenizer_path=os.path.join(PROJECT_ROOT, "model"),
        save_dir=os.path.join(PROJECT_ROOT, "out"),
        device="cpu",
    )

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    count = 0
    total_tokens = 0

    with open(DATA_PATH, "r", encoding="utf-8") as fin, \
         open(SAVE_PATH, "w", encoding="utf-8") as fout:

        for line in fin:
            if count >= MAX_SAMPLES:
                break

            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)
            text = sample.get("text", "")
            if not text:
                continue

            input_ids = tokenizer.encode(text)
            decoded_text = tokenizer.decode(input_ids)

            record = {
                "id": count,
                "text": text,
                "input_ids": input_ids,
                "num_tokens": len(input_ids),
                "decoded_text": decoded_text,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_tokens += len(input_ids)
            count += 1

            if count <= 3:
                print(f"\n===== Sample {count} =====")
                print("原文：", text[:200])
                print("token数：", len(input_ids))
                print("前20个token id：", input_ids[:20])
                print("decode预览：", decoded_text[:200])

    print(f"\n完成，共处理 {count} 条样本")
    print(f"总 token 数: {total_tokens}")
    print(f"平均每条 token 数: {total_tokens / count:.2f}" if count > 0 else "无有效样本")
    print(f"结果已保存到: {SAVE_PATH}")


if __name__ == "__main__":
    main()