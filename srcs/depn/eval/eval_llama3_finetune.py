import math
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_dir: str):
    """Load model + tokenizer on a single GPU in float16 for evaluation."""
    model_dir = str(model_dir)
    print(f"[INFO] Loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    print(f"[INFO] Loading model from {model_dir}")
    # 单卡评估，直接全模型放到 0 号 GPU 上，使用 fp16
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model.eval()
    return model, tokenizer


def build_eval_dataset(tokenizer, valid_path: str, max_seq_len: int = 256, max_samples: int = 400):
    """Tokenize valid.txt into fixed-length blocks for perplexity evaluation."""
    valid_path = Path(valid_path)
    assert valid_path.is_file(), f"valid file not found: {valid_path}"

    print(f"[INFO] Loading validation text from {valid_path}")
    text = valid_path.read_text(encoding="utf-8")
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"][0]

    total_tokens = input_ids.size(0)
    num_blocks = total_tokens // max_seq_len
    if num_blocks == 0:
        raise ValueError(f"Validation text too short ({total_tokens} tokens) for block size {max_seq_len}")

    max_blocks = max_samples
    num_blocks = min(num_blocks, max_blocks)
    input_ids = input_ids[: num_blocks * max_seq_len]
    input_ids = input_ids.view(num_blocks, max_seq_len)

    print(
        f"[INFO] Built eval dataset: {num_blocks} blocks of {max_seq_len} tokens "
        f"(~{num_blocks * max_seq_len} tokens total)"
    )
    return input_ids


@torch.no_grad()
def compute_perplexity(model, input_ids: torch.Tensor, batch_size: int = 1):
    device = next(model.parameters()).device
    n_blocks = input_ids.size(0)
    losses = []

    print(f"[INFO] Computing perplexity on {n_blocks} blocks, batch_size={batch_size}")
    for start in range(0, n_blocks, batch_size):
        end = min(start + batch_size, n_blocks)
        batch = input_ids[start:end].to(device)
        # Shift labels same as input_ids for CLM
        outputs = model(batch, labels=batch)
        loss = outputs.loss.detach().float()
        losses.append(loss)

        if (start // batch_size) % 20 == 0:
            print(f"  Processed {end}/{n_blocks} blocks, loss={loss.item():.4f}")

    mean_loss = torch.stack(losses).mean().item()
    ppl = math.exp(mean_loss)
    return mean_loss, ppl


@torch.no_grad()
def generate_samples(model, tokenizer, prompts, max_new_tokens: int = 128):
    device = next(model.parameters()).device
    for i, prompt in enumerate(prompts):
        print("=" * 80)
        print(f"[PROMPT {i}] {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"[GEN {i}] {generated}")


def main():
    base_model_dir = "/data1/D-PIKE/pretrained_models/llama3-8B"
    finetuned_model_dir = "/data1/D-PIKE/DEPN-main/data/model/llama3_8b_ep5"
    valid_path = "/data1/D-PIKE/DEPN-main/data/temp_data/valid.txt"
    max_seq_len = 256

    t0 = time.time()
    print("[INFO] ===== Loading base model (for comparison) =====")
    base_model, base_tok = load_model_and_tokenizer(base_model_dir)
    print(f"[TIME] Loaded base model in {time.time() - t0:.1f}s")

    # 使用 base tokenizer 在验证集上构造评估样本
    base_ids = build_eval_dataset(base_tok, valid_path, max_seq_len=max_seq_len, max_samples=400)

    # 计算 base model 的困惑度
    base_loss, base_ppl = compute_perplexity(base_model, base_ids, batch_size=1)
    print(f"[RESULT] Base model:     loss={base_loss:.4f}, ppl={base_ppl:.4f}")

    # 释放 base model 显存
    del base_model
    torch.cuda.empty_cache()

    # 加载微调模型并用其 tokenizer 构造同一验证集的评估样本
    t1 = time.time()
    print("[INFO] ===== Loading finetuned model =====")
    ft_model, ft_tok = load_model_and_tokenizer(finetuned_model_dir)
    print(f"[TIME] Loaded finetuned model in {time.time() - t1:.1f}s")

    ft_ids = build_eval_dataset(ft_tok, valid_path, max_seq_len=max_seq_len, max_samples=400)

    # 计算微调模型的困惑度
    ft_loss, ft_ppl = compute_perplexity(ft_model, ft_ids, batch_size=1)
    print(f"[RESULT] Finetuned model: loss={ft_loss:.4f}, ppl={ft_ppl:.4f}")

    # Some qualitative generations (Chinese + English)
    prompts = [
        "请根据以下邮件内容，概括主要观点并判断其中是否包含敏感隐私信息：",
        "请判断这封邮件中是否出现了个人电话号码、家庭住址或银行账户等敏感信息，并说明理由：",
        "Given the following internal company email, summarize the main topic in one sentence:",
        "You are an email privacy assistant. Explain whether the following email leaks any sensitive personal data:",
    ]
    print("[INFO] ===== Finetuned model generation samples =====")
    generate_samples(ft_model, ft_tok, prompts, max_new_tokens=128)


if __name__ == "__main__":
    main()

