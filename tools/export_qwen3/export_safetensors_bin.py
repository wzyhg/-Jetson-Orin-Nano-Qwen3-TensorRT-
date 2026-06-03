import argparse
import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open


def build_weight_keys(num_layers):
    keys = []
    keys += [f"model.layers.{i}.input_layernorm.weight" for i in range(num_layers)]
    keys += [f"model.layers.{i}.post_attention_layernorm.weight" for i in range(num_layers)]
    keys += ["model.norm.weight"]

    keys += ["model.embed_tokens.weight"]

    keys += [f"model.layers.{i}.self_attn.q_proj.weight" for i in range(num_layers)]
    keys += [f"model.layers.{i}.self_attn.q_norm.weight" for i in range(num_layers)]

    keys += [f"model.layers.{i}.self_attn.k_proj.weight" for i in range(num_layers)]
    keys += [f"model.layers.{i}.self_attn.k_norm.weight" for i in range(num_layers)]

    keys += [f"model.layers.{i}.self_attn.v_proj.weight" for i in range(num_layers)]
    keys += [f"model.layers.{i}.self_attn.o_proj.weight" for i in range(num_layers)]

    keys += [f"model.layers.{i}.mlp.gate_proj.weight" for i in range(num_layers)]
    keys += [f"model.layers.{i}.mlp.down_proj.weight" for i in range(num_layers)]
    keys += [f"model.layers.{i}.mlp.up_proj.weight" for i in range(num_layers)]
    keys += ["lm_head.weight"]
    return keys


def index_safetensors_files(model_dir):
    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    index = {}
    for file_path in safetensors_files:
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                index[key] = file_path
    return index


def write_tensor(handle, tensor):
    data = tensor.detach().cpu().contiguous().view(-1).to(torch.float32).numpy()
    handle.write(data.tobytes())


def quantize_q80(tensor, group_size):
    data = tensor.detach().cpu().contiguous().view(-1).to(torch.float32)
    if data.numel() % group_size != 0:
        raise ValueError(f"Tensor with {data.numel()} values is not divisible by group_size={group_size}")
    grouped = data.view(-1, group_size)
    scale = grouped.abs().max(dim=1).values / 127.0
    scale[scale == 0] = 1.0
    quant = torch.round(grouped / scale[:, None]).clamp(-127, 127).to(torch.int8)
    dequant = (quant.to(torch.float32) * scale[:, None]).view(-1)
    max_error = (dequant - data).abs().max().item()
    return quant.view(-1), scale.to(torch.float32), max_error


def write_q80(handle, tensor, group_size):
    quant, scale, max_error = quantize_q80(tensor, group_size)
    handle.write(quant.numpy().tobytes())
    handle.write(scale.numpy().tobytes())
    return max_error


def is_qwen3_int8_quantized_key(key):
    return (
        key.endswith(".self_attn.q_proj.weight")
        or key.endswith(".self_attn.k_proj.weight")
        or key.endswith(".self_attn.v_proj.weight")
        or key.endswith(".self_attn.o_proj.weight")
        or key.endswith(".mlp.gate_proj.weight")
        or key.endswith(".mlp.down_proj.weight")
        or key.endswith(".mlp.up_proj.weight")
        or key == "lm_head.weight"
    )


def export_qwen3_bin(model_dir, output_file, weight_format="fp32", group_size=64):
    model_dir = Path(model_dir)
    output_file = Path(output_file)

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    dim = config["num_attention_heads"] * config["head_dim"]
    hidden_dim = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    vocab_size = config["vocab_size"]
    max_seq_len = config["max_position_embeddings"]
    intermediate_size = config["intermediate_size"]

    keys = build_weight_keys(num_layers)
    tensor_index = index_safetensors_files(model_dir)

    missing_keys = [key for key in keys if key not in tensor_index]
    if missing_keys:
        print("Missing keys:")
        for key in missing_keys[:50]:
            print(key)
        raise KeyError(f"Missing {len(missing_keys)} required tensors")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as out:
        out.write(
            struct.pack(
                "iiiiiiii",
                dim,
                hidden_dim,
                num_layers,
                num_heads,
                num_kv_heads,
                vocab_size,
                max_seq_len,
                intermediate_size,
            )
        )
        if weight_format == "int8":
            out.write(struct.pack("i", group_size))

        quant_errors = []
        for index, key in enumerate(keys, 1):
            file_path = tensor_index[key]
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(key)
                if weight_format == "int8" and is_qwen3_int8_quantized_key(key):
                    max_error = write_q80(out, tensor, group_size)
                    quant_errors.append((max_error, key))
                    print(f"{index}/{len(keys)} wrote int8 {key}, max_error={max_error:.6f}")
                else:
                    write_tensor(out, tensor)
                    print(f"{index}/{len(keys)} wrote fp32 {key}")

    if quant_errors:
        max_error, key = max(quant_errors)
        print(f"max int8 quantization error: {max_error:.6f} at {key}")

    print(f"wrote {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export Qwen3 safetensors weights to KuiperLLama .bin format.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/models/Qwen3-0.6B",
        help="Path to the HuggingFace Qwen3 model directory.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/models/Qwen3-0.6B.bin",
        help="Output KuiperLLama .bin file path.",
    )
    parser.add_argument(
        "--format",
        choices=["fp32", "int8"],
        default="fp32",
        help="Weight format to export. int8 keeps norm/embedding fp32 and quantizes matmul weights as Q8_0.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=64,
        help="Q8_0 quantization group size, only used with --format int8.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_qwen3_bin(args.model_dir, args.output_file, args.format, args.group_size)
