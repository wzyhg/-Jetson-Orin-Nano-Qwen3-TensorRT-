import argparse
import struct
from pathlib import Path

import numpy as np


HEADER_FORMAT = "iiiiiiii"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def quantize_q80(values, group_size):
    if values.size % group_size != 0:
        raise ValueError(f"Tensor with {values.size} values is not divisible by group_size={group_size}")

    grouped = values.reshape(-1, group_size).astype(np.float32, copy=False)
    scales = np.max(np.abs(grouped), axis=1).astype(np.float32) / np.float32(127.0)
    scales[scales == 0] = np.float32(1.0)
    quantized = np.round(grouped / scales[:, None])
    quantized = np.clip(quantized, -127, 127).astype(np.int8)
    dequantized = (quantized.astype(np.float32) * scales[:, None]).reshape(-1)
    max_error = float(np.max(np.abs(dequantized - values)))
    return quantized.reshape(-1), scales, max_error


def read_fp32_tensor(data, offset, count):
    byte_count = count * 4
    end = offset + byte_count
    if end > len(data):
        raise ValueError(f"Unexpected EOF while reading fp32 tensor at byte {offset}")
    tensor = np.frombuffer(data, dtype=np.float32, count=count, offset=offset)
    return tensor, end


def write_fp32_tensor(out, data, offset, count):
    tensor, offset = read_fp32_tensor(data, offset, count)
    out.write(tensor.tobytes())
    return offset


def write_int8_tensor(out, data, offset, count, group_size, name):
    tensor, offset = read_fp32_tensor(data, offset, count)
    quantized, scales, max_error = quantize_q80(tensor, group_size)
    out.write(quantized.tobytes())
    out.write(scales.tobytes())
    print(f"wrote int8 {name}, values={count}, scales={scales.size}, max_error={max_error:.6f}")
    return offset, max_error


def write_int8_layer_tensors(out, data, offset, num_layers, values_per_layer, group_size, name):
    errors = []
    for layer_idx in range(num_layers):
        offset, max_error = write_int8_tensor(
            out,
            data,
            offset,
            values_per_layer,
            group_size,
            f"{name}.{layer_idx}",
        )
        errors.append(max_error)
    return offset, max(errors)


def quantize_qwen3_bin2(input_file, output_file, group_size):
    input_file = Path(input_file)
    output_file = Path(output_file)

    data = input_file.read_bytes()
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Input file is too small: {input_file}")

    dim, hidden_dim, num_layers, num_heads, num_kv_heads, vocab_size, max_seq_len, intermediate_size = struct.unpack(
        HEADER_FORMAT, data[:HEADER_SIZE]
    )
    kv_dim = (dim * num_kv_heads) // num_heads
    head_size = dim // num_heads

    offset = HEADER_SIZE
    output_file.parent.mkdir(parents=True, exist_ok=True)
    errors = []

    with output_file.open("wb") as out:
        out.write(data[:HEADER_SIZE])
        out.write(struct.pack("i", group_size))

        # fp32: input_layernorm, post_attention_layernorm, final norm
        offset = write_fp32_tensor(out, data, offset, (2 * num_layers + 1) * hidden_dim)

        # fp32: embedding
        offset = write_fp32_tensor(out, data, offset, vocab_size * hidden_dim)

        # int8 weights must be written as each layer's int8 values immediately followed by
        # that layer's scales, matching Qwen3Model::create_param_quant_layers().
        offset, err = write_int8_layer_tensors(out, data, offset, num_layers, hidden_dim * dim, group_size, "q_proj")
        errors.append(("q_proj", err))

        # fp32: q_norm
        offset = write_fp32_tensor(out, data, offset, num_layers * head_size)

        # int8: k_proj
        offset, err = write_int8_layer_tensors(
            out, data, offset, num_layers, hidden_dim * kv_dim, group_size, "k_proj"
        )
        errors.append(("k_proj", err))

        # fp32: k_norm
        offset = write_fp32_tensor(out, data, offset, num_layers * head_size)

        # int8: v_proj, o_proj, gate_proj, down_proj, up_proj
        quantized_layer_blocks = [
            ("v_proj", hidden_dim * kv_dim),
            ("o_proj", dim * hidden_dim),
            ("gate_proj", hidden_dim * intermediate_size),
            ("down_proj", intermediate_size * hidden_dim),
            ("up_proj", intermediate_size * hidden_dim),
        ]
        for name, count in quantized_layer_blocks:
            offset, err = write_int8_layer_tensors(out, data, offset, num_layers, count, group_size, name)
            errors.append((name, err))

        offset, err = write_int8_tensor(out, data, offset, vocab_size * hidden_dim, group_size, "lm_head")
        errors.append(("lm_head", err))

    if offset != len(data):
        raise ValueError(f"Did not consume whole input file: consumed {offset}, size {len(data)}")

    name, err = max(errors, key=lambda item: item[1])
    print(f"max int8 quantization error: {err:.6f} at {name}")
    print(f"wrote {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize an existing Qwen3 fp32 bin2 file to Q8_0 int8.")
    parser.add_argument("--input_file", required=True, help="Input Qwen3 fp32 .bin2 file.")
    parser.add_argument("--output_file", required=True, help="Output Qwen3 Q8_0 int8 file.")
    parser.add_argument("--group_size", type=int, default=64, help="Q8_0 quantization group size.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    quantize_qwen3_bin2(args.input_file, args.output_file, args.group_size)
