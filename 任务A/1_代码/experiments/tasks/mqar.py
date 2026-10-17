# Adapted from HazyResearch/zoology, Apache-2.0.
# Source: zoology/data/multiquery_ar.py at 1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb.
"""保留原 MQAR 的符号序列、幂律查询位置和查询处预测标签。"""

import numpy as np


def generate_mqar(num_samples, seq_len, symbol_to_token_id, seed, num_kv_pairs=8, power_a=0.01):
    vocab_size = len(symbol_to_token_id)
    assert seq_len % 2 == 0
    assert vocab_size > seq_len
    assert 4 * num_kv_pairs <= seq_len
    rng = np.random.RandomState(seed)

    # 与作者代码相同：前半词表是 key，后半词表是 value；0 是非查询位置的填充值。
    key_choices = np.arange(1, vocab_size // 2)
    value_choices = np.arange(vocab_size // 2, vocab_size)
    keys = np.array([rng.choice(key_choices, size=num_kv_pairs, replace=False) for _ in range(num_samples)])
    values = np.array([rng.choice(value_choices, size=num_kv_pairs, replace=False) for _ in range(num_samples)])
    context_size = 2 * num_kv_pairs
    symbols = np.zeros((num_samples, seq_len), dtype=np.int64)
    symbols[:, :context_size:2] = keys
    symbols[:, 1:context_size:2] = values

    # 这是作者的查询位置分布；power_a=0.01。保持 KV 数量不变，只改变 seq_len。
    space = (seq_len - context_size) // 2
    probabilities = power_a * np.arange(1, space + 1) ** (power_a - 1)
    probabilities /= probabilities.sum()
    gaps = np.array([rng.choice(space, size=num_kv_pairs, replace=False, p=probabilities) for _ in range(num_samples)])
    query_positions = context_size + 2 * gaps
    np.put_along_axis(symbols, query_positions, keys, axis=1)

    # 标签已经与输入位置对齐：看到 query 的位置，应预测对应 value。不要再 shift。
    mapping = np.asarray(symbol_to_token_id, dtype=np.int64)
    input_ids = mapping[symbols]
    labels = np.full_like(input_ids, -100)
    np.put_along_axis(labels, query_positions, mapping[values], axis=1)
    source_positions = np.broadcast_to(np.arange(0, context_size, 2), query_positions.shape).copy()
    return {
        "input_ids": input_ids,
        "labels": labels,
        "query_positions": query_positions,
        "source_positions": source_positions,
        "dependency_distances": query_positions - source_positions,
        "symbol_to_token_id": mapping,
    }

