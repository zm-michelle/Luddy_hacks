from __future__ import annotations

from collections.abc import Sequence

import torch

from config import CTC_BLANK_INDEX, DEFAULT_CHARSET


def build_charset_maps(charset: str = DEFAULT_CHARSET) -> tuple[dict[str, int], dict[int, str]]:
    char_to_idx = {ch: i + 1 for i, ch in enumerate(charset)}
    idx_to_char = {i + 1: ch for i, ch in enumerate(charset)}
    return char_to_idx, idx_to_char


def encode_text(text: str, charset: str = DEFAULT_CHARSET) -> list[int]:
    char_to_idx, _ = build_charset_maps(charset)
    return [char_to_idx[ch] for ch in text if ch in char_to_idx]


def decode_indices(indices: Sequence[int], charset: str = DEFAULT_CHARSET) -> str:
    _, idx_to_char = build_charset_maps(charset)
    return "".join(idx_to_char[i] for i in indices if i != CTC_BLANK_INDEX and i in idx_to_char)


def decode_ctc_greedy(logits: torch.Tensor, charset: str = DEFAULT_CHARSET) -> list[str]:
    """Greedy CTC decode. No lexicon, spell correction, or language model."""
    if logits.dim() != 3:
        raise ValueError("Expected logits shaped [T, B, C] or [B, T, C].")

    if logits.shape[0] < logits.shape[1]:
        # Most CRNN code returns [T, B, C], but accept [B, T, C] for convenience.
        logits = logits.transpose(0, 1)

    best = logits.detach().argmax(dim=-1).cpu()
    _, idx_to_char = build_charset_maps(charset)
    decoded: list[str] = []

    for b in range(best.shape[1]):
        chars: list[str] = []
        prev = CTC_BLANK_INDEX
        for t in range(best.shape[0]):
            idx = int(best[t, b])
            if idx != CTC_BLANK_INDEX and idx != prev:
                ch = idx_to_char.get(idx)
                if ch is not None:
                    chars.append(ch)
            prev = idx
        decoded.append("".join(chars))

    return decoded
