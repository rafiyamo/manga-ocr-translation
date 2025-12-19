"""
Translation-related helper functions.

There are two layers:

1. placeholder_translate(text: str) -> str
   - Simple placeholder used if the real model isn't available.

2. translate_lines_with_model(lines: List[str]) -> List[str]
   - Uses a custom seq2seq model trained in notebooks/train_translation_model.ipynb
     and saved to models/translator.pt.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Public placeholder (kept so the rest of the code can still use it)
# ---------------------------------------------------------------------------


def placeholder_translate(text: str) -> str:
    """
    Temporary placeholder so the pipeline can run even if the real model
    isn't available.
    """
    return f"[EN translation of]: {text}"


# ---------------------------------------------------------------------------
# Char-level tokenizer utilities (must match the Colab notebook)
# ---------------------------------------------------------------------------

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    """
    Encode a string as a list of token ids, with SOS/EOS and padding.
    """
    ids: List[int] = [vocab[SOS_TOKEN]]
    for ch in text:
        ids.append(vocab.get(ch, vocab[PAD_TOKEN]))
        if len(ids) >= max_len - 1:
            break
    ids.append(vocab[EOS_TOKEN])

    # pad or truncate
    if len(ids) < max_len:
        ids.extend([vocab[PAD_TOKEN]] * (max_len - len(ids)))
    else:
        ids = ids[:max_len]
    return ids


def decode_ids(ids: List[int], inv_vocab: Dict[int, str]) -> str:
    """
    Decode token ids back into a string, ignoring PAD and special tokens.
    """
    chars: List[str] = []
    for i in ids:
        token = inv_vocab.get(i, "")
        if token in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
            continue
        chars.append(token)
    return "".join(chars)


# ---------------------------------------------------------------------------
# Model definitions (must match the Colab notebook)
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: [batch, src_len]
        # returns hidden: [1, batch, hidden_dim]
        embedded = self.embedding(src)
        _, hidden = self.gru(embedded)
        return hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_tokens: torch.Tensor, hidden: torch.Tensor):
        """
        input_tokens: [batch]
        hidden: [1, batch, hidden_dim]
        returns logits: [batch, vocab_size], hidden: [1, batch, hidden_dim]
        """
        embedded = self.embedding(input_tokens.unsqueeze(1))  # [batch, 1, embed_dim]
        output, hidden = self.gru(embedded, hidden)           # [batch, 1, hidden_dim]
        logits = self.fc_out(output.squeeze(1))               # [batch, vocab_size]
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        pad_idx: int,
        sos_idx: int,
        eos_idx: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        """
        src: [batch, src_len]
        tgt: [batch, tgt_len]
        returns: outputs [batch, tgt_len, vocab_size]
        """
        batch_size, tgt_len = tgt.size()
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)

        hidden = self.encoder(src)
        input_tok = tgt[:, 0]  # <sos>

        for t in range(1, tgt_len):
            logits, hidden = self.decoder(input_tok, hidden)
            outputs[:, t] = logits

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = logits.argmax(dim=1)

            input_tok = tgt[:, t] if teacher_force else top1

        return outputs


# ---------------------------------------------------------------------------
# Loading the trained checkpoint
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    model: Seq2Seq
    vocab: Dict[str, int]
    inv_vocab: Dict[int, str]
    pad_idx: int
    sos_idx: int
    eos_idx: int
    max_src_len: int
    max_tgt_len: int


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "translator.pt"


@lru_cache(maxsize=1)
def load_translator_model() -> LoadedModel:
    """
    Load the trained translator model from models/translator.pt.

    Returns a LoadedModel with the model in eval() mode on CPU.
    """
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Translator model not found at: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    vocab: Dict[str, int] = checkpoint["vocab"]
    pad_idx: int = checkpoint["pad_idx"]
    sos_idx: int = checkpoint["sos_idx"]
    eos_idx: int = checkpoint["eos_idx"]
    config_dict = checkpoint.get("config", {})

    embed_dim = int(config_dict.get("embed_dim", 128))
    hidden_dim = int(config_dict.get("hidden_dim", 256))
    max_src_len = int(config_dict.get("max_src_len", 64))
    max_tgt_len = int(config_dict.get("max_tgt_len", 64))

    inv_vocab: Dict[int, str] = {i: tok for tok, i in vocab.items()}

    encoder = Encoder(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        pad_idx=pad_idx,
    )
    decoder = Decoder(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        pad_idx=pad_idx,
    )
    model = Seq2Seq(encoder, decoder, pad_idx, sos_idx, eos_idx)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return LoadedModel(
        model=model,
        vocab=vocab,
        inv_vocab=inv_vocab,
        pad_idx=pad_idx,
        sos_idx=sos_idx,
        eos_idx=eos_idx,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )


# ---------------------------------------------------------------------------
# Public API: translate with the custom model (with safe fallback)
# ---------------------------------------------------------------------------


def translate_text_with_model(text: str) -> str:
    """
    Translate a single string using the custom model.
    If the model checkpoint is missing or something goes wrong,
    fall back to placeholder_translate().
    """
    try:
        loaded = load_translator_model()
    except Exception:
        # Safe fallback: still return something rather than crashing the app.
        return placeholder_translate(text)

    model = loaded.model
    vocab = loaded.vocab
    inv_vocab = loaded.inv_vocab
    sos_idx = loaded.sos_idx
    eos_idx = loaded.eos_idx
    max_src_len = loaded.max_src_len
    max_tgt_len = loaded.max_tgt_len

    src_ids = encode_text(text, vocab, max_src_len)
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0)  # [1, src_len]

    model.eval()
    with torch.no_grad():
        hidden = model.encoder(src_tensor)

        input_tok = torch.tensor([sos_idx], dtype=torch.long)
        decoded_ids: List[int] = [sos_idx]

        for _ in range(max_tgt_len - 1):
            logits, hidden = model.decoder(input_tok, hidden)
            next_tok = logits.argmax(dim=1)  # [1]
            token_id = next_tok.item()
            decoded_ids.append(token_id)

            if token_id == eos_idx:
                break

            input_tok = next_tok

    return decode_ids(decoded_ids, inv_vocab)


def translate_lines_with_model(lines: List[str]) -> List[str]:
    """
    Translate a list of strings using the custom model.
    Falls back to placeholder_translate() if the model can't be loaded.
    """
    return [translate_text_with_model(line) for line in lines]
