from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# --- Special tokens (must match what you used in training) ---
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

# --- Global device + lazy-loaded checkpoint ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODEL: "CharEncoderDecoder | None" = None
_VOCAB: Dict[str, int] | None = None
_INV_VOCAB: Dict[int, str] | None = None
_CONFIG: Dict[str, int] | None = None


# ---------------------------------------------------------------------------
#  Model definitions (must match the notebook)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src_ids: torch.Tensor) -> torch.Tensor:
        """
        src_ids: [batch, src_len]
        returns: hidden state [1, batch, hidden_dim]
        """
        emb = self.embedding(src_ids)
        _, hidden = self.rnn(emb)
        return hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_tok: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_tok: [batch] (current token ids)
        hidden: [1, batch, hidden_dim]
        returns:
          logits: [batch, vocab_size]
          hidden: [1, batch, hidden_dim]
        """
        emb = self.embedding(input_tok.unsqueeze(1))  # [batch, 1, embed_dim]
        output, hidden = self.rnn(emb, hidden)        # output: [batch, 1, hidden_dim]
        logits = self.out(output.squeeze(1))          # [batch, vocab_size]
        return logits, hidden


class CharEncoderDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int = 0):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, pad_idx)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, pad_idx)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        src_ids: [batch, src_len]
        tgt_ids: [batch, tgt_len]
        returns:
          logits: [batch, tgt_len, vocab_size]
        """
        batch_size, tgt_len = tgt_ids.shape
        hidden = self.encoder(src_ids)
        vocab_size = self.decoder.out.out_features

        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src_ids.device)

        input_tok = tgt_ids[:, 0]  # start with SOS for each example

        for t in range(1, tgt_len):
            logits, hidden = self.decoder(input_tok, hidden)
            outputs[:, t, :] = logits

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = logits.argmax(dim=-1)

            input_tok = tgt_ids[:, t] if teacher_force else top1

        return outputs


# ---------------------------------------------------------------------------
#  Vocab helper functions (copied from notebook)
# ---------------------------------------------------------------------------

def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    """
    Turn a string into a fixed-length list of token ids.
    Same logic as in the training notebook.
    """
    ids = [vocab[SOS_TOKEN]]
    for ch in text.strip():
        ids.append(vocab.get(ch, vocab[PAD_TOKEN]))
    ids.append(vocab[EOS_TOKEN])

    if len(ids) < max_len:
        ids += [vocab[PAD_TOKEN]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return ids


def decode_ids(ids: List[int], inv_vocab: Dict[int, str]) -> str:
    """
    Convert token ids back to a string, skipping PAD/SOS/EOS.
    """
    chars: List[str] = []
    for idx in ids:
        tok = inv_vocab.get(idx, "")
        if tok in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
            continue
        chars.append(tok)
    return "".join(chars)


# ---------------------------------------------------------------------------
#  Checkpoint loading
# ---------------------------------------------------------------------------

def _default_model_path() -> Path:
    # src/translate.py  ->  project_root/models/translator.pt
    here = Path(__file__).resolve()
    root = here.parent.parent
    return root / "models" / "translator.pt"


def _load_checkpoint(model_path: Path | None = None) -> Tuple[CharEncoderDecoder, Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Lazily load model, vocab and config from translator.pt.
    """
    global _MODEL, _VOCAB, _INV_VOCAB, _CONFIG

    if _MODEL is not None:
        return _MODEL, _VOCAB, _INV_VOCAB, _CONFIG

    if model_path is None:
        model_path = _default_model_path()

    ckpt = torch.load(str(model_path), map_location=DEVICE)

    vocab: Dict[str, int] = ckpt["vocab"]
    config: Dict[str, int] = ckpt["config"]

    model = CharEncoderDecoder(
        vocab_size=len(vocab),
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        pad_idx=vocab[PAD_TOKEN],
    )
    # strict=True here because we matched the notebook architecture
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    inv_vocab = {i: tok for tok, i in vocab.items()}

    _MODEL = model
    _VOCAB = vocab
    _INV_VOCAB = inv_vocab
    _CONFIG = config

    return model, vocab, inv_vocab, config


# ---------------------------------------------------------------------------
#  Public translation API
# ---------------------------------------------------------------------------

def translate_text_with_model(text: str, max_len: int | None = None) -> str:
    """
    Translate a single Japanese string into English using the trained model.
    """
    model, vocab, inv_vocab, config = _load_checkpoint()

    if max_len is None:
        max_len = config["max_tgt_len"]

    # Encode source
    src_ids = encode_text(text, vocab, config["max_src_len"])
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)

    # Run encoder
    hidden = model.encoder(src_tensor)

    eos_idx = vocab[EOS_TOKEN]
    input_tok = torch.tensor([vocab[SOS_TOKEN]], dtype=torch.long, device=DEVICE)

    decoded_ids: List[int] = []

    for _ in range(max_len - 1):
        logits, hidden = model.decoder(input_tok, hidden)
        next_tok = logits.argmax(dim=-1)          # [1]
        token_id = int(next_tok.item())

        if token_id == eos_idx:
            break

        decoded_ids.append(token_id)
        input_tok = next_tok

    return decode_ids(decoded_ids, inv_vocab)


def translate_lines_with_model(lines: List[str]) -> List[str]:
    """
    Convenience helper for pipeline: translate a list of JP lines.
    """
    out: List[str] = []
    for line in lines:
        try:
            out.append(translate_text_with_model(line))
        except Exception:
            # If *anything* goes wrong for one line, fall back gracefully.
            out.append(placeholder_translate(line))
    return out


def placeholder_translate(text: str) -> str:
    """
    Fallback used by the pipeline if the model can't be loaded.
    Just echoes the JP text with a prefix so the CLI still works.
    """
    return f"[EN translation of]: {text}"
