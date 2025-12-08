import json
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF


# ================== BMES-теги ==================

BMES_TAGS = ["B", "M", "E", "S"]
TAG2ID = {t: i for i, t in enumerate(BMES_TAGS)}
ID2TAG = {i: t for t, i in TAG2ID.items()}


# ================== Модель FEMSegV3 ==================

class FEMSegV3(nn.Module):
    """
    CNN + BiLSTM + TransformerEncoder + CRF
    Должна совпадать по архитектуре с train_femseg_v3.py
    """

    def __init__(
        self,
        vocab_size: int,
        char_emb_dim: int = 256,
        cnn_out_channels: int = 256,
        cnn_kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        lstm_hidden_dim: int = 512,
        lstm_layers: int = 2,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.3,
        num_tags: int = len(BMES_TAGS),
    ):
        super().__init__()

        self.char_emb = nn.Embedding(vocab_size, char_emb_dim, padding_idx=0)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=char_emb_dim,
                    out_channels=cnn_out_channels,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in cnn_kernel_sizes
            ]
        )

        self.lstm = nn.LSTM(
            input_size=cnn_out_channels * len(cnn_kernel_sizes),
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        lstm_out_dim = lstm_hidden_dim * 2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_out_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

        self.fc = nn.Linear(lstm_out_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T]  – индексы символов
        mask: [B, T] – bool (True = не PAD)
        """
        emb = self.char_emb(x)          # [B,T,emb]
        emb = self.dropout(emb)

        # CNN
        h = emb.transpose(1, 2)         # [B,emb,T]
        cnn_outs = [F.relu(conv(h)) for conv in self.convs]
        h = torch.cat(cnn_outs, dim=1)  # [B, C*len(k), T]
        h = h.transpose(1, 2)           # [B,T,C*len(k)]
        h = self.dropout(h)

        # BiLSTM
        h, _ = self.lstm(h)             # [B,T,2*hidden]
        h = self.dropout(h)

        # TransformerEncoder
        h = self.transformer(h, src_key_padding_mask=~mask)  # [B,T,2*hidden]
        h = self.dropout(h)
        return h

    def forward_logits(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x, mask)
        emissions = self.fc(feats)
        return emissions


# ================== BMES → CSE-морфемы ==================

def bmes_to_cse(chars: List[str], tags: List[str]) -> str:
    """
    chars: список символов слова
    tags:  BMES-теги той же длины

    Возвращает морфемную строку формата:
      "терін@@ і@@ міз@@ ден"
    """
    morphs: List[str] = []
    cur = ""

    for ch, t in zip(chars, tags):
        if t == "B":
            if cur:
                morphs.append(cur)
            cur = ch
        elif t == "M":
            cur += ch
        elif t == "E":
            cur += ch
            morphs.append(cur)
            cur = ""
        elif t == "S":
            if cur:
                morphs.append(cur)
            morphs.append(ch)
            cur = ""

    if cur:
        morphs.append(cur)

    return "@@ ".join(morphs)


# ================== Обёртка как "токенизатор" ==================

class KazMorphSegmentor:
    """
    Обёртка над FEMSegV3 для удобного использования.

    Пример:
        seg = KazMorphSegmentor(char2id_path, ckpt_path, use_cuda=True)
        seg.segment_word("мектептерімізден")
        seg.segment_text("Мен мектепке барамын")
    """

    def __init__(
        self,
        char2id_path: str,
        ckpt_path: str,
        max_len: int = 256,   # для совместимости с старыми вызовами, можно игнорировать
        use_cuda: bool = True,
    ):
        # --- загружаем словарь символов ---
        with open(char2id_path, encoding="utf-8") as f:
            self.char2id: Dict[str, int] = json.load(f)

        self.pad_id = self.char2id.get("<pad>", 0)
        self.unk_id = self.char2id.get("<unk>", 1)

        vocab_size = len(self.char2id)

        # --- создаём модель ---
        self.device = torch.device(
            "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        )
        self.model = FEMSegV3(vocab_size=vocab_size).to(self.device)

        # --- загружаем веса ---
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    # --------- внутренний encode ---------

    def _encode_chars(self, text: str):
        chars = list(text)
        ids = [self.char2id.get(ch, self.unk_id) for ch in chars]
        x = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)  # [1,T]
        mask = x != self.pad_id
        return chars, x, mask

    # --------- сегментация одного слова ---------

    def segment_word(self, word: str) -> str:
        """
        Возвращает морфемную сегментацию слова в формате CSE:
          "мектеп@@ тер@@ і@@ міз@@ ден"
        """
        word = word.strip()
        if not word:
            return ""

        chars, x, mask = self._encode_chars(word)

        with torch.no_grad():
            emissions = self.model.forward_logits(x, mask)
            best_paths = self.model.crf.decode(emissions, mask=mask)
        tag_ids = best_paths[0]
        tags = [ID2TAG[int(t)] for t in tag_ids]

        return bmes_to_cse(chars, tags)

    # --------- сегментация предложения ---------

    def segment_text(self, text: str) -> str:
        """
        Сегментирует предложение ПОСЛОВНО.
        Выходной формат:
          "Мен | мек@@ теп@@ ке | бар@@ а@@ мын"
        – слова разделены " | ", внутри слова морфемы через '@@ '.
        """
        text = text.strip()
        if not text:
            return ""

        words = text.split()
        out_words: List[str] = []

        for w in words:
            seg_w = self.segment_word(w)
            out_words.append(seg_w)

        return " | ".join(out_words)
