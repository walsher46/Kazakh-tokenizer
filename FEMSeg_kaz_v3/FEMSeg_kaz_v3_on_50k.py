!pip install -q pytorch-crf

import os
import json
from typing import List, Tuple

import torch
import torch.nn as nn
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Пути и подготовка 10k корпуса
# -----------------------------
BASE_DIR = "/content/drive/MyDrive/KAZ_MORPH"
os.makedirs(BASE_DIR, exist_ok=True)

SENT_FULL_PATH = os.path.join(BASE_DIR, "kazakh_segmented_corpus_284707.txt")
SENT_50K_PATH  = os.path.join(BASE_DIR, "kazakh_segmented_corpus_50k.txt")
CHAR2ID_PATH   = os.path.join(BASE_DIR, "char2id_femseg_v3_50k.json")
MODEL_DIR      = os.path.join(BASE_DIR, "models_femseg_v3_50k")
os.makedirs(MODEL_DIR, exist_ok=True)

# если файла на 10k ещё нет — создаём
if not os.path.exists(SENT_50K_PATH):
    print("Создаём подкорпус на 50 000 предложений...")
    !head -n 10000 "$SENT_FULL_PATH" > "$SENT_10K_PATH"
else:
    print("Файл на 50k уже существует:", SENT_50K_PATH)

# -----------------------------
# 2. BMES по морфам
# -----------------------------

TAGS = ["B", "M", "E", "S"]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
ID2TAG = {i: t for t, i in TAG2ID.items()}

def morphs_to_bmes_char(word: str, morphs: List[str]) -> List[str]:
    """
    Преобразует список морфов в последовательность BMES-тегов по символам.
    """
    tags: List[str] = []
    offset = 0
    for m in morphs:
        m_len = len(m)
        if m_len == 1:
            tags.append("S")
        else:
            for i in range(m_len):
                if i == 0:
                    tags.append("B")
                elif i == m_len - 1:
                    tags.append("E")
                else:
                    tags.append("M")
        offset += m_len

    if len(tags) != len(word):
        # на всякий случай проверим длину
        # если что-то не так — лучше пропустить слово
        return []
    return tags

# -----------------------------
# 3. Разбор CSE-предложения → слова
# -----------------------------

PUNCT_TOKENS = {",", ".", "?", "!", ";", ":", "—", "-", "…", "„", "“", "«", "»", "(", ")", "[", "]"}

def line_to_word_morphs(line: str) -> List[Tuple[str, List[str]]]:
    """
    Из строки CSE-предложения делает список (word_str, morphs_list).
    Формат строки: "соңғы бес жылда ана өлім@@ і шамамен 3 есе ..."
    """
    tokens = line.strip().split()
    words = []
    cur = []

    for tok in tokens:
        # пунктуация как отдельное "слово"
        if tok in PUNCT_TOKENS:
            if cur:
                # закрываем текущее слово
                morphs = []
                for t in cur:
                    if t.endswith("@@"):
                        morphs.append(t[:-2])
                    else:
                        morphs.append(t)
                word = "".join(morphs)
                words.append((word, morphs))
                cur = []
            # пунктуация: односимвольное слово
            words.append((tok, [tok]))
            continue

        cur.append(tok)
        if not tok.endswith("@@"):
            # конец слова
            morphs = []
            for t in cur:
                if t.endswith("@@"):
                    morphs.append(t[:-2])
                else:
                    morphs.append(t)
            word = "".join(morphs)
            words.append((word, morphs))
            cur = []

    if cur:
        morphs = []
        for t in cur:
            if t.endswith("@@"):
                morphs.append(t[:-2])
            else:
                morphs.append(t)
        word = "".join(morphs)
        words.append((word, morphs))

    return words

# -----------------------------
# 4. Чтение корпуса и построение выборки (слова)
# -----------------------------

def build_samples_from_cse(path: str) -> List[Tuple[List[str], List[int]]]:
    """
    Читает CSE-предложения и возвращает список (chars, tag_ids) по словам.
    chars: список символов слова
    tag_ids: BMES-теги по символам (id)
    """
    samples = []
    total_lines = 0
    bad_words = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            word_morphs = line_to_word_morphs(line)
            for word, morphs in word_morphs:
                chars = list(word)
                tags = morphs_to_bmes_char(word, morphs)
                if not tags or len(tags) != len(chars):
                    bad_words += 1
                    continue
                tag_ids = [TAG2ID[t] for t in tags]
                samples.append((chars, tag_ids))

    print(f"Всего строк: {total_lines}")
    print(f"Слов-семплов: {len(samples)}")
    print(f"Проблемных слов (пропущено): {bad_words}")
    return samples

samples = build_samples_from_cse(SENT_50K_PATH)

# -----------------------------
# 5. Строим char2id и кодируем данные
# -----------------------------

def build_char_vocab(samples, min_freq: int = 1):
    from collections import Counter
    cnt = Counter()
    for chars, _ in samples:
        for ch in chars:
            cnt[ch] += 1

    char2id = {"<pad>": 0, "<unk>": 1}
    for ch, c in cnt.items():
        if c >= min_freq:
            char2id[ch] = len(char2id)

    print("Размер char-вокабуляра:", len(char2id))
    return char2id

char2id = build_char_vocab(samples, min_freq=1)

with open(CHAR2ID_PATH, "w", encoding="utf-8") as f:
    json.dump(char2id, f, ensure_ascii=False, indent=2)
print("char2id сохранён в:", CHAR2ID_PATH)

def encode_samples(samples, char2id):
    enc = []
    for chars, tag_ids in samples:
        ids = [char2id.get(ch, char2id["<unk>"]) for ch in chars]
        enc.append((ids, tag_ids))
    return enc

encoded = encode_samples(samples, char2id)

# -----------------------------
# 6. Train / Val split
# -----------------------------

train_data, val_data = train_test_split(encoded, test_size=0.1, random_state=42)
print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

# -----------------------------
# 7. Dataset и collate_fn
# -----------------------------

class MorphDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # (ids, tag_ids)

def collate_fn(batch):
    # batch: list of (ids, tag_ids)
    max_len = max(len(x[0]) for x in batch)
    pad_id = 0

    input_ids = []
    tag_ids   = []
    mask      = []

    for ids, tags in batch:
        l = len(ids)
        padded_ids  = ids + [pad_id] * (max_len - l)
        padded_tags = tags + [-1] * (max_len - l)  # -1 для паддинга
        m = [1] * l + [0] * (max_len - l)

        input_ids.append(padded_ids)
        tag_ids.append(padded_tags)
        mask.append(m)

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(tag_ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.bool),
    )

train_ds = MorphDataset(train_data)
val_ds   = MorphDataset(val_data)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, collate_fn=collate_fn)

# -----------------------------
# 8. Модель: BiLSTM + CRF
# -----------------------------

class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size: int, tagset_size: int,
                 emb_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, input_ids, tags=None, mask=None):
        x = self.emb(input_ids)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        emissions = self.fc(x)

        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            # decode
            best_paths = self.crf.decode(emissions, mask=mask)
            return best_paths

# -----------------------------
# 9. Обучение
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = BiLSTMCRF(vocab_size=len(char2id), tagset_size=len(TAGS)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

def evaluate(model, loader):
    model.eval()
    total_tokens = 0
    correct = 0

    with torch.no_grad():
        for input_ids, tag_ids, mask in loader:
            input_ids = input_ids.to(device)
            tag_ids   = tag_ids.to(device)
            mask      = mask.to(device)

            paths = model(input_ids, tags=None, mask=mask)  # список списков
            # приводим в tensor
            max_len = input_ids.size(1)
            pred_ids = torch.full_like(tag_ids, fill_value=-1)
            for i, seq in enumerate(paths):
                for j, t in enumerate(seq):
                    pred_ids[i, j] = t

            # считаем точность только там, где mask == 1
            mask_flat = mask.view(-1)
            gold_flat = tag_ids.view(-1)
            pred_flat = pred_ids.view(-1)

            valid = mask_flat & (gold_flat >= 0)
            total_tokens += valid.sum().item()
            correct += (gold_flat[valid] == pred_flat[valid]).sum().item()

    if total_tokens == 0:
        return 0.0
    return correct / total_tokens

num_epochs = 3

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for input_ids, tag_ids, mask in train_loader:
        input_ids = input_ids.to(device)
        tag_ids   = tag_ids.to(device)
        mask      = mask.to(device)

        loss = model(input_ids, tags=tag_ids, mask=mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    val_acc = evaluate(model, val_loader)

    print(f"[Epoch {epoch}] train_loss={avg_loss:.4f}  val_token_acc={val_acc:.4f}")

    # сохраняем чекпоинт
    ckpt_path = os.path.join(MODEL_DIR, f"femseg_v3_50k_epoch{epoch}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print("[main] checkpoint saved:", ckpt_path)


print("Обучение завершено.")
