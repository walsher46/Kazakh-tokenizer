import numpy as np
import sentencepiece as spm
from transformers import MT5Tokenizer, AutoTokenizer
from tqdm import tqdm

# ====== 1. ЗАГРУЖАЕМ ФАЙЛ ======
file_path = "/content/drive/MyDrive/KAZAKH_BPE/chunk_099.txt"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# ====== 2. ЗАГРУЖАЕМ ТОКЕНИЗАТОРЫ ======
sp_bpe = spm.SentencePieceProcessor()
sp_bpe.load("/content/drive/MyDrive/KAZAKH_BPE/kazakh_bpe.model")

sp_unigram = spm.SentencePieceProcessor()
sp_unigram.load("/content/drive/MyDrive/KAZAKH_BPE/kazakh_unigram.model")

mt5_tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
minilm_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# ====== 3. ФУНКЦИЯ АНАЛИЗА ======
def analyze_seq_lengths(tokenizer, name):
    seq_lengths = []

    for line in tqdm(lines, desc=f"Tokenizing with {name}"):
        tokens = tokenizer.tokenize(line)
        seq_lengths.append(len(tokens))

    seq_lengths = np.array(seq_lengths)

    result = {
        "Tokenizer": name,
        "Mean length": float(seq_lengths.mean()),
        "Median": float(np.median(seq_lengths)),
        "90th percentile": float(np.percentile(seq_lengths, 90)),
        "95th percentile": float(np.percentile(seq_lengths, 95)),
        "Min": int(seq_lengths.min()),
        "Max": int(seq_lengths.max()),
        "Sentences": len(seq_lengths),
    }

    print("\n===== RESULT:", name, "=====")
    for k, v in result.items():
        print(f"{k}: {v}")

    return result


# ====== 4. ЗАПУСК ДЛЯ ВСЕХ ТОКЕНИЗАТОРОВ ======
results = []

results.append(analyze_seq_lengths(sp_bpe, "SentencePiece BPE_KAZ"))
results.append(analyze_seq_lengths(sp_unigram, "SentencePiece UNIGRAM_KAZ"))
results.append(analyze_seq_lengths(mt5_tokenizer, "mT5 Standard"))
results.append(analyze_seq_lengths(minilm_tokenizer, "MiniLM Standard"))


# ====== 5. ВЫВОД В ВИДЕ ТАБЛИЦЫ ======
import pandas as pd

df = pd.DataFrame(results)
df

