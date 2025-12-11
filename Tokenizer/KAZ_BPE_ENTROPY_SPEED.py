import math
import time
from collections import Counter
from tqdm import tqdm
import sentencepiece as spm
from transformers import MT5Tokenizer, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Пути
corpus_path = "/content/drive/MyDrive/KAZAKH_BPE/chunk_099.txt"
spm_model_path = "/content/drive/MyDrive/KAZAKH_BPE/kazakh_bpe.model"

# Загрузка корпуса
with open(corpus_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Загрузка токенизаторов
sp = spm.SentencePieceProcessor()
sp.load(spm_model_path)

mt5_tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
minilm_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def analyze_tokenizer_with_compression(tokenizer, lines, name=""):
    print(f"\n🔍 Анализируем {name}:")

    total_chars = sum(len(line) for line in lines)
    all_tokens = []

    start_time = time.time()  # 🕒 Засекаем время токенизации

    for line in tqdm(lines, desc=f"Токенизирую для {name}"):
        tokens = tokenizer.tokenize(line)
        all_tokens.extend(tokens)

    elapsed_time = time.time() - start_time  # Общее время
    tokens_per_sec = len(all_tokens) / elapsed_time if elapsed_time > 0 else 0  # ✅ Скорость токенизации

    # Подсчёт токенов и статистика
    token_counts = Counter(all_tokens)
    total_tokens = sum(token_counts.values())
    vocab_size = len(token_counts)
    theoretical_bits_per_token = math.log2(vocab_size)
    entropy = -sum((count / total_tokens) * math.log2(count / total_tokens) for count in token_counts.values())
    total_info_theoretical = theoretical_bits_per_token * total_tokens
    total_info_real = entropy * total_tokens
    compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0

    # Вывод
    print(f"Реальный размер словаря корпуса: {vocab_size}")
    print(f"Общее количество токенов: {total_tokens}")
    print(f"Общее количество символов до токенизации: {total_chars}")
    print(f"Коэффициент компрессии (символы/токены): {compression_ratio:.4f}")
    print(f"Теоретическая Bits per token (log2 словаря): {theoretical_bits_per_token:.4f}")
    print(f"Реальная Bits per token (энтропия): {entropy:.4f}")
    print(f"Общая информация (теоретическая): {total_info_theoretical / 1e6:.2f} Мбит")
    print(f"Общая информация (реальная): {total_info_real / 1e6:.2f} Мбит")
    print(f"⚡ Скорость токенизации: {tokens_per_sec:,.2f} токенов/сек")

    return {
        "name": name,
        "vocab_size": vocab_size,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "compression_ratio": compression_ratio,
        "theoretical_bpt": theoretical_bits_per_token,
        "real_bpt": entropy,
        "theoretical_total_info": total_info_theoretical,
        "real_total_info": total_info_real,
        "tokens_per_sec": tokens_per_sec,
        "elapsed_time": elapsed_time
    }


def plot_token_distribution(token_counts, name=""):
    common_tokens = token_counts.most_common(40)
    tokens, counts = zip(*common_tokens)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(tokens), y=list(counts), palette="viridis")
    plt.xticks(rotation=90)
    plt.title(f"Токенизация {name} - Частота токенов")
    plt.xlabel("Токены")
    plt.ylabel("Частота")
    plt.show()


# === Запуск анализа ===
results_sp = analyze_tokenizer_with_compression(sp, lines, name="SentencePiece BPE_KAZ")
results_mt5 = analyze_tokenizer_with_compression(mt5_tokenizer, lines, name="mT5 Стандартный")
results_minilm = analyze_tokenizer_with_compression(minilm_tokenizer, lines, name="MiniLM Стандартный")

