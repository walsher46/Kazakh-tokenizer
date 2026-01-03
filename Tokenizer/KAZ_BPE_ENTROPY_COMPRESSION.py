import math
from collections import Counter
from tqdm import tqdm
import sentencepiece as spm
from transformers import MT5Tokenizer, AutoTokenizer

#   ИСХОДНЫЙ ДАТАСЕТ -  СЕГМЕНТИРОВАННЫЙ ТЕКСТ STEM+АФФИКСЫ

# Пути
corpus_path = "/content/drive/MyDrive/KAZAKH_BPE/segmented_2.txt"
spm_model_path = "/content/drive/MyDrive/KAZAKH_BPE/kazakh_bpe.model"


# Загрузка корпуса
#with open(corpus_path, "r", encoding="utf-8") as f:
    #lines = f.readlines()
#!mv /content/drive/MyDrive/KAZAKH_BPE/qazaq_synthetic_qa.csv /content/drive/MyDrive/KAZAKH_BPE/qazaq_synthetic_qa.txt
file_path = "/content/drive/MyDrive/KAZAKH_BPE/segmented_2.txt"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()  # Список строк

# Загрузка токенизаторов
sp = spm.SentencePieceProcessor()
sp.load(spm_model_path)

mt5_tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

# Загрузка стандартного MiniLM токенизатора
minilm_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def analyze_tokenizer_with_compression(tokenizer, lines, name=""):
    print(f"\n🔍 Анализируем {name}:")

    total_chars = sum(len(line) for line in lines)  # Общее количество символов до токенизации

    # Токенизация всех предложений
    all_tokens = []
    for line in tqdm(lines, desc=f"Токенизирую для {name}"):
        tokens = tokenizer.tokenize(line)
        all_tokens.extend(tokens)

    # Подсчёт токенов
    token_counts = Counter(all_tokens)
    total_tokens = sum(token_counts.values())
    vocab_size = len(token_counts)

    # Теоретическая информация (логарифм словаря)
    theoretical_bits_per_token = math.log2(vocab_size)

    # Реальная энтропия токенов
    entropy = -sum((count / total_tokens) * math.log2(count / total_tokens) for count in token_counts.values())

    # Общая информация
    total_info_theoretical = theoretical_bits_per_token * total_tokens
    total_info_real = entropy * total_tokens

    # КОМПРЕССИЯ
    compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0

    # Печать результатов
    print(f"📚 Реальный размер словаря корпуса: {vocab_size}")
    print(f"✍️ Общее количество токенов: {total_tokens}")
    print(f"🔠 Общее количество символов до токенизации: {total_chars}")
    print(f"📉 Коэффициент компрессии (символы/токены): {compression_ratio:.4f}")
    print(f"🧠 Теоретическая Bits per token (log2 словаря): {theoretical_bits_per_token:.4f}")
    print(f"📈 Реальная Bits per token (энтропия): {entropy:.4f}")
    print(f"💾 Общая информация (теоретическая): {total_info_theoretical/1e6:.2f} Мбит")
    print(f"💾 Общая информация (реальная): {total_info_real/1e6:.2f} Мбит")

    return {
        "vocab_size": vocab_size,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "compression_ratio": compression_ratio,
        "theoretical_bpt": theoretical_bits_per_token,
        "real_bpt": entropy,
        "theoretical_total_info": total_info_theoretical,
        "real_total_info": total_info_real,
    }

# Предположим, что у вас есть список строк (lines) и токенизаторы:
# sp (SentencePiece), mt5_tokenizer, minilm_tokenizer

# Пример вызова для SentencePiece:
results_sp = analyze_tokenizer_with_compression(sp, lines, name="SentencePiece BPE")

# Пример вызова для mT5:
results_mt5 = analyze_tokenizer_with_compression(mt5_tokenizer, lines, name="mT5 Стандартный")

# Пример вызова для MiniLM:
results_minilm = analyze_tokenizer_with_compression(minilm_tokenizer, lines, name="MiniLM Стандартный")


import matplotlib.pyplot as plt
import seaborn as sns

def plot_token_distribution(token_counts, name=""):
    # Получаем 20 самых частых токенов
    common_tokens = token_counts.most_common(40)
    tokens, counts = zip(*common_tokens)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(tokens), y=list(counts), palette="viridis")
    plt.xticks(rotation=90)
    plt.title(f"Токенизация {name} - Частота токенов")
    plt.xlabel("Токены")
    plt.ylabel("Частота")
    plt.show()


# Исправление: Разворачиваем вложенные списки в один
sp_tokenized = [token for line in lines for token in sp.encode_as_pieces(line)]  # SentencePiece
mt5_tokenized = [token for line in lines for token in mt5_tokenizer.tokenize(line)]  # mT5
minilm_tokenized = [token for line in lines for token in minilm_tokenizer.tokenize(line)]  # MiniLM


# Печать распределения для каждого токенизатора
plot_token_distribution(Counter(sp_tokenized), name="SentencePiece BPE")
plot_token_distribution(Counter(mt5_tokenized), name="mT5 Стандартный")
plot_token_distribution(Counter(minilm_tokenized), name="MiniLM Стандартный")




