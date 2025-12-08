from kazmorphseg import KazMorphSegmentor

def segment_file(
    input_path,
    output_path,
    max_len=256,
    use_cuda=True,
):
    seg = KazMorphSegmentor(
        "/home/proart/PycharmProjects/PythonProject2/data/char2id_femseg_v3.json",
        "/home/proart/PycharmProjects/PythonProject2/models/femseg_v3_best.pt",
        max_len=max_len,
        use_cuda=use_cuda,
    )


    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                fout.write("\n")
                continue

            # 1) морфологическая сегментация строкой:
            #    пример: "Терін@@і | тер@@ең | ылғал@@дан@@дыр@@а@@ды@@,"
            seg_line = seg.segment_text(line)

            # 2) преобразуем в "SentencePiece-подобный" формат
            sp_tokens = []
            for word_seg in seg_line.split(" | "):
                word_seg = word_seg.strip()
                if not word_seg:
                    continue

                # морфы: ["Терін", "і"], ["тер", "ең"], ...
                morphs = [m.strip() for m in word_seg.split("@@ ") if m.strip()]
                if not morphs:
                    continue

                # первая морфа слова — с префиксом ▁
                sp_tokens.append("▁" + morphs[0])
                # остальные — без префикса
                sp_tokens.extend(morphs[1:])

            out_line = " ".join(sp_tokens)
            fout.write(out_line + "\n")

    print(f"Готово! Сохранено в: {output_path}")


def main():
    import sys
    if len(sys.argv) != 3:
        print("Использование: python kazmorphseg_batch.py input.txt output.txt")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    segment_file(input_path, output_path)


if __name__ == "__main__":
    main()
