import sys
from pathlib import Path

# 让仓库根目录可被导入（utils 等位于仓库根目录，而本脚本在 core/ 下）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jieba
import hanlp
from collections import Counter
from utils.logger import logger
import argparse
# from special_chars import SPECIAL_CHARACTERS  # 已改为整段分词，不再按标点 split，暂不需要
import csv
import math
import regex
from tqdm import tqdm

def load_corpus(corpus_dir: Path):
    """
    按空行读取段落
    """
    for txt_file in tqdm(sorted(corpus_dir.rglob("*.txt")), total=len(list(corpus_dir.rglob("*.txt"))), desc="Loading corpus"):
        with txt_file.open("r", encoding="utf-8") as f:

            para_lines = []

            for line in f:
                line = line.strip()

                if line:
                    para_lines.append(line)
                             
                elif para_lines:
                    paragraph = "".join(para_lines)
                    para_lines = []
                    yield paragraph
                    logger.info(f"Loaded paragraph from {txt_file}: {paragraph.strip()[:30]}...")

            if para_lines:
                yield "".join(para_lines)
                logger.info(f"Loaded paragraph from {txt_file}: {''.join(para_lines).strip()[:30]}...")


def clean_corpus(paragraph: str):
    """
    corpus来源为https://github.com/CLUEbenchmark/CLUE，已经做过清洗,就不清洗了
    """
    
    
    
    paragraph = regex.sub(r"\p{Z}+", "", paragraph)   # 只去掉空白/分隔符，保留标点符号
    return paragraph


def load_charset(charset_path: Path):
    """
    使用来自paddocr的中文字符集
    """
    with charset_path.open("r", encoding="utf-8") as f:
        charset = set([line.strip() for line in f if line.strip()])
    return charset


def count_frequency(corpus_dir: Path, charset: set, segment: str):
    """统计词频、字频与文档频率。

    segment: 分词器，接受句子字符串，返回词的可迭代对象。
    """
    word_freq = Counter()
    char_freq = Counter(dict.fromkeys(charset, 0))
    DF_word = Counter()  # DF(w)=包含词w的文档数量，这里采用段落数
    words_sum = 0
    DF_sum = 0
    char_sum = 0

    if segment == "jieba":
        # # 仅在非 Windows 系统启用并行分词
        # if sys.platform != "win32":
        #     try:
        #         jieba.enable_parallel(10)
        #     except Exception:
        #         pass
        segment_tool = lambda sents: [list(jieba.cut(s)) for s in sents]
        logger.info("Using jieba for segmentation")
    elif segment == "hanlp":
        hanlp_tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)   
        segment_tool = lambda sents: hanlp_tok(sents)    #hanlp支持批量处理
        logger.info("Using hanlp for segmentation")
    else:
        logger.error(f"Unsupported segmentation tool: {segment}")
        raise ValueError(f"Unsupported segmentation tool: {segment}")

    for paragraph in tqdm(iterable=load_corpus(corpus_dir), desc="Processing paragraphs"):
        paragraph = clean_corpus(paragraph)
        word_set = set()  # 用于统计DF_word
        
        # sentences = [
        #     s.strip()
        #     for s in SPECIAL_CHARACTERS.split(paragraph)
        #     if s.strip()
        # ]
        
        # sentences = [s for s in regex.split(r"(?<=[\p{P}\p{S}])", paragraph) if s]  # 按标点/符号切句，标点保留在句尾
        
        sentence_split_pattern = r"(?<=[。！？；.!?])"

        sentences = [
            s for s in regex.split(sentence_split_pattern, paragraph)
            if s
        ]
        
        if not sentences:
            continue
        
        segmented_sentences = segment_tool(sentences)
        for words in segmented_sentences:
            logger.info(f"Segmented words: {words}")
            filtered_words = [
                w for w in words
                if w and all(c in charset for c in w)
            ]
            if not filtered_words:
                continue

            words_sum += len(filtered_words)
            word_freq.update(filtered_words)
            word_set.update(filtered_words)

            merged_chars = "".join(filtered_words)
            char_sum += len(merged_chars)
            char_freq.update(merged_chars)

        
        if word_set:
            DF_word.update(word_set)
        DF_sum += 1

    logger.info(f"Total words: {words_sum}, Total chars: {char_sum}, Total paragraphs: {DF_sum}")
    save_frequency_to_file(word_freq, Path("word_freq.csv"))
    save_frequency_to_file(char_freq, Path("char_freq.csv"))
    save_frequency_to_file(DF_word, Path("DF_word.csv"))
    return word_freq, words_sum, char_freq, char_sum, DF_word, DF_sum


def save_frequency_to_file(freq_counter, output_path: Path, output_dir: Path = Path("output")):
    freq_sum = sum(freq_counter.values())
    output_path = output_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(freq_counter.items(), key=lambda kv: kv[1], reverse=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for item, freq in items:
            ratio = freq / freq_sum if freq_sum else 0.0
            writer.writerow([item, freq, ratio])
    logger.info(f"Saved frequency data to {output_path}")


def save_word_list(words, output_dir: Path = Path("output"), filename: str = "lexicon.txt"):
    """把最终词表（仅词，一行一个）写入文本文件。"""
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        if words:
            f.write("\n".join(words) + "\n")
    logger.info(f"Saved {len(words)} words to {output_path}")


def cumulative_token_coverage(freq_counter: Counter, threshold: float = 0.99):
    """累积覆盖率法过滤低频词（语言服从齐普夫定律 Zipf's law）。

    按词频从高到低排序，逐个累加 token 占比，直到累积覆盖率 >= threshold 为止，
    返回覆盖率达标的高频词子集（保持频率降序）。

    截断标准：
    95% 覆盖率：核心精炼常用词表（通常几千到一万词，足够日常交流）。
    98% ~ 99% 覆盖率：完备常用词表（工业界预训练常用词的标准线，通常 2万 ~ 5万词）。
    99% 之后：剩下的长尾词多为错别字、生僻专有名词或分词碎片，直接全部扔掉。
    """
    total = sum(freq_counter.values())
    if total <= 0:
        logger.info("Empty frequency counter, nothing to filter")
        return Counter(), []

    filtered_counter = Counter()
    filtered_words = list()

    cumulative = 0

    for word, freq in freq_counter.most_common():
        filtered_counter[word] = freq
        filtered_words.append(word)
        cumulative += freq
        if cumulative / total >= threshold:
            break

    logger.info(
        f"Kept {len(filtered_counter)} words, covering {cumulative / total:.2%} "
        f"of total tokens (threshold={threshold})"
    )

    save_frequency_to_file(filtered_counter, Path("word_freq_ctc.csv"))
    return filtered_counter, filtered_words


def filter_by_df(DF_counter: Counter, total_docs: int, threshold: float = 0.00001):
    """Filter words by document frequency.

    Args:
        DF_counter: Document frequency counter.
        total_docs: Total number of documents.
        threshold: Minimum document frequency threshold.

    Returns:
        Filtered document frequency counter.

    threshold: 0.00001 means that the word must appear in at least 0.001% of the documents to be kept.
    """
    rdf = {}
    for word, df in DF_counter.items():
        if df / total_docs >= threshold:
            rdf[word] = df / total_docs
    save_frequency_to_file(rdf, Path("rdf.csv"))
    filtered_words = list(rdf.keys())
    logger.info(
        f"Kept {len(filtered_words)} words with document frequency >= {threshold} "
        f"({len(filtered_words) / len(DF_counter):.2%} of total words)"
    )
    return rdf, filtered_words


def filter(word_freq: Counter, DF_counter: Counter, total_docs: int):
    """


    Args:
        word_freq (Counter): _description_
        DF_counter (Counter): _description_
        total_docs (int): _description_
    """
    filtered_DF, filtered_words = filter_by_df(DF_counter, total_docs, threshold=0.0001)
    logger.info(f"Filtered document frequencies with {len(filtered_DF)} documents")

    word_freq = Counter({w: word_freq[w] for w in filtered_words if w in word_freq})

    filtered_word_freq, filtered_words = cumulative_token_coverage(word_freq, threshold=0.99)
    logger.info(f"Filtered word frequencies with {len(filtered_word_freq)} words")

    filtered_DF = Counter({w: filtered_DF[w] for w in filtered_words if w in filtered_DF})
    return filtered_word_freq, filtered_DF, filtered_words


# def compute_juilland_d(total_freq: Counter, sum_sq: Counter, total_docs: int):
#     """Juilland's D：D = 1 - V/sqrt(n-1)，V = sigma/f_bar 为变异系数。

#     total_freq: 每词总频次 Σf
#     sum_sq: 每词在各文档中频次的平方和 Σf_i^2
#     """
#     def juilland_d(df: int, sq: int) -> float:
#         if total_docs < 2:
#             return 1.0
#         if df <= 0:
#             return 0.0
#         mean = df / total_docs
#         variance = sq / total_docs - mean * mean
#         if variance < 0:
#             variance = 0.0
#         coeff_var = math.sqrt(variance) / mean
#         return max(0.0, 1.0 - coeff_var / math.sqrt(total_docs - 1))

#     d_index = {}
#     for word, df in total_freq.items():
#         d_index[word] = juilland_d(df, sum_sq.get(word, 0))
#     return d_index


def main(corpus_dir: Path, charset_path: Path, output_dir: Path, segmentation_tool: str = "jieba"):
    charset = load_charset(charset_path)
    logger.info(f"Loaded charset from {charset_path}")
    logger.info(f"Counted frequencies using {segmentation_tool}")


    word_freq, words_sum, char_freq, char_sum, DF_word, DF_sum = count_frequency(corpus_dir, charset, segmentation_tool)
    logger.info(f"Total words: {words_sum}, Total chars: {char_sum}, Total paragraphs: {DF_sum}")

    filtered_word_freq, filtered_DF, filtered_words = filter(word_freq, DF_word, DF_sum)

    save_word_list(filtered_words, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=Path, default="./corpus")
    parser.add_argument("--charset_path", type=Path, default="./characters.txt")
    parser.add_argument("--output_dir", type=Path, default="./output")
    parser.add_argument("--segmentation_tool", '-s', type=str, default="jieba", choices=["jieba", "hanlp"], help="Segmentation tool to use: 'jieba' or 'hanlp'")

    args = parser.parse_args()
    main(args.corpus_dir, args.charset_path, args.output_dir, args.segmentation_tool)
