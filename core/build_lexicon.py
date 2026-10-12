from pathlib import Path
import re
import jieba
import hanlp
from collections import Counter, defaultdict
from utils.logger import logger
import argparse
from special_chars import SPECIAL_CHARACTERS
import csv
import math


def load_corpus(corpus_dir: Path):
    for txt_file in corpus_dir.rglob("*.txt"):
        with txt_file.open("r", encoding="utf-8") as f:
            content = f.read()
            paragraphs = re.split(
                r'\n\s*\n',
                content
            )
            for para in paragraphs:
                para = para.strip()
                if para:  # 过滤空行
                    yield para
                    logger.info(f"Loaded paragraph from {txt_file}: {para[:30]}...")  # 记录加载的段落信息


def clean_corpus(paragraph: str):
    """
    corpus来源为https://github.com/CLUEbenchmark/CLUE，已经做过清洗,就不清洗了
    """
    
    
    pass


def load_charset(charset_path: Path):
    with charset_path.open("r", encoding="utf-8") as f:
        charset=set([line.strip() for line in f if line.strip()])
    return charset


def count_frequency(corpus_dir: Path, charset: set, segment):
    """统计词频、字频与文档频率。

    segment: 分词器，接受句子字符串，返回词的可迭代对象。
    """
    word_freq = Counter()
    char_freq = Counter(dict.fromkeys(charset, 0))
    DF_word=Counter()  #DF(w)=包含词w的文档数量，这里采用段落数
    words_sum=0
    DF_sum=0
    char_sum=0
    
    for paragraph in load_corpus(corpus_dir):
        word_set=set()  # 用于统计DF_word
        for sentence in SPECIAL_CHARACTERS.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            filtered_words = [word for word in segment(sentence) if all(char in charset for char in word)]
            word_set.update(filtered_words)
            for word in filtered_words:
                words_sum += 1
                for char in word:
                    char_sum += 1
                    char_freq[char] += 1
            word_freq.update(filtered_words)
        DF_sum += 1
        DF_word.update(word_set)
    
    word_freq=
    logger.info(f"Total words: {words_sum}, Total chars: {char_sum}, Total paragraphs: {DF_sum}")
    save_frequency_to_file(word_freq, Path("word_freq.csv"))
    save_frequency_to_file(char_freq, Path("char_freq.csv"))
    save_frequency_to_file(DF_word, Path("DF_word.csv"))
    # word_freq={word: freq / words_sum for word, freq in word_freq.items()}
    # DF_word={word: freq / DF_sum for word, freq in DF_word.items()}
    # char_freq={char: freq / char_sum for char, freq in char_freq.items()}
    return word_freq, words_sum, char_freq, char_sum, DF_word, DF_sum


def save_frequency_to_file(freq_counter: Counter, output_path: Path):
    output_dir = Path("output")
    freq_sum=sum(freq_counter.values())
    output_path=output_dir / output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for item, freq in freq_counter.most_common():
            writer.writerow([item, freq, freq / freq_sum])
    logger.info(f"Saved frequency data to {output_path}")


def cumulative_token_coverage(freq_counter: Counter, threshold: float = 0.99):
    """累积 Token 覆盖率法过滤低频词（语言服从齐普夫定律 Zipf's law）。

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
        return Counter()

    filtered_counter = Counter()
    filtered_words=list()
    
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
    
    save_frequency_to_file(filtered_counter, output_dir / "word_freq_ctc.csv")
    return filtered_counter, filtered_words


def filter_by_df(DF_counter: Counter, total_docs: int, threshold: float = 0.0001):
    """Filter words by document frequency.

    Args:
        DF_counter: Document frequency counter.
        total_docs: Total number of documents.
        threshold: Minimum document frequency threshold.

    Returns:
        Filtered document frequency counter.
    
    threshold: 0.0001 means that the word must appear in at least 0.01% of the documents to be kept.
    """
    rdf = defaultdict(float)
    for word, df in DF_counter.items():
        if df / total_docs >= threshold:
            rdf[word] = df / total_docs
    save_frequency_to_file(rdf, Path("output/rdf.csv"))
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


def main(corpus_dir: Path, charset_path: Path, output_dir: Path,segmentation_tool: str = "jieba"):
    charset = load_charset(charset_path)
    logger.info(f"Loaded charset from {charset_path}")
    logger.info(f"Counted frequencies using {segmentation_tool}")
    if segmentation_tool == "jieba":
        segment = jieba.cut
        word_freq_file = "word_freq_jieba.csv"
        tool_label = "Jieba"
    elif segmentation_tool == "hanlp":
        segment = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
        word_freq_file = "word_freq_hanlp.csv"
        tool_label = "HanLP"
    else:
        logger.error(f"Unsupported segmentation tool: {segmentation_tool}")
        raise ValueError(f"Unsupported segmentation tool: {segmentation_tool}")


    word_freq, words_sum, char_freq, chars_sum, DF_word, DF_sum = count_frequency(corpus_dir, charset, segment)
    logger.info(f"Total words: {words_sum}, Total chars: {chars_sum}, Total paragraphs: {DF_sum}")

    filtered_word_freq, filtered_words = cumulative_token_coverage(word_freq, threshold=0.99)
    logger.info(f"Filtered word frequencies with {len(filtered_word_freq)} words")

    filtered_DF = Counter({w: DF_word[w] for w in filtered_words if w in DF_word})
    logger.info(f"Filtered document frequencies with {len(filtered_DF)} documents")

    # save_frequency_to_file(word_freq, output_dir / word_freq_file)
    # save_frequency_to_file(char_freq, output_dir / "char_freq.csv")
    # save_frequency_to_file(DF_word, output_dir / "DF_word.csv")

    # word_freq_jieba, char_freq_jieba = count_frequency_jieba(corpus_dir, charset)
    # logger.info("Counted frequencies using Jieba")

    # word_freq_hanlp, char_freq_hanlp = count_frequency_hanlp(corpus_dir, charset)
    # logger.info("Counted frequencies using HanLP")

    # output_dir.mkdir(parents=True, exist_ok=True)

    # save_frequency_to_file(word_freq_jieba, output_dir / "word_freq_jieba.txt")
    # save_frequency_to_file(char_freq_jieba, output_dir / "char_freq_jieba.txt")
    # save_frequency_to_file(word_freq_hanlp, output_dir / "word_freq_hanlp.txt")
    # save_frequency_to_file(char_freq_hanlp, output_dir / "char_freq_hanlp.txt")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=Path, default="./corpus")
    parser.add_argument("--charset_path", type=Path, default="./charset.txt")
    parser.add_argument("--output_dir", type=Path, default="./output")
    parser.add_argument("--segmentation_tool", type=str, default="hanlp", choices=["jieba", "hanlp"], help="Segmentation tool to use: 'jieba' or 'hanlp'")
    
    args = parser.parse_args()
    main(args.corpus_dir, args.charset_path, args.output_dir, args.segmentation_tool)
