from pathlib import Path
import re
import jieba
import hanlp
from collections import Counter, defaultdict
from utils.logger import logger
import argparse
from special_characters import SPECIAL_CHARACTERS


def load_corpus(corpus_dir: Path):
    corpus_path = corpus_dir 
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


def count_frequency_jieba(corpus_dir: Path, charset: set):
    words_sum=0
    word_freq = Counter()
    char_freq=Counter()
    DF_word=Counter()  #DF(w)=包含词w的文档数量，这里采用段落数
    DF_sum=0
    char_sum=0
    for paragraph in load_corpus(corpus_dir):
        word_set=set()  # 用于统计DF_word
        for sentence in SPECIAL_CHARACTERS.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            words = jieba.cut(sentence)
            filtered_words = [word for word in words if all(char in charset for char in word)]
            word_set.update(filtered_words)
            for word in filtered_words:
                words_sum += 1
                for char in word:
                    char_sum += 1
                    char_freq[char] += 1
            word_freq.update(filtered_words)
        DF_sum += 1
        DF_word.update(word_set)

    word_freq={word: freq / words_sum for word, freq in word_freq.items()}
    DF_word={word: freq / DF_sum for word, freq in DF_word.items()}
    char_freq={char: freq / char_sum for char, freq in char_freq.items()}
    return word_freq,char_freq,DF_word

def count_frequency_hanlp(corpus_dir: Path, charset: set):
    words_sum=0
    word_freq = Counter()
    char_freq=Counter()
    char_sum=0
    DF_word=Counter()
    DF_sum=0
    
    for paragraph in load_corpus(corpus_dir):
        word_set=set()  # 用于统计DF_word
        for sentence in SPECIAL_CHARACTERS.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            words = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')(sentence)
            filtered_words = [word for word in words if all(char in charset for char in word)]
            word_set.update(filtered_words)
            for word in filtered_words:
                words_sum += 1
                for char in word:
                    char_sum += 1
                    char_freq[char] += 1
            word_freq.update(filtered_words)
        DF_sum += 1
        DF_word.update(word_set)

    word_freq={word: freq / words_sum for word, freq in word_freq.items()}
    DF_word={word: freq / DF_sum for word, freq in DF_word.items()}
    char_freq={char: freq / char_sum for char, freq in char_freq.items()}
    return word_freq,char_freq,DF_word


def save_frequency_to_file(freq_counter: Counter, output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        for item, freq in freq_counter.most_common():
            f.write(f"{item}\t{freq}\n")
    logger.info(f"Saved frequency data to {output_path}")





def main(corpus_dir: Path, charset_path: Path, output_dir: Path,segmentation_tool: str = "jieba"):
    charset = load_charset(charset_path)
    logger.info(f"Loaded charset from {charset_path}")
    
    if segmentation_tool == "jieba":
        word_freq, char_freq, DF_word = count_frequency_jieba(corpus_dir, charset)
        logger.info("Counted frequencies using Jieba")
        save_frequency_to_file(word_freq, output_dir / "word_freq_jieba.txt")
        save_frequency_to_file(char_freq, output_dir / "char_freq.txt")
        save_frequency_to_file(DF_word, output_dir / "DF_word.txt")

    elif segmentation_tool == "hanlp":
        word_freq, char_freq, DF_word = count_frequency_hanlp(corpus_dir, charset)
        logger.info("Counted frequencies using HanLP")
        save_frequency_to_file(word_freq, output_dir / "word_freq_hanlp.txt")
        save_frequency_to_file(char_freq, output_dir / "char_freq.txt")
        save_frequency_to_file(DF_word, output_dir / "DF_word.txt")
        
    else:
        logger.error(f"Unsupported segmentation tool: {segmentation_tool}")
        raise ValueError(f"Unsupported segmentation tool: {segmentation_tool}")

    # word_freq_jieba, char_freq_jieba = count_frequency_jieba(corpus_dir, charset)
    # logger.info("Counted frequencies using Jieba")

    # word_freq_hanlp, char_freq_hanlp = count_frequency_hanlp(corpus_dir, charset)
    # logger.info("Counted frequencies using HanLP")

    # output_dir.mkdir(parents=True, exist_ok=True)

    # save_frequency_to_file(word_freq_jieba, output_dir / "word_freq_jieba.txt")
    # save_frequency_to_file(char_freq_jieba, output_dir / "char_freq_jieba.txt")
    # save_frequency_to_file(word_freq_hanlp, output_dir / "word_freq_hanlp.txt")
    # save_frequency_to_file(char_freq_hanlp, output_dir / "char_freq_hanlp.txt")

def filter_low_frequency_words(freq_counter: Counter, threshold: float):
    """过滤掉频率低于阈值的词
    累积 Token 覆盖率法，语言服从齐普夫定律（Zipf's law）
    截断标准：
    95% 覆盖率：得到的是核心精炼常用词表（通常几千到一万词，足够日常交流）。
    98% ~ 99% 覆盖率：得到的是完备常用词表（工业界预训练常用词的标准线，通常 2万 ~ 5万词）。
    99% 之后：剩下的几万甚至几十万长尾词，90% 都是错别字、生僻专有名词或分词碎片，直接全部扔掉。
    
    df
    
    """
    filtered_counter = Counter({word: freq for word, freq in freq_counter.items() if freq >= threshold})
    logger.info(f"Filtered words with frequency below {threshold}")
    return filtered_counter



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=Path, default="./corpus")
    parser.add_argument("--charset_path", type=Path, default="./charset.txt")
    parser.add_argument("--output_dir", type=Path, default="./output")
    parser.add_argument("--segmentation_tool", type=str, default="jieba")
    args = parser.parse_args()
    main(args.corpus_dir, args.charset_path, args.output_dir, args.segmentation_tool)