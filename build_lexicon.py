import sys
from pathlib import Path

# # 让仓库根目录可被导入（utils 等位于仓库根目录，而本脚本在 core/ 下）
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import Counter
from utils.logger import logger
import argparse
import csv
import tempfile   # 兼容薄壳 count_frequency 需要一个「临时库路径」占位

# ===== 以下依赖已随统计逻辑迁移到 word_freq.py（此处注释保留，不再使用）=====
# import jieba
# import hanlp
# import torch
# import math
# import os
# import sqlite3
# import tempfile
# import regex
# from collections import defaultdict
# from tqdm import tqdm
# from special_chars import SPECIAL_CHARACTERS  # 已改为整段分词，不再按标点 split，暂不需要

# ===== 统计执行层：语料读取 / 分组倒排 / 统计主循环（见 word_freq.py）=====
from word_freq import load_charset, count_word_frequency, open_group_store
# ===== 指标定义层（见 metrics.py）=====
from metrics import (
    DEFAULT_GROUP_KEY,
    required_instances,
    apply_metrics,
    RDFMetric,
    CTCMetric,
    CVMetric,
    GriesDPMetric,
    UniversalityMetric,
    UsageMetric,
)


# ============================================================================
# 以下「语料读取 / 分组倒排 / 旧指标实现」已迁移到 word_freq.py 与 metrics.py，
# 按仓库规则（不需要的代码不要删除，而是注释掉）注释保留，不再执行。
# ============================================================================

# def load_corpus(corpus_dir: Path):
#     """
#     按空行分割读取段落
#
#     corpus来源：https://github.com/CLUEbenchmark/CLUE
#     包含如下子语料库（总共14G语料）：
#
#     1、新闻语料 news2016zh_corpus: 8G语料，分成两个上下两部分，总共有2000个小文件。
#
#     2、社区互动-语料 webText2019zh_corpus：3G语料，包含3G文本，总共有900多个小文件。
#
#     3、维基百科-语料 wiki2019zh_corpus：1.1G左右文本，包含300左右小文件。
#
#     4、评论数据-语料 comments2019zh_corpus：2.3G左右文本，共784个小文件，包括点评评论547个、亚马逊评论227个，合并ChineseNLPCorpus的多个评论数据，清洗、格式转换、拆分成小文件。
#     """
#     for txt_file in tqdm(sorted(corpus_dir.rglob("*.txt")), total=len(list(corpus_dir.rglob("*.txt"))), desc="Loading corpus"):
#         with txt_file.open("r", encoding="utf-8") as f:
#
#             para_lines = []
#
#             for line in f:
#                 line = line.strip()
#
#                 if line:
#                     para_lines.append(line)
#
#                 elif para_lines:
#                     paragraph = "".join(para_lines)
#                     para_lines = []
#                     # yield paragraph
#                     yield txt_file, paragraph
#                     logger.info(f"Loaded paragraph from {txt_file}: {paragraph.strip()[:30]}...")
#
#             if para_lines:
#                 # yield "".join(para_lines)
#                 yield txt_file, "".join(para_lines)
#                 logger.info(f"Loaded paragraph from {txt_file}: {''.join(para_lines).strip()[:30]}...")
#
#
# def clean_corpus(paragraph: str):
#     """
#     corpus来源为https://github.com/CLUEbenchmark/CLUE，已经做过清洗,就不清洗了
#     """
#
#
#
#     # paragraph = regex.sub(r"\p{Z}+", "", paragraph)   # 只去掉空白/分隔符，保留标点符号
#     return paragraph
#
#
# def load_charset(charset_path: Path):
#     """
#     使用来自paddle ocr的中文字符集
#     """
#     with charset_path.open("r", encoding="utf-8") as f:
#         charset = set([line.strip() for line in f if line.strip()])
#     return charset
#
#
# def _open_dp_store():
#     """Gries DP 用的临时 sqlite：按语料小文件累计 posting，避免把倒排表全放内存。"""
#     tmp = tempfile.NamedTemporaryFile(prefix="gries_dp_", suffix=".sqlite", delete=False)
#     tmp.close()
#     conn = sqlite3.connect(tmp.name)
#     conn.execute("PRAGMA synchronous = OFF")
#     conn.execute("PRAGMA journal_mode = MEMORY")
#     conn.execute("CREATE TABLE docs (doc_id INTEGER PRIMARY KEY, size INTEGER NOT NULL)")
#     conn.execute("CREATE TABLE postings (word TEXT NOT NULL, doc_id INTEGER NOT NULL, fi INTEGER NOT NULL)")
#     return conn, tmp.name
#
#
# def _flush_dp_file(conn, doc_id: int, file_tf: Counter) -> bool:
#     size = sum(file_tf.values())
#     if size <= 0:
#         return False
#     conn.execute("INSERT INTO docs (doc_id, size) VALUES (?, ?)", (doc_id, size))
#     conn.executemany(
#         "INSERT INTO postings (word, doc_id, fi) VALUES (?, ?, ?)",
#         ((w, doc_id, c) for w, c in file_tf.items()),
#     )
#     return True
#
#
# def count_frequency(corpus_dir: Path, charset: set, segment: str):
#     """统计词频、字频与文档频率。
#
#     segment: 分词器，接受句子字符串，返回词的可迭代对象。
#     """
#     word_freq = Counter()
#     char_freq = Counter(dict.fromkeys(charset, 0))
#     DF_word = Counter()  # DF(w)=包含词w的文档数量，这里采用段落数,不过这里的段落应该是相互独立的文档
#     word_sum_sq = Counter()  # Σ f_i^2，用于计算各文档词频的标准差 / CV
#     # postings = defaultdict(list)  # word -> [(doc_id, f_i), ...]，供 Gries DP（已改为临时 sqlite）
#     # doc_sizes = []  # 各文档（段落）token 数 n_i，空文档不计入
#     # word_positions = defaultdict(list)  # word -> [token 全局下标, ...]，供 ARF（暂不使用）
#     dp_conn, dp_db_path = _open_dp_store()
#     file_tf = Counter()  # 当前语料小文件内词频，刷入临时库后清空
#     current_file = None
#     dp_doc_id = 0
#     words_sum = 0
#     DF_sum = 0
#     char_sum = 0
#
#     if segment == "jieba":
#         # # 仅在非 Windows 系统启用并行分词
#         # if sys.platform != "win32":
#         #     try:
#         #         jieba.enable_parallel(10)
#         #     except Exception:
#         #         pass
#         segment_tool = lambda sents: [list(jieba.cut(s)) for s in sents]
#         logger.info("Using jieba for segmentation")
#     elif segment == "hanlp":
#         hanlp_tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH, devices=list(range(torch.cuda.device_count())))
#         # hanlp_tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH,device=[0,1])
#         segment_tool = lambda sents: hanlp_tok(sents)    #hanlp支持批量处理
#         logger.info("Using hanlp for segmentation")
#     else:
#         logger.error(f"Unsupported segmentation tool: {segment}")
#         raise ValueError(f"Unsupported segmentation tool: {segment}")
#
#     for txt_file, paragraph in tqdm(iterable=load_corpus(corpus_dir), total=16605369, desc="Processing paragraphs"):
#         if current_file is not None and txt_file != current_file:
#             if _flush_dp_file(dp_conn, dp_doc_id, file_tf):
#                 dp_doc_id += 1
#             file_tf = Counter()
#         current_file = txt_file
#
#         paragraph = clean_corpus(paragraph)
#         word_set = set()  # 用于统计DF_word
#         doc_tf = Counter()  # 当前文档（段落）内各词出现次数
#
#         # sentences = [
#         #     s.strip()
#         #     for s in SPECIAL_CHARACTERS.split(paragraph)
#         #     if s.strip()
#         # ]
#
#         # sentences = [s for s in regex.split(r"(?<=[\p{P}\p{S}])", paragraph) if s]  # 按标点/符号切句，标点保留在句尾
#
#         sentence_split_pattern = r"(?<=[。！？；.!?])"
#
#         sentences = [
#             s for s in regex.split(sentence_split_pattern, paragraph)
#             if s
#         ]
#
#         if not sentences:
#             continue
#
#         segmented_sentences = segment_tool(sentences)
#         for words in segmented_sentences:
#             logger.info(f"Segmented words: {words}")
#             filtered_words = [
#                 w for w in words
#                 if w and all(c in charset for c in w)
#             ]
#             if not filtered_words:
#                 continue
#
#             # start_pos = words_sum  # ARF 全局下标，暂不使用
#             words_sum += len(filtered_words)
#             word_freq.update(filtered_words)
#             word_set.update(filtered_words)
#             doc_tf.update(filtered_words)
#             # for i, w in enumerate(filtered_words):
#             #     word_positions[w].append(start_pos + i)
#
#             merged_chars = "".join(filtered_words)
#             char_sum += len(merged_chars)
#             char_freq.update(merged_chars)
#
#
#         if word_set:
#             DF_word.update(word_set)
#         for w, c in doc_tf.items():
#             word_sum_sq[w] += c * c
#         file_tf.update(doc_tf)
#         # if doc_tf:
#         #     doc_id = len(doc_sizes)
#         #     doc_sizes.append(sum(doc_tf.values()))
#         #     for w, c in doc_tf.items():
#         #         postings[w].append((doc_id, c))
#         DF_sum += 1
#
#     if file_tf:
#         _flush_dp_file(dp_conn, dp_doc_id, file_tf)
#     dp_conn.commit()
#     dp_conn.close()
#     logger.info(f"DP postings stored in temp sqlite: {dp_db_path}")
#
#     logger.info(f"Total words: {words_sum}, Total chars: {char_sum}, Total paragraphs: {DF_sum}")
#     save_frequency_to_file(word_freq, Path("word_freq.csv"))
#     save_frequency_to_file(char_freq, Path("char_freq.csv"))
#     save_frequency_to_file(DF_word, Path("DF_word.csv"))
#     # return word_freq, words_sum, char_freq, char_sum, DF_word, DF_sum, word_sum_sq, postings, doc_sizes, word_positions
#     return word_freq, words_sum, char_freq, char_sum, DF_word, DF_sum, word_sum_sq, dp_db_path


# ============================================================================
# 兼容薄壳：旧调用方（build_lexicon_v2.py / tests）仍从本模块 import 这些旧接口。
# 实现已迁移到 word_freq.py + metrics.py，这里只做转发，保持旧签名与返回值。
# ============================================================================

def count_frequency(corpus_dir: Path, charset: set, segment: str):
    """[兼容薄壳] 旧接口：委托 word_freq.count_word_frequency，返回旧 8 元组。

    返回 (word_freq, words_sum, char_freq, char_sum, DF_word, total_docs, word_sum_sq, dp_db_path)。
    新架构下分组倒排存于临时 sqlite，随 open_group_store 的 with 退出即删除，
    不再存在持久的 dp_db_path；旧调用方只对它做 Path(...).unlink(missing_ok=True)，
    故返回一个「不存在的占位路径」即可。旧调用方仅使用 word_freq / df / total_docs。
    """
    with open_group_store([]) as store:   # 旧接口不涉及分组统计，active_gkeys 为空
        stats = count_word_frequency(corpus_dir, charset, segment, [], store)

    # 复刻旧行为：旧 count_frequency 会顺便写出这三个全局统计 CSV
    word_freq = stats["word_freq"]
    char_freq = stats["char_freq"]
    save_frequency_to_file(word_freq, Path("word_freq.csv"))
    save_frequency_to_file(char_freq, Path("char_freq.csv"))
    save_frequency_to_file(stats["df_word"], Path("DF_word.csv"))

    dp_db_path = str(Path(tempfile.gettempdir()) / "lexicon_legacy_dp.sqlite")  # 占位（不创建）
    return (
        word_freq,
        stats["total_words"],
        char_freq,
        sum(char_freq.values()),    # char_sum：所有字符的总出现次数
        stats["df_word"],
        stats["total_docs"],
        stats["word_sum_sq"],
        dp_db_path,
    )


# ============================================================================
# 以下旧指标实现已由 metrics.py 的 RDFMetric / CTCMetric / CVMetric / GriesDPMetric
# 等类取代，注释保留备查。
# ============================================================================

# [兼容薄壳还原] tests/ 多个用例仍从本模块 import cumulative_token_coverage，
# 故按旧实现原样还原（不改动逻辑）。新架构下等价物为 metrics.CTCMetric。
def cumulative_token_coverage(freq_counter: Counter, threshold: float = 0.9995):
    """累积覆盖率截断过滤低频词（语言服从齐普夫定律 Zipf's law）。

    按词频从高到低排序，逐个累加 token 占比，直到累积覆盖率 >= threshold 为止，
    返回覆盖率达标的高频词子集（保持频率降序）。

    《电磁泄漏还原图像中的中文文本识别技术研究》（2021，中文期刊）

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

    how_much_filtered = len(filtered_words) / len(freq_counter) if freq_counter else 0

    logger.info(
        f"Kept {len(filtered_counter)} words, covering {cumulative / total:.2%} "
        f"of total tokens (threshold={threshold})"
        f', which is {how_much_filtered:.2%} of total words'
    )

    save_frequency_to_file(filtered_counter, Path("word_freq_ctc.csv"))
    return filtered_counter, filtered_words


# def filter_by_df(DF_counter: Counter, total_docs: int, threshold: float = 0.00001):
#     """Filter words by document frequency.
#
#     Args:
#         DF_counter: Document frequency counter.
#         total_docs: Total number of documents.
#         threshold: Minimum document frequency threshold.
#
#     Returns:
#         Filtered document frequency counter.
#
#     threshold: 0.00001 means that the word must appear in at least 0.001% of the documents to be kept.
#     """
#     rdf = {}
#     for word, df in DF_counter.items():
#         if df / total_docs >= threshold:
#             rdf[word] = df / total_docs
#     save_frequency_to_file(rdf, Path("rdf.csv"))
#     filtered_words = list(rdf.keys())
#
#     how_much_filtered = len(filtered_words) / len(DF_counter) if DF_counter else 0
#     logger.info(
#         f"Kept {len(filtered_words)} words with document frequency >= {threshold} "
#         f"({how_much_filtered:.2%} of total words)"
#     )
#     return rdf, filtered_words
#
#
# def compute_idf(DF_counter: Counter, total_docs: int):
#     """计算逆文档频率 IDF(w) = log(N / (DF(w) + 1))，衡量词有多“罕见”。
#
#     N=total_docs 为文档总数（这里即总段落数），DF(w) 为包含词 w 的文档数。
#     DF(w)+1 是加一平滑，避免 DF=0 时除零；词越“罕见”，IDF 值越大。
#     """
#     idf = {}
#     for word, df in DF_counter.items():
#         idf[word] = math.log(total_docs / (df + 1))
#     save_frequency_to_file(idf, Path("idf.csv"))
#     logger.info(f"Computed IDF for {len(idf)} words")
#     return idf
#
#
# def compute_tf_idf(word_freq: Counter, idf: dict):
#     """计算 TF-IDF(w) = TF(w) * IDF(w)，衡量词有多“重要”。
#
#     TF(w) = 词 w 的频率 / 总词数，IDF(w) = log(N / (DF(w) + 1))。
#     """
#     total_words = sum(word_freq.values())
#     tf_idf = {}
#     for word, freq in word_freq.items():
#         tf = freq / total_words
#         tf_idf[word] = tf * idf.get(word, 0.0)
#     save_frequency_to_file(tf_idf, Path("tf_idf.csv"))
#     logger.info(f"Computed TF-IDF for {len(tf_idf)} words")
#     return tf_idf
#
#
# def compute_cv(word_freq: Counter, word_sum_sq: Counter, total_docs: int):
#     """计算变异系数 CV(w) = σ(w) / μ(w)。
#
#     无分类标签时，把每段当作一篇文档。词 w 在文档 i 中的出现次数为 f_i（未出现为 0）。
#     μ(w) = (Σ f_i) / N，σ(w) 为总体标准差 sqrt(Σ f_i² / N − μ²)。
#     CV 越小：各文档分布越稳（通用词）；CV 极大：多数文档为 0、少数文档扎堆（偏置词）。
#     """
#     cv = {}
#     if total_docs <= 0:
#         logger.info("No documents, skip CV")
#         return cv
#
#     for word, freq in word_freq.items():
#         mean = freq / total_docs
#         if mean <= 0:
#             continue
#         variance = word_sum_sq.get(word, 0) / total_docs - mean * mean
#         if variance < 0:
#             variance = 0.0
#         cv[word] = math.sqrt(variance) / mean
#
#     save_frequency_to_file(cv, Path("cv.csv"))
#     logger.info(f"Computed CV for {len(cv)} words")
#     return cv
#
#
# # def compute_gries_dp(postings: dict, doc_sizes: list, word_freq: Counter):
# #     """Gries DP：内存倒排版（段落为 part）。"""
# #     dp = {}
# #     dp_norm = {}
# #     token_total = sum(doc_sizes)
# #     if token_total <= 0 or not doc_sizes:
# #         logger.info("No documents, skip Gries DP")
# #         return dp, dp_norm
# #
# #     s = [size / token_total for size in doc_sizes]
# #     min_s = min(s)
# #     norm_denom = 1.0 - min_s
# #
# #     for word, pairs in postings.items():
# #         freq = word_freq.get(word, 0)
# #         if freq <= 0:
# #             continue
# #         abs_sum = 0.0
# #         covered_s = 0.0
# #         for doc_id, fi in pairs:
# #             abs_sum += abs(fi / freq - s[doc_id])
# #             covered_s += s[doc_id]
# #         abs_sum += 1.0 - covered_s
# #         value = 0.5 * abs_sum
# #         dp[word] = value
# #         dp_norm[word] = value / norm_denom if norm_denom > 0 else 0.0
# #
# #     save_frequency_to_file(dp, Path("dp.csv"))
# #     save_frequency_to_file(dp_norm, Path("dp_norm.csv"))
# #     logger.info(f"Computed Gries DP for {len(dp)} words")
# #     return dp, dp_norm
#
#
# def compute_gries_dp(dp_db_path: str, word_freq: Counter):
#     """Gries DP：DP(w) = 1/2 Σ |v_i - s_i|（Deviation of Proportions）。
#
#     part 为语料小文件（长短不一）：s_i = n_i / N，v_i = f_i / f。
#     posting 在临时 sqlite 中按文件累计，计算时按词扫描，算完删除临时库。
#     DP → 0：分布与文件长度成比例（通用词）；DP 大：相对长度扎堆（偏置词）。
#     DP_norm = DP / (1 - min s_i)（Lijffijt & Gries, 2012）。
#     """
#     dp = {}
#     dp_norm = {}
#     conn = sqlite3.connect(dp_db_path)
#     try:
#         doc_rows = conn.execute("SELECT doc_id, size FROM docs ORDER BY doc_id").fetchall()
#         if not doc_rows:
#             logger.info("No documents, skip Gries DP")
#             return dp, dp_norm
#
#         token_total = sum(size for _, size in doc_rows)
#         if token_total <= 0:
#             logger.info("No tokens, skip Gries DP")
#             return dp, dp_norm
#
#         s = {doc_id: size / token_total for doc_id, size in doc_rows}
#         min_s = min(s.values())
#         norm_denom = 1.0 - min_s
#
#         def flush_word(word, pairs):
#             freq = word_freq.get(word, 0)
#             if freq <= 0:
#                 return
#             abs_sum = 0.0
#             covered_s = 0.0
#             for doc_id, fi in pairs:
#                 abs_sum += abs(fi / freq - s[doc_id])
#                 covered_s += s[doc_id]
#             abs_sum += 1.0 - covered_s
#             value = 0.5 * abs_sum
#             dp[word] = value
#             dp_norm[word] = value / norm_denom if norm_denom > 0 else 0.0
#
#         current_word = None
#         pairs = []
#         for word, doc_id, fi in conn.execute(
#             "SELECT word, doc_id, fi FROM postings ORDER BY word"
#         ):
#             if word != current_word:
#                 if current_word is not None:
#                     flush_word(current_word, pairs)
#                 current_word = word
#                 pairs = [(doc_id, fi)]
#             else:
#                 pairs.append((doc_id, fi))
#         if current_word is not None:
#             flush_word(current_word, pairs)
#
#         save_frequency_to_file(dp, Path("dp.csv"))
#         save_frequency_to_file(dp_norm, Path("dp_norm.csv"))
#         logger.info(f"Computed Gries DP for {len(dp)} words from {len(doc_rows)} files")
#         return dp, dp_norm
#     finally:
#         conn.close()
#         try:
#             os.unlink(dp_db_path)
#         except OSError:
#             logger.warning(f"Could not remove DP temp file: {dp_db_path}")
#
#
# # def compute_arf(word_positions: dict, total_tokens: int):
# #     """ARF(w) = (1/v) Σ min(d_i, v)（Adjusted Relative Frequency）。
# #
# #     把全部过滤后的 token 按出现顺序连成一条环形序列。
# #     f 为词 w 的总次数，N 为总 token 数，v = N/f 为平均间隔；
# #     d_i 为相邻两次出现的距离，最后一次到第一次用 N 做首尾衔接。
# #     均匀出现时 ARF ≈ f；高度扎堆时 ARF 接近 1。
# #     """
# #     arf = {}
# #     if total_tokens <= 0:
# #         logger.info("No tokens, skip ARF")
# #         return arf
# #
# #     for word, positions in word_positions.items():
# #         f = len(positions)
# #         if f <= 0:
# #             continue
# #         if f == 1:
# #             arf[word] = 1.0
# #             continue
# #         v = total_tokens / f
# #         gap_sum = 0.0
# #         for i in range(f - 1):
# #             gap_sum += min(positions[i + 1] - positions[i], v)
# #         gap_sum += min(positions[0] + total_tokens - positions[f - 1], v)
# #         arf[word] = gap_sum / v
# #
# #     save_frequency_to_file(arf, Path("arf.csv"))
# #     logger.info(f"Computed ARF for {len(arf)} words")
# #     return arf
#
#
# def compute_pmi(co_occur: Counter, word_freq: Counter, total_tokens: int):
#     """计算点互信息 PMI(A,B) = log( p(A,B) / (p(A)·p(B)) )，衡量词对搭配紧密程度。
#
#     用计数表示：p(A,B)=c(A,B)/N，p(A)=c(A)/N，p(B)=c(B)/N，故
#     PMI(A,B) = log( c(A,B)·N / (c(A)·c(B)) )。
#     PMI 越大搭配越紧密；但 PMI 对低频词过敏感（Rare Event Bias），
#     两个各出现 1 次且恰好共现的词 PMI 高达 logN，需用 PKL 校正。
#
#     co_occur: {(A, B): 共现次数 c(A,B)}；word_freq: 词 w 的独立出现次数 c(w)；total_tokens: 总词数 N。
#     """
#     pmi = {}
#     if total_tokens <= 0:
#         logger.info("No tokens, skip PMI")
#         return pmi
#     for pair, c_ab in co_occur.items():
#         a, b = pair
#         c_a, c_b = word_freq.get(a, 0), word_freq.get(b, 0)
#         if c_a <= 0 or c_b <= 0 or c_ab <= 0:
#             continue
#         pmi[pair] = math.log((c_ab * total_tokens) / (c_a * c_b))
#     save_frequency_to_file(pmi, Path("pmi.csv"))
#     logger.info(f"Computed PMI for {len(pmi)} word pairs")
#     return pmi
#
#
# def compute_pkl(co_occur: Counter, word_freq: Counter, total_tokens: int):
#     """计算逐点 KL 散度 PKL(A,B) = p(A,B) · PMI(A,B)，校正 PMI 的低频偏差。
#
#     在 PMI 前乘联合概率 p(A,B)=c(A,B)/N 作加权：偶然共现的罕见词 PMI 虽高，
#     但 p(A,B) 极小，相乘后得分被压低；只有「既搭配紧密、又有一定出现频率」的词对才得高分。
#
#     用计数表示：PKL(A,B) = (c(A,B)/N) · log( c(A,B)·N / (c(A)·c(B)) )。
#     """
#     pkl = {}
#     if total_tokens <= 0:
#         logger.info("No tokens, skip PKL")
#         return pkl
#     for pair, c_ab in co_occur.items():
#         a, b = pair
#         c_a, c_b = word_freq.get(a, 0), word_freq.get(b, 0)
#         if c_a <= 0 or c_b <= 0 or c_ab <= 0:
#             continue
#         pkl[pair] = (c_ab / total_tokens) * math.log((c_ab * total_tokens) / (c_a * c_b))
#     save_frequency_to_file(pkl, Path("pkl.csv"))
#     logger.info(f"Computed PKL for {len(pkl)} word pairs")
#     return pkl
#
#
# def filter(word_freq: Counter, DF_counter: Counter, total_docs: int):
#     """
#
#
#     Args:
#         word_freq (Counter): _description_
#         DF_counter (Counter): _description_
#         total_docs (int): _description_
#     """
#     # 原阈值 0.0001（df/total_docs，即词需出现在 ≥1661 篇文档）。改为 1e-6（≥17 篇）：
#     # 依据 evaluate_thresholds.py 的阈值扫描，Recall 优先（参考表为 2011 年，Precision 被系统性低估），
#     # df=1e-6 + ctc=0.99 时对《现代汉语常用词表》Recall 40.87% → 86.54%。
#     filtered_DF, filtered_words = filter_by_df(DF_counter, total_docs, threshold=1e-6)
#     logger.info(f"Filtered document frequencies with {len(filtered_DF)} documents")
#
#     word_freq = Counter({w: word_freq[w] for w in filtered_words if w in word_freq})
#
#     filtered_word_freq, filtered_words = cumulative_token_coverage(word_freq, threshold=0.99)
#     logger.info(f"Filtered word frequencies with {len(filtered_word_freq)} words")
#
#     filtered_DF = Counter({w: filtered_DF[w] for w in filtered_words if w in filtered_DF})
#     return filtered_word_freq, filtered_DF, filtered_words
#
#
# # def compute_juilland_d(total_freq: Counter, sum_sq: Counter, total_docs: int):
# #     """Juilland's D：D = 1 - V/sqrt(n-1)，V = sigma/f_bar 为变异系数。
# #
# #     total_freq: 每词总频次 Σf
# #     sum_sq: 每词在各文档中频次的平方和 Σf_i^2
# #     """
# #     def juilland_d(df: int, sq: int) -> float:
# #         if total_docs < 2:
# #             return 1.0
# #         if df <= 0:
# #             return 0.0
# #         mean = df / total_docs
# #         variance = sq / total_docs - mean * mean
# #         if variance < 0:
# #             variance = 0.0
# #         coeff_var = math.sqrt(variance) / mean
# #         return max(0.0, 1.0 - coeff_var / math.sqrt(total_docs - 1))
# #
# #     d_index = {}
# #     for word, df in total_freq.items():
# #         d_index[word] = juilland_d(df, sum_sq.get(word, 0))
# #     return d_index


# ============================================================================
# 输出与编排（保留）
# ============================================================================

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


def save_word_list(word_freq: Counter, output_dir: Path = Path("output"), filename: str = "lexicon.csv"):
    """把最终词表（词、归一化频率、原始频率）写入 CSV 文件，频率用于采样。"""
    total = sum(word_freq.values())
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "normalized_frequency", "raw_frequency"])
        for word, freq in word_freq.most_common():
            writer.writerow([word, freq / total, freq])
    logger.info(f"Saved {len(word_freq)} words to {output_path}")


def main(corpus_dir: Path, charset_path: Path, output_dir: Path,
         segmentation_tool: str = "hanlp", batch_size: int = 64):
    charset = load_charset(charset_path)
    logger.info(f"Loaded charset from {charset_path}")
    logger.info(f"Counted frequencies using {segmentation_tool}")

    # 指标配置：
    # - lexicon 求交用 RDF(df/total_docs >= 1e-6) + CTC(累积覆盖率 >= 0.99)，
    #   复刻原 filter() 的「先 DF 后 CTC」顺序（apply_metrics 依次求交）。
    # - CV / GriesDP 仅计算并落 CSV，不参与求交，故其阈值不影响最终词表。
    lexicon_metrics = [RDFMetric(1e-6), CTCMetric(0.99)]
    report_metrics = [
        CVMetric(1.0),
        GriesDPMetric(1.0, group_key=DEFAULT_GROUP_KEY, normalize=False),
        UniversalityMetric(1.0, group_key=DEFAULT_GROUP_KEY, t=2),
        UsageMetric(1.0),
    ]
    all_metrics = lexicon_metrics + report_metrics

    # 由指标声明推导这一趟遍历需要维护哪些分组倒排表（GROUP 统计才建表）
    active_gkeys = sorted({gk for _, gk in required_instances(all_metrics) if gk})
    logger.info(f"Active group keys: {active_gkeys or '无'}")

    # 临时 sqlite 生命周期必须覆盖「统计主循环 + 所有指标消费」
    with open_group_store(active_gkeys) as store:
        stats = count_word_frequency(corpus_dir, charset, segmentation_tool, all_metrics, store, batch_size)

        # 全局统计落 CSV（与原 count_frequency 产出一致）
        save_frequency_to_file(stats["word_freq"], Path("word_freq.csv"))
        save_frequency_to_file(stats["char_freq"], Path("char_freq.csv"))
        save_frequency_to_file(stats["df_word"], Path("DF_word.csv"))

        # 指标中间产物（沿用原有 csv 命名）
        save_frequency_to_file(lexicon_metrics[0].compute(stats), Path("rdf.csv"))            # RDF
        save_frequency_to_file(lexicon_metrics[1].compute(stats), Path("word_freq_ctc.csv"))  # CTC 输入词频
        save_frequency_to_file(report_metrics[0].compute(stats), Path("cv.csv"))              # CV
        save_frequency_to_file(report_metrics[1].compute(stats), Path("dp.csv"))              # DP

        report_metrics[1].normalize = True                                                    # DP_norm
        save_frequency_to_file(report_metrics[1].compute(stats), Path("dp_norm.csv"))

        save_frequency_to_file(report_metrics[2].compute(stats), Path("universality.csv"))    # 通用度
        save_frequency_to_file(report_metrics[3].compute(stats), Path("usage.csv"))           # 使用度

        # 依次求交得到最终词表（必须在 with 内消费 group_tf）
        lexicon = apply_metrics(stats, lexicon_metrics)

    save_word_list(lexicon, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=Path, default="./corpus")
    parser.add_argument("--charset_path", type=Path, default="./charset.txt")
    parser.add_argument("--output_dir", type=Path, default="./output")
    parser.add_argument("--segmentation_tool", '-s', type=str, default="hanlp", choices=["hanlp"], help="分词工具（已弃用 jieba，仅支持 'hanlp'）")
    parser.add_argument("--batch_size", type=int, default=64, help="Segmentation batch size (paragraphs per batch, batch never crosses files)")

    args = parser.parse_args()
    main(args.corpus_dir, args.charset_path, args.output_dir, args.segmentation_tool, args.batch_size)
