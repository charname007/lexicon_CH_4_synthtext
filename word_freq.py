"""统计执行层：语料读取 + 按 metrics 驱动的多口径统计。

从 build_lexicon.py 迁移而来：
- 语料读取：load_charset / clean_corpus / load_corpus / iter_paragraph_batches
- 分组倒排：GroupStore / open_group_store / group_key_of
- 统计主循环：count_word_frequency（按 required_instances(metrics) 按需累计）
- 读取辅助：read_group_sizes / read_total_groups / iter_group_postings / make_postings_factory

设计要点（追加式倒排方案）：
1. batch 不跨文件（一个 batch 只装同一 txt 文件的段落）；单级分组。
2. 每个 group_key 一张组表 groups__{gkey} + 一张纯追加倒排表 postings__{gkey}，共用一个临时 sqlite。
3. 遍历里只 append，永不判断组边界：段落 -> doc_tf -> 累进 buf[(gkey,label)] -> 批末落库。
4. size 与 postings 在同一事务内增量更新，保证 Σ size == total_words。
5. 读取侧以「工厂」形式提供倒排迭代器：索引顺序扫描不可回放，故每次调用返回新迭代器。
"""

import csv
import json
import os
import random
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

# import jieba    # 已弃用分词器：统一改用 hanlp（按规则注释保留，不删除）
import hanlp
# import regex    # 已弃用按句切分：hanlp 改为直接批量输入段落
import torch
from tqdm import tqdm

from utils.logger import logger
from metrics import DEFAULT_GROUP_KEY, Stats, required_instances, stat_key


# SENT_SPLIT = r"(?<=[。！？；.!?])"          # [已弃用] 按句末标点切句：hanlp 改为直接批量输入段落
_GKEY_RE = re.compile(r"^[A-Za-z0-9_]+$")   # 分组键允许的字符（防表名注入）


def load_charset(charset_path: Path):
    """
    使用来自paddle ocr的中文字符集
    """
    with charset_path.open("r", encoding="utf-8") as f:
        charset = set([line.strip() for line in f if line.strip()])
    return charset


def clean_corpus(paragraph: str):
    """
    corpus来源为https://github.com/CLUEbenchmark/CLUE，已经做过清洗,就不清洗了
    """
    # paragraph = regex.sub(r"\p{Z}+", "", paragraph)   # 只去掉空白/分隔符，保留标点符号
    return paragraph


def iter_paragraph_batches(txt_file: Path, batch_size: int):
    """读取单个 txt 文件，按空行切段，同一文件内累积到 batch_size 个段落再一次性 yield。

    hanlp 等分词器支持批量处理，按 batch 输出可摊薄逐次调用分词器的开销。
    yield: (txt_file, [paragraph, ...])，一个 batch 只包含同一个 txt_file 的段落。
    """
    with txt_file.open("r", encoding="utf-8") as f:

        para_lines = []
        batch = []

        for line in f:
            line = line.strip()

            if line:
                para_lines.append(line)

            elif para_lines:
                paragraph = "".join(para_lines)
                para_lines = []
                batch.append(paragraph)

                if len(batch) >= batch_size:
                    logger.info(f"Yielding batch of {len(batch)} paragraphs from {txt_file}")
                    yield txt_file, batch
                    batch = []

        if para_lines:
            batch.append("".join(para_lines))

        if batch:
            logger.info(f"Yielding batch of {len(batch)} paragraphs from {txt_file}")
            yield txt_file, batch


def load_corpus(corpus_dir: Path, batch_size: int = 64, txt_files: list[Path] | None = None):
    """顺序遍历所有 txt（batch 不跨文件），yield (txt_file, [paragraph, ...])。

    corpus来源：https://github.com/CLUEbenchmark/CLUE
    包含 news2016zh / webText2019zh / wiki2019zh / comments2019zh 等子语料。

    txt_files 为 None 时自行 rglob 并随机打乱；否则按传入列表遍历
    （调用方已负责是否打乱，避免与分块映射不一致）。
    """
    if txt_files is None:
        txt_files = list(corpus_dir.rglob("*.txt"))
        random.shuffle(txt_files)
    for txt_file in tqdm(txt_files, total=len(txt_files), desc="Loading corpus"):
        yield from iter_paragraph_batches(txt_file, batch_size)


# ===== 分组倒排（临时 sqlite）=====

def _validate_gkey(gkey: str) -> None:
    if not _GKEY_RE.match(gkey):
        raise ValueError(f"非法分组键（仅允许字母/数字/下划线）: {gkey!r}")


class GroupStore:
    """sqlite 的组表 / 倒排表封装。

    - group_id(gkey,label)：内存缓存，首次才 INSERT OR IGNORE + 回查组 id。
    - add(gkey,label,tf)：同事务内 UPDATE size += sum(tf.values()) 后 executemany 追加 postings。
    - db_path：sqlite 文件路径（keep=True 时供后续调试脚本重连）。
    """

    def __init__(self, conn: sqlite3.Connection, db_path: str | None = None):
        self.conn = conn
        self.db_path = db_path
        self._cache: dict[tuple[str, str], int] = {}

    def group_id(self, gkey: str, label: str) -> int:
        key = (gkey, label)
        if key in self._cache:
            return self._cache[key]
        self.conn.execute(
            f"INSERT OR IGNORE INTO groups__{gkey}(label) VALUES (?)", (label,)
        )
        gid = self.conn.execute(
            f"SELECT group_id FROM groups__{gkey} WHERE label = ?", (label,)
        ).fetchone()[0]
        self._cache[key] = gid
        return gid

    def add(self, gkey: str, label: str, tf: Counter) -> None:
        """把一个 (gkey, label) 的词频累进组表与倒排表（纯追加）。"""
        size = sum(tf.values())
        if size <= 0:
            return
        gid = self.group_id(gkey, label)
        self.conn.execute(
            f"UPDATE groups__{gkey} SET size = size + ? WHERE group_id = ?", (size, gid)
        )
        self.conn.executemany(
            f"INSERT INTO postings__{gkey}(word, group_id, fi) VALUES (?, ?, ?)",
            ((w, gid, c) for w, c in tf.items()),
        )


@contextmanager
def open_group_store(gkeys, keep: bool = False, db_path: str | None = None):
    """打开 sqlite：为每个 gkey 建 groups__/postings__ 表及覆盖索引。

    参数：
        gkeys: 分组键列表
        keep: True 时出 with 后**不删除** sqlite 文件（供后续调试脚本重连）
        db_path: 指定 sqlite 路径；None 则用临时文件
    生命周期覆盖「统计主循环 + 所有指标消费」；keep=False 时出 with 后删除临时库。
    """
    if db_path is None:
        tmp = tempfile.NamedTemporaryFile(prefix="group_tf_", suffix=".sqlite", delete=False)
        tmp.close()
        db_path = tmp.name
    else:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = MEMORY")
    try:
        for gkey in gkeys:
            _validate_gkey(gkey)
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS groups__{gkey} "
                "(group_id INTEGER PRIMARY KEY, label TEXT NOT NULL UNIQUE, "
                "size INTEGER NOT NULL DEFAULT 0)"
            )
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS postings__{gkey} "
                "(word TEXT NOT NULL, group_id INTEGER NOT NULL, fi INTEGER NOT NULL)"
            )
            # 覆盖索引：让读取侧 ORDER BY word 走索引顺序扫描，避免临时 B-tree
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_postings__{gkey} "
                f"ON postings__{gkey}(word, group_id, fi)"
            )
        conn.commit()
        logger.info(f"Opened group sqlite: {db_path} (gkeys={list(gkeys)}, keep={keep})")
        yield GroupStore(conn, db_path)
    finally:
        conn.commit()
        conn.close()
        if not keep:
            try:
                os.unlink(db_path)
            except OSError:
                logger.warning(f"Could not remove group sqlite: {db_path}")
        else:
            logger.info(f"Kept group sqlite for later tuning: {db_path}")


def _relative_label(txt_file: Path, corpus_root: Path | None) -> str:
    if corpus_root is not None:
        try:
            return str(txt_file.relative_to(corpus_root))
        except ValueError:
            pass
    return str(txt_file)


# ===== 定点分块（chunk 口径：文件随机打乱后每 N 个一组；N 可编进口径名，如 "chunk50"）=====

FILES_PER_GROUP_DEFAULT = 20                 # 裸 "chunk" 口径的默认粒度：约 20 个文件一组

_CHUNK_GKEY_RE = re.compile(r"^chunk(?:_?(\d+))?$")    # 匹配 "chunk" / "chunk20" / "chunk_20"

_FILE_CHUNKS: dict[str, dict[Path, str]] = {}          # 口径名 -> (文件 -> 组标签)


def is_chunk_gkey(gkey) -> bool:
    """该口径名是否属于 chunk 家族（"chunk" 或 "chunk<N>"）。"""
    return isinstance(gkey, str) and _CHUNK_GKEY_RE.match(gkey) is not None


def chunk_size_of(gkey: str, default: int = FILES_PER_GROUP_DEFAULT) -> int:
    """从口径名解析每组文件数：裸 "chunk" → default；"chunk50"/"chunk_50" → 50。"""
    m = _CHUNK_GKEY_RE.match(gkey)
    if m is None:
        raise ValueError(f"非 chunk 口径: {gkey!r}")
    return int(m.group(1)) if m.group(1) else default


def build_file_chunks(txt_files, files_per_group: int = FILES_PER_GROUP_DEFAULT,
                      shuffle: bool = True, seed: int | None = None) -> dict[Path, str]:
    """把文件列表（随机打乱后）按 files_per_group 个一组切块，返回 {文件: 组标签}。

    - shuffle=True → 先 random.shuffle 再切块（每次跑分组不同，不可复现）；
    - 组标签形如 chunk_0000，按切块序号编号。
    """
    if files_per_group <= 0:
        raise ValueError(f"files_per_group 必须为正整数: {files_per_group!r}")
    files = list(txt_files)
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(files)
    return {f: f"chunk_{i // files_per_group:04d}" for i, f in enumerate(files)}


def set_file_chunks(mapping: dict[str, dict[Path, str]]) -> None:
    """登记 {口径名: (文件 -> 组标签)}，供 group_key_of 按口径名查询。"""
    global _FILE_CHUNKS
    _FILE_CHUNKS = mapping


def group_key_of(para: str, txt_file: Path, gkey, corpus_root: Path | None = None) -> str:
    """把一个段落映射到它所属的组标签。

    gkey 可为内置名或自定义 callable：
    - "file"   → 相对 corpus_dir 的 txt 路径
    - "chunk"   → 文件随机打乱后每 FILES_PER_GROUP_DEFAULT 个一组（如 chunk_0007）
    - "chunk<N>" → 同上但每组 N 个文件（如 "chunk50"），粒度随口径名走
    - "source" → 相对 corpus_dir 路径的一级子目录名（如 news2016zh_corpus）
    - callable → str(callable(para, txt_file))
    """
    if gkey is None:
        gkey = DEFAULT_GROUP_KEY
    if callable(gkey):
        return str(gkey(para, txt_file))
    if gkey == "file":
        return _relative_label(txt_file, corpus_root)
    if is_chunk_gkey(gkey):
        mapping = _FILE_CHUNKS.get(gkey)
        if mapping is None:
            raise KeyError(f"未登记的分块口径: {gkey!r}（需先 set_file_chunks）")
        label = mapping.get(txt_file)
        if label is None:
            raise KeyError(f"文件未登记分块映射: {txt_file}（口径 {gkey!r}）")
        return label
    if gkey == "source":
        label = _relative_label(txt_file, corpus_root)
        parts = Path(label).parts
        return parts[0] if len(parts) > 1 else Path(txt_file).parent.name
    raise ValueError(f"未知分组键: {gkey!r}")


# ===== 读取辅助（供指标消费；须在 open_group_store 生命周期内调用）=====

def read_group_sizes(store: GroupStore, gkey: str) -> dict[int, int]:
    """{组 id: 组大小}，用于 s_i = size_i / total_words。"""
    return dict(store.conn.execute(f"SELECT group_id, size FROM groups__{gkey}"))


def read_total_groups(store: GroupStore, gkey: str) -> int:
    return store.conn.execute(f"SELECT COUNT(*) FROM groups__{gkey}").fetchone()[0]


def iter_group_postings(store: GroupStore, gkey: str):
    """流式产出 (word, [(group_id, fi), ...])。

    按 word 有序归并；同一 (word, group_id) 的多次追加在 Python 侧累加 fi。
    不使用 GROUP BY（会强制 sqlite 建临时 B-tree）；ORDER BY word 由覆盖索引支撑。
    """
    cur_word = None
    acc: dict[int, int] = {}
    order: list[int] = []
    for word, gid, fi in store.conn.execute(
        f"SELECT word, group_id, fi FROM postings__{gkey} ORDER BY word"
    ):
        if word != cur_word:
            if cur_word is not None:
                yield cur_word, [(g, acc[g]) for g in order]
            cur_word = word
            acc = {}
            order = []
        if gid in acc:
            acc[gid] += fi
        else:
            acc[gid] = fi
            order.append(gid)
    if cur_word is not None:
        yield cur_word, [(g, acc[g]) for g in order]


def make_postings_factory(store: GroupStore, gkey: str):
    """返回工厂：每次调用产出新的倒排迭代器（索引扫描不可回放，避免单次消费耗尽）。"""
    def factory():
        return iter_group_postings(store, gkey)
    return factory


# ===== 分词器 =====

def _make_segmenter(segment: str):
    """构造分词器：接受「段落字符串列表」，返回与之一一对应同序的词列表。"""
    # [已弃用] jieba 分支（按规则注释保留，不删除）：分词统一改用 hanlp
    # if segment == "jieba":
    #     # # 仅在非 Windows 系统启用并行分词
    #     # if sys.platform != "win32":
    #     #     try:
    #     #         jieba.enable_parallel(10)
    #     #     except Exception:
    #     #         pass
    #     logger.info("Using jieba for segmentation")
    #     return lambda sents: [list(jieba.cut(s)) for s in sents]
    if segment == "hanlp":
        device_count = torch.cuda.device_count()
        if device_count > 0:
            # 有多张 GPU 时按 devices=[0,1,...] 数据并行
            hanlp_tok = hanlp.load(
                hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH,
                devices=list(range(device_count)),
            )
        else:
            # 纯 CPU 环境：不传 devices（默认 CPU），避免 devices=[] 报错
            hanlp_tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        # hanlp_tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH, device=[0,1])
        logger.info(f"Using hanlp for segmentation (cuda devices={device_count})")
        return lambda paras: hanlp_tok(paras)   # 一次性批量输入多个段落
    logger.error(f"Unsupported segmentation tool: {segment}")
    raise ValueError(f"Unsupported segmentation tool: {segment}")


# ===== 统计主循环 =====

def count_word_frequency(corpus_dir: Path, charset: set, segment: str, metrics, store: GroupStore,
                         batch_size: int = 64, files_per_group: int = FILES_PER_GROUP_DEFAULT,
                         shuffle_files: bool = True,
                         txt_files: list[Path] | None = None) -> Stats:
    """按 required_instances(metrics) 驱动「按需累计」的一趟遍历，返回填好的 Stats。

    全局统计（word_freq/char_freq/df_word/word_sum_sq/total_words/total_docs）照旧；
    组内统计在每个 batch 末尾追加写入临时 sqlite（纯 append，不判组边界）。
    GROUP 统计由传入的 store 提供，须在其生命周期内消费。
    """
    active_gkeys = sorted({gk for _, gk in required_instances(metrics) if gk})
    logger.info(f"分口径统计（active group keys）: {active_gkeys or '无'}")

    # 文件列表只扫一次盘；也允许调用方传入子集，用于抽样评估完整流程耗时。
    if txt_files is None:
        txt_files = list(corpus_dir.rglob("*.txt"))
    else:
        txt_files = list(txt_files)
    if shuffle_files:
        random.shuffle(txt_files)

    # chunk 家族口径：各自按粒度切块（列表已打乱，传 shuffle=False 避免二次打乱）
    chunk_gkeys = sorted(gk for gk in active_gkeys if is_chunk_gkey(gk))
    if chunk_gkeys:
        chunk_maps = {
            gk: build_file_chunks(txt_files, chunk_size_of(gk, files_per_group), shuffle=False)
            for gk in chunk_gkeys
        }
        set_file_chunks(chunk_maps)
        logger.info(
            "Chunk grouping: "
            + ", ".join(f"{gk}={len(set(m.values()))}组" for gk, m in chunk_maps.items())
            + f" (from {len(txt_files)} files, default={files_per_group}, shuffle={shuffle_files})"
        )

    word_freq = Counter()
    char_freq = Counter(dict.fromkeys(charset, 0))
    df_word = Counter()        # DF(w)=包含词 w 的文档（段落）数量
    word_sum_sq = Counter()    # Σ f_i^2，用于计算各文档词频的标准差 / CV
    words_sum = 0
    char_sum = 0
    total_docs = 0
    buf: dict[tuple[str, str], Counter] = defaultdict(Counter)

    seg = _make_segmenter(segment)

    for txt_file, batch in tqdm(load_corpus(corpus_dir, batch_size, txt_files), desc="Processing batches"):
        # 1) 清洗得到这一批的段落列表
        cleaned = [clean_corpus(para) for para in batch]
        if not cleaned:
            continue

        # 2) 一次批量分词：把「多个段落」一次性喂给 hanlp，返回与 cleaned 同序一一对应的词列表
        segmented = seg(cleaned)

        # 3) 逐段累计 doc_tf，同时累全局词频/字频
        doc_tfs: list[Counter] = []
        for words in segmented:
            doc_tf = Counter()

            filtered = [w for w in words if w and all(c in charset for c in w)]
            if filtered:
                words_sum += len(filtered)
                word_freq.update(filtered)
                doc_tf.update(filtered)

                merged_chars = "".join(filtered)
                char_sum += len(merged_chars)
                char_freq.update(merged_chars)

            doc_tfs.append(doc_tf)

        # 4) 段落级全局统计（每段即一篇文档；复刻旧语义：非空段落即计入 total_docs）
        for doc_tf in doc_tfs:
            if doc_tf:
                df_word.update(doc_tf.keys())
            for w, c in doc_tf.items():
                word_sum_sq[w] += c * c
            total_docs += 1

        # 5) 分组累积：只 append，不判组边界
        for p, doc_tf in zip(cleaned, doc_tfs):
            if not doc_tf:
                continue
            for gk in active_gkeys:
                label = group_key_of(p, txt_file, gk, corpus_root=corpus_dir)
                buf[(gk, label)].update(doc_tf)

        # 6) 批末落库 + 提交（避免巨型事务）
        for (gk, label), tf in buf.items():
            store.add(gk, label, tf)
        buf.clear()
        store.conn.commit()

    # 7) 填充 Stats
    stats = Stats()
    stats.set("word_freq", word_freq)
    stats.set("char_freq", char_freq)
    stats.set("df_word", df_word)
    stats.set("word_sum_sq", word_sum_sq)
    stats.set("total_words", words_sum)
    stats.set("total_docs", total_docs)
    for gk in active_gkeys:
        stats.set(stat_key("group_size", gk), read_group_sizes(store, gk))
        stats.set(stat_key("total_groups", gk), read_total_groups(store, gk))
        stats.set(stat_key("group_tf", gk), make_postings_factory(store, gk))

    logger.info(f"Total words: {words_sum}, Total chars: {char_sum}, Total paragraphs: {total_docs}")
    return stats
