"""指标（Metric）与统计量（Stats）的骨架实现。

设计要点：
1. 统计量用 StatSpec 的「三元描述」定义：(聚合方式 mode, 聚合范围 scope, 聚合单位 unit)。
   例：word_freq = count×global（出现一次计一次）；df_word = binary×global（每段落至多一次）；
   group_tf = count×group（将文档分组后，保留每组计数）。
2. 指标 Metric 只声明 requires（统计量基础名）+ 单值 threshold + 过滤方向 direction。
3. 分组口径属于指标（group_key），允许多口径并存：同一趟遍历按 (基础名, 分组键)
   各维护一份统计量，key 记作 group_tf@file / group_tf@source，互不覆盖。
4. count_word_frequency 使用 required_instances(metrics) 求并集来驱动「按需累计」：
   没有 GROUP 统计就不建临时 sqlite，不声明 word_positions 就不记位置。
"""

import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Iterable

from utils.logger import logger


# ===== 旧的 metric stub（按仓库规则注释保留，不删除）=====
# class
#
#
#
# class metric:
#     def __init__(self):
#         pass
#
#     def compute(self):
#         pass
#
#     def filter(self, threshhold: float):
#         pass
#
#     def require(self):
#         pass


class Mode(str, Enum):
    """聚合方式：一个单位内怎么计数。"""
    COUNT = "count"           # 出现一次计一次（词频）
    BINARY = "binary"         # 每个单位内至多计一次（文档频）
    SUM_SQ = "sum_sq"         # 单位计数的平方和（算方差 / CV）
    POSITIONS = "positions"   # 记录出现位置（ARF）
    PAIRS = "pairs"           # 共现词对（PMI / PKL）
    SIZE = "size"             # 单位长度，作归一化分母


class Scope(str, Enum):
    """聚合范围：直接全局汇总，还是先按组保留供组间指标回看。"""
    GLOBAL = "global"
    GROUP = "group"


@dataclass(frozen=True)
class StatSpec:
    """一个统计量的定义：(聚合方式, 聚合范围, 聚合单位)。"""
    mode: Mode
    scope: Scope = Scope.GLOBAL
    unit: str = "doc"          # token / char / doc / group / window / corpus


DEFAULT_GROUP_KEY = "file"     # 分组键默认值：按 txt 文件分组

REGISTRY: dict[str, StatSpec] = {
    # —— 全局统计 ——
    "word_freq": StatSpec(Mode.COUNT, Scope.GLOBAL, "token"),
    "char_freq": StatSpec(Mode.COUNT, Scope.GLOBAL, "char"),
    "df_word": StatSpec(Mode.BINARY, Scope.GLOBAL, "doc"),
    "word_sum_sq": StatSpec(Mode.SUM_SQ, Scope.GLOBAL, "doc"),
    "word_positions": StatSpec(Mode.POSITIONS, Scope.GLOBAL, "token"),
    "co_occur": StatSpec(Mode.PAIRS, Scope.GLOBAL, "window"),
    # —— 组内统计（多口径并存；生产由临时 sqlite 倒排提供 word -> [(组, f_i)]）——
    "group_tf": StatSpec(Mode.COUNT, Scope.GROUP, "group"),
    "group_df": StatSpec(Mode.BINARY, Scope.GROUP, "group"),
    "group_size": StatSpec(Mode.SIZE, Scope.GROUP, "group"),
    # —— 归一化分母 / 规模量 ——
    "total_words": StatSpec(Mode.SIZE, Scope.GLOBAL, "corpus"),   # 语料级标量
    "total_docs": StatSpec(Mode.SIZE, Scope.GLOBAL, "corpus"),    # 语料级标量
    "total_groups": StatSpec(Mode.SIZE, Scope.GROUP, "corpus"),   # 组数随分组键变化，故为 GROUP
}


def stat_key(name: str, group_key: str | None = None) -> str:
    """统计量实例的 key：GLOBAL 用基础名；GROUP 追加 @分组键，如 group_tf@source。"""
    return name if group_key is None else f"{name}@{group_key}"


class Stats:
    """统计量容器：按 stat_key 存取；具体结构由对应 StatSpec 的 mode/scope 决定。

    - GLOBAL + COUNT/BINARY/SUM_SQ/SIZE → Counter / 标量
    - GLOBAL + POSITIONS               → {词: [位置, ...]}
    - GLOBAL + PAIRS                   → Counter（键为 (a, b)）
    - GROUP_SIZE (group_size@<gkey>)   → {组 id: 组大小}
    - GROUP_TF   (group_tf@<gkey>)     → 工厂 callable()，每次调用返回新的流式迭代器
                                          (词, [(组 id, f_i), ...])；倒排由临时 sqlite 提供，
                                          索引扫描不可回放，故存「工厂」而非迭代器本身。
    """

    def __init__(self):
        self._data: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()


def required_instances(metrics: Iterable["Metric"]) -> list[tuple[str, str | None]]:
    """把指标列表解析成需要实例化的统计量实例 (基础名, 分组键)。

    供 count_word_frequency 按需累计：返回值即「这一趟遍历需要维护哪些累加器」。
    GLOBAL 统计的分组键为 None；GROUP 统计取指标自己的 group_key（缺省 DEFAULT_GROUP_KEY）。
    """
    instances: set[tuple[str, str | None]] = set()
    for m in metrics:
        for name in m.requires:
            spec = REGISTRY.get(name)
            if spec is None:
                raise KeyError(f"{m.name} 声明了未注册的统计量: {name}")
            if spec.scope is Scope.GROUP:
                instances.add((name, m.group_key or DEFAULT_GROUP_KEY))
            else:
                instances.add((name, None))
    return sorted(instances, key=lambda item: (item[0], item[1] or ""))


class Metric(ABC):
    """指标基类：声明所需统计量 → compute 计算 → 按单值阈值过滤。"""

    name: ClassVar[str] = "metric"
    requires: ClassVar[tuple[str, ...]] = ()      # 计算所需的变量（统计量基础名）
    group_key: ClassVar[str | None] = None        # 分组口径；None 表示沿用 DEFAULT_GROUP_KEY
    direction: ClassVar[str] = "ge"               # "ge" 保留 >= 阈值；"le" 保留 <= 阈值
    level: ClassVar[str] = "word"                 # "word" 词级（可组合成词表）；"pair" 词对级

    def __init__(self, threshold: float):
        self.threshold = threshold                # 单个阈值

    def require(self) -> tuple[str, ...]:
        return self.requires

    def instance_key(self, name: str) -> str:
        """把 requires 里的基础名解析成本指标实际使用的统计量 key。"""
        spec = REGISTRY[name]
        if spec.scope is Scope.GROUP:
            return stat_key(name, self.group_key or DEFAULT_GROUP_KEY)
        return name

    @abstractmethod
    def compute(self, stats: Stats) -> dict:
        """纯计算：统计量 → {key: 指标值}；词对指标的 key 为 (a, b)。"""

    def filter(self, values: dict) -> set:
        """基类默认实现：逐值比较单阈值；全局排序类指标（CTC）覆盖它。"""
        if self.direction == "ge":
            return {k for k, v in values.items() if v >= self.threshold}
        return {k for k, v in values.items() if v <= self.threshold}

    def run(self, stats: Stats) -> set:
        return self.filter(self.compute(stats))


# ===== 逐词指标 =====

class RDFMetric(Metric):
    """相对文档频 RDF(w) = DF(w) / N；越大越通用。"""

    name = "rdf"
    requires = ("df_word", "total_docs")
    direction = "ge"

    def compute(self, stats: Stats) -> dict:
        total_docs = stats["total_docs"]
        return {w: df / total_docs for w, df in stats["df_word"].items()}


class IDFMetric(Metric):
    """逆文档频 IDF(w) = log(N / (DF(w) + 1))；越大越罕见。"""

    name = "idf"
    requires = ("df_word", "total_docs")
    direction = "ge"

    def compute(self, stats: Stats) -> dict:
        total_docs = stats["total_docs"]
        return {w: math.log(total_docs / (df + 1)) for w, df in stats["df_word"].items()}


class CVMetric(Metric):
    """变异系数 CV(w) = σ(w) / μ(w)；越小说明跨文档分布越稳（越通用）。"""

    name = "cv"
    requires = ("word_freq", "word_sum_sq", "total_docs")
    direction = "le"

    def compute(self, stats: Stats) -> dict:
        total_docs = stats["total_docs"]
        sum_sq = stats["word_sum_sq"]
        out: dict[str, float] = {}
        for w, freq in stats["word_freq"].items():
            mean = freq / total_docs
            if mean <= 0:
                continue
            variance = max(sum_sq.get(w, 0) / total_docs - mean * mean, 0.0)
            out[w] = math.sqrt(variance) / mean
        return out


# ===== 全局排序指标（阈值语义不同于逐值比较）=====

class CTCMetric(Metric):
    """累积覆盖率截断 CTC：阈值是目标覆盖率，不是逐值下界，故覆盖 filter。"""

    name = "ctc"
    requires = ("word_freq",)

    def compute(self, stats: Stats) -> dict:
        return dict(stats["word_freq"])

    def filter(self, values: dict) -> set:
        total = sum(values.values())
        if total <= 0:
            return set()
        kept: set = set()
        cumulative = 0
        for w, freq in sorted(values.items(), key=lambda kv: kv[1], reverse=True):
            kept.add(w)
            cumulative += freq
            if cumulative / total >= self.threshold:
                break
        return kept


# ===== 组内指标（多口径并存）=====

class GriesDPMetric(Metric):
    """Gries DP = ½ Σ |f_i/f − s_i|，衡量词在组间分布的偏离程度；越小越均匀。

    group_size@<gkey> 为 {组 id: 组大小}；group_tf@<gkey> 为工厂，调用后得到
    (词, [(组 id, f_i), ...]) 的流式迭代器；s_i = size_i / total_words。
    DP_norm = DP / (1 − min s_i)（Lijffijt & Gries, 2012）。
    """

    name = "gries_dp"
    requires = ("word_freq", "group_tf", "group_size", "total_words")
    direction = "le"

    def __init__(self, threshold: float, group_key: str | None = None, normalize: bool = False):
        super().__init__(threshold)
        self.group_key = group_key or DEFAULT_GROUP_KEY
        self.normalize = normalize

    def compute(self, stats: Stats) -> dict:
        word_freq = stats["word_freq"]
        total_words = stats["total_words"]
        sizes = stats[self.instance_key("group_size")]     # {组 id: 组大小}
        postings_factory = stats[self.instance_key("group_tf")]   # 工厂：() -> 迭代器
        if total_words <= 0 or not sizes:
            return {}

        s = {gid: size / total_words for gid, size in sizes.items()}
        norm_denom = 1.0 - min(s.values())

        out: dict[str, float] = {}
        # 每次调用取新迭代器（工厂），因此可被多个指标/多次计算重复消费
        for word, pairs in postings_factory():
            freq = word_freq.get(word, 0)
            if freq <= 0:
                continue
            abs_sum = 0.0
            covered = 0.0
            for gid, fi in pairs:
                if fi:
                    abs_sum += abs(fi / freq - s[gid])
                    covered += s[gid]
            abs_sum += 1.0 - covered
            value = 0.5 * abs_sum
            if self.normalize and norm_denom > 0:
                value /= norm_denom
            out[word] = value
        return out


# ===== 复合词表指标 =====

class UniversalityMetric(Metric):
    """通用度（张普《词频统计的新概念和新方法》）：把语料分为 k 组，逐词取各组频率 f_i。

    T = ( (Σ f_i^{1/t}) / k )^t，t 默认 2；未出现的组 f_i=0，贡献 0。
    T 越大越通用（跨组分布越均衡、总量越大）。
    """

    name = "universality"
    requires = ("group_tf", "total_groups")
    direction = "ge"

    def __init__(self, threshold: float, group_key: str | None = None, t: int = 2):
        super().__init__(threshold)
        self.group_key = group_key or DEFAULT_GROUP_KEY
        self.t = t

    def compute(self, stats: Stats) -> dict:
        k = stats[self.instance_key("total_groups")]          # 组数
        postings_factory = stats[self.instance_key("group_tf")]
        if not k or k <= 0:
            return {}

        out: dict[str, float] = {}
        for word, pairs in postings_factory():               # pairs = [(gid, f_i), ...]
            root_sum = math.fsum(fi ** (1.0 / self.t) for _, fi in pairs if fi > 0)
            out[word] = (root_sum / k) ** self.t
        return out


class UsageMetric(Metric):
    """使用度 U = F × D（传统复合指标）：F 为总词频，D = t_i / T 为分布率。

    t_i 为出现该词的文档数（df_word），T 为语料文档总数（total_docs）。
    归一化：U_i = F_i·D_i / Σ_j (F_j·D_j)；U 越大越常用。
    """

    name = "usage"
    requires = ("word_freq", "df_word", "total_docs")
    direction = "ge"

    def compute(self, stats: Stats) -> dict:
        word_freq = stats["word_freq"]
        df_word = stats["df_word"]
        total_docs = stats["total_docs"]
        if total_docs <= 0:
            return {}

        raw = {w: f * (df_word.get(w, 0) / total_docs) for w, f in word_freq.items()}
        total = math.fsum(raw.values())
        if total <= 0:
            return {}
        return {w: v / total for w, v in raw.items()}


# ===== 词对指标 =====

class PMIMetric(Metric):
    """点互信息 PMI(a,b) = log( c(a,b)·N / (c(a)·c(b)) )；作用于词对（level="pair"）。"""

    name = "pmi"
    requires = ("co_occur", "word_freq", "total_words")
    direction = "ge"
    level = "pair"

    def compute(self, stats: Stats) -> dict:
        co_occur = stats["co_occur"]
        word_freq = stats["word_freq"]
        total_words = stats["total_words"]
        out: dict[tuple, float] = {}
        for pair, c_ab in co_occur.items():
            a, b = pair
            c_a, c_b = word_freq.get(a, 0), word_freq.get(b, 0)
            if c_a <= 0 or c_b <= 0 or c_ab <= 0:
                continue
            out[pair] = math.log((c_ab * total_words) / (c_a * c_b))
        return out


def apply_metrics(stats: Stats, metrics: list[Metric]) -> Counter:
    """依次求交形成最终词表；顺序即优先级（复刻现有「先 DF 后 CTC」）。"""
    base = stats["word_freq"]
    kept = set(base)
    for m in metrics:
        if m.level != "word":
            continue                      # 词对级指标不参与词表求交
        kept &= set(m.run(stats))
    logger.info(f"apply_metrics: {len(metrics)} 个指标 → 保留 {len(kept)}/{len(base)} 词")
    return Counter({w: base[w] for w in kept})
