# lexicon_CH_4_synthtext
为了生成适配synthtext的用于生成中文文字图片的，要求符合中文自然分布的词表

## 使用算法
主要使用统计学方法？

### ngram
n取 1 - 6？


### 词频统计
阈值？：200？
freq(w) ≥ θ_freq  （千万字级语料建议 θ_freq = 5–10，宁宽勿窄，精筛交给 PMI 与熵）？


### PMI
阈值？： 10/ 20 ？12∼15？ 的 (n-1) 次幂？

PPMI？nPMI?PMIK(2,3)?


对数似然比替代PMI？10.83？
大津法（Otsu's Method）？

### 左右熵
阈值？：1.5？

对于独立成行的片段词汇，引入虚拟边界符 <s> 和 <\/s>

### TopWORDS?


### 停用词表
Zipf 定律
自然语言中，词的出现频率与其频率排名成反比。
f(r)∝1/r^s(C/r)
r：词的频率排名（第1名、第2名、第3名……）
f(r)：排名为r的词的出现频率
s：指数，英语等自然语言中通常接近 1
最高频(top 50?)的字词添加进表
词头词尾黑名单？

哈工大停用词表,百度停用词表、SCU 停用词表


### 子串抑制
独立频次折减法

动态规划全局回剪​

### 子串消解
邻接依赖消解
文本跨度非极大值抑制
有向无环图全局路径消解

### 词频TF /文档频率DF 统计  

下限截断：设置 DF≥3∼5（防止生僻孤立词进表）；
爆发比上限拦截：计算 TF/DF，如果比值异常偏大（如 >10且 DF偏小），判定为单篇刷屏词，直接剔除；
全局通用度加权（可选）：如果需要衡量词的通用度，应该乘以 文档覆盖率（Document Ratio）：Doc_Ratio(w)= DF(w)/N覆盖的文章越多（
Doc_Ratio越高），代表这个词在真实社会越通用。

### Unigram LM/ PE / WordPiece
可能不适合当前场景？会出现半个词之类？


## 参考词表
《义务教育常用词表（草案）》（2019，15,114 词，四级分级）
《现代汉语常用词表》约 5.6 万词

## 已实现脚本

### build_backup_lexicon.py —— 备用词表

从 `corpus/wiki_zh`（JSON Lines，取 `title`）与 `corpus/xdhycd`（取 `【词头】`）提取条目标题，按 `charset.txt` 白名单过滤后合并去重，写入 `output/lexicon（non-frequency）.txt`（约 94 万条，无词频信息）。

```bash
python build_backup_lexicon.py --wiki_dir ./corpus/wiki_zh --dict_dir ./corpus/xdhycd --charset_path ./charset.txt
```

### evaluate_thresholds.py —— 阈值扫描评测

在 `output/word_freq.csv` / `output/DF_word.csv` 上扫描 `cumulative_token_coverage` 与 `filter_by_df` 的阈值，与《现代汉语常用词表（第2版）》比对计算 Precision / Recall / F1，结果写 `output/threshold_sweep.csv`。默认扫描 CTC=0.5~0.999、DF=1e-6~1e-1；最优 F1 ≈ 47.7%（df=1e-6, ctc=0.95）。

```bash
python evaluate_thresholds.py
```

## 中文语料：
CLUECorpusSmall ：
wiki2019zh
news2016zh
webText2019zh
comment2019zh

真实中文 OCR 数据集（RCTW、LSVT、ReCTS、MLT、MTWI、ArT、COCO-Text）的人工标注
TC-STR（有正式论文）
Chen et al., "Traditional Chinese Synthetic Datasets Verified with Labeled Data for Scene Text Recognition"（ICPR Workshop DLVDR 2022, arXiv:2111.13327）​，我下载了全文核实其采样设计，这是词表路线的典型：

词表来源：台湾教育部辞典 + 维基百科条目标题，共 107.6 万个词、12108 个字符——注意这个选源的巧思：招牌/条目式文本天然接近“词典词”，而不是新闻长句；


## 关于采样
∝freq^0.5/0.75采样？

反频率加权



子串频次虚高问题
优化（Reduced N-gram / 词频扣减）：在统计子串频次时，应当减去其超序列（Super-sequence）的出现频次，即采用项贡献频次（Term Contributed Frequency, TCF），若一个子序列完全依附于父序列存在，则直接将其剔除(Unsupervised Overlapping Feature Selection for Conditional Random Fields Learning in Chinese Word Segmentation - ACL Anthology)

高频虚词与黏着字污染
现象：中文中的“的”、“了”、“是在”、“进行”等高频助词/介词极易与前后实词黏着，形成“有的”、“完成了”、“是在进行”等伪词，常规模型很难将这些依附助词切开
。
优化：
黑名单/停用字惩罚：在候选词首尾出现高频黏着虚词时，对其进行边界惩罚或直接利用前后缀规则剔除
。
邻接变化数（Accessor Variety, AV）：如果计算连续概率的信息熵开销过大，也可以统计左右独立不同字符的数量（即离散版的分支多样性）作为快速替代



现代技术路线的进阶补充
如果单纯依靠静态统计特征仍无法完全满足对语义完整性的要求，目前前沿研究通常有两种升级路径：
大模型语义验证（LLACA 范式）：利用轻量/预训练大语言模型生成候选片段，再使用一元 PMI（Unigram PMI）对大模型输出进行边界一致性过滤，能大幅减少死板统计导致的切分歧义(Segment First or Comprehend First?Explore the Limit of Unsupervised Word Segmentation with Large Language Models - arXiv
)
。
熵驱动的子词合并（Entropy-Driven Pre-Tokenization）：先用左右熵和 PMI 确定高置信度的物理词界，再在此约束下运行 SentencePiece / BPE 算法进行高频合并，既能保留高频双字/三字完整词，又能避免跨词界乱合并(Entropy-Driven Pre-Tokenization for Byte-Pair Encoding - arXiv
)(基于中文语料库构建常用词表与Tokenizer词表的论文指南与技术演进报告
)



## 现成优质词库
THUOCL（清华大学开放中文词库）：包含 IT、金融、财经、饮食、成语等几十万高质量分门别类的词汇。
搜狗细胞词库（导出词表）：包含网络热词、全国地名、品牌名、明星人名、菜名等，非常适合做招牌、广告、场景文字生成。
Jieba 分词核心词典（dict.txt.big）：带词频标记的数万高质量现代汉语词汇。


## 字符表
来自paddle ocr 的Chinese dict


使用现有分词工具
jieba，hanlp等。。。



# 参考论文
《Traditional Chinese Synthetic Datasets Verified with Labeled Data for Scene Text Recognition》
台湾教育部辞典与维基百科条目标题

Jaderberg et al., MJSynth（NIPS DL Workshop 2014 / IJCV 2016）
 Hunspell 词典 ∩ 真实 STR 数据集的 ground truth（IC03、SVT、IIIT）。

A CNN Based Scene Chinese Text Recognition Algorithm With Synthetic Data Engine
  generates representative character images according to the fonts use frequency of Chinese texts.



synthtiger采用策略为
"Languages such as Chinese and Japanese use a large number of characters. A synthesized dataset for such a language often lacks enough amount of samples for rare characters. To deal with this problem, SynthTIGER conducts character distribution augmentation with the probability p 
c.d.
​
 . When the augmentation is triggered, it randomly chooses a character from vocabulary and samples a word having that character."