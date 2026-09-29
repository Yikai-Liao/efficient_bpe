# Efficient BPE

博客[高效中文BPE实现](https://lyk-ai.com/post/2)代码仓库。

* ebpe.py   我的高效BPE实现
* bpe_eval.ipynb    评测代码
* requirements.txt  为 `bpe_eval.ipynb`的依赖，ebpe.py 本身仅以来 `tqdm` 显示进度条

2026 年的后续研究与复现实验：

* [源码审计和文献调研](benchmarks/research_archive/research.md)：原实现的复杂度、边缘问题，以及与 Re-Pair 等工作的关系。
* [共同 Python 核心比较](benchmarks/bpe_core_comparison/REPORT.md)：在相同规则语义与数据上比较端点、链式和紧凑布局。
* [实现演进与精确并行](benchmarks/bpe_core_comparison/evolution/REPORT.md)：pair key、位置池、长 token 编码、分片并行的代码、实测与取舍。

实验代码与原始实现分开存放；各报告附有数据来源、原始结果和复现命令。
