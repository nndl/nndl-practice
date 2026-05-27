# nndl-practice

邱锡鹏《神经网络与深度学习：案例与实践》代码仓库。

| 目录 | 内容 |
|---|---|
| [`pytorch/`](pytorch/) | **案例与实践**（第 2 版印刷书配套）的 PyTorch 实现。8 章初版完成。 |
| [`legacy/`](legacy/) | 原 `nndl/exercise` 仓库内容，对应**理论书第 1 版**的章末编程练习（numpy / 早期 PyTorch）。 |

PaddlePaddle 版（第一版印刷书指向）：[nndl/practice-in-paddle](https://github.com/nndl/practice-in-paddle)。

## PyTorch 章节进度

| 章 | 主题 | 状态 |
|----|------|------|
| 1  | [实践基础](pytorch/chap1实践基础/)             | 初版完成 |
| 2  | [机器学习概述](pytorch/chap2机器学习概述/)      | 初版完成 |
| 3  | [线性模型](pytorch/chap3线性模型/)             | 初版完成 |
| 4  | [前馈神经网络](pytorch/chap4前馈神经网络/)      | 初版完成 |
| 5  | [卷积神经网络](pytorch/chap5卷积神经网络/)      | 初版完成 |
| 6  | [循环神经网络](pytorch/chap6循环神经网络/)      | 初版完成 |
| 7  | [网络优化与正则化](pytorch/chap7网络优化与正则化/) | 初版完成 |
| 8  | [注意力机制](pytorch/chap8注意力机制/)          | 初版完成 |

每章产出：可运行 notebook + README（实现要点）+ pytest sanity 测试。环境依赖见 [`pytorch/requirements.txt`](pytorch/requirements.txt)。

## 系列与主站

- 主站：https://nndl.ai
- 理论书（v2）+ 通识版：https://github.com/nndl/nndl
- 大模型与智能体：https://github.com/nndl/llm-beginner

## 元数据

[`_meta.yml`](_meta.yml) 是主站书目卡片的数据源；主站构建时由 [`aggregate-books.py`](https://github.com/nndl/nndl.github.io/blob/main/scripts/aggregate-books.py) 汇总。

## 历史

本仓库前身是 `nndl/exercise`（2017–2024 的"课程练习"仓库），2026 年改名为 `nndl-practice`，原章节全部归档到 [`legacy/`](legacy/)。改名后 GitHub 自动建立旧 URL `github.com/nndl/exercise/*` → 新 URL 的跳转，stars 和 forks 全部保留。
