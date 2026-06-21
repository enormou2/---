# ---
人工智能课程作业
# 基于CNN的服装图像分类研究

使用卷积神经网络（CNN）对 Fashion-MNIST 数据集进行10类服装图像分类。

## 环境要求

- Python 3.8+
- 依赖包见 `requirements.txt`

## 安装

```bash
pip install -r requirements.txt
```

## 项目结构

```
├── main.py                     # 主程序入口
├── src/
│   ├── data_preprocessing.py   # 数据加载、预处理、可视化
│   ├── model.py                # CNN / MLP 模型定义
│   ├── train.py                # 训练循环、早停、学习率调度
│   ├── evaluate.py             # 评估指标、混淆矩阵
│   └── visualize.py            # 训练曲线、特征图可视化
├── results/
│   └── figures/                # 实验输出图表
├── notebooks/                  # Jupyter 交互式笔记本
├── requirements.txt
└── README.md
```

## 运行

**训练模型：**

```bash
python main.py
```

首次运行会自动下载 Fashion-MNIST 数据集（约30MB）。在 CPU 上约需 10-15 分钟完成训练。

输出内容：训练过程日志（每轮的 loss、accuracy）、测试集准确率和各类别 F1 分数。

**生成全部实验图表：**

```bash
python quick_figures.py
```

会在 `results/figures/` 下生成 13 张图表（样本展示、训练曲线、混淆矩阵、特征图可视化、超参数对比等）。

## 实验结果

| 模型 | 测试准确率 | 参数量 |
|------|-----------|--------|
| CNN（4层） | ~92.0% | 276,554 |
| MLP（3层） | ~84.8% | 39,456,768 |
| 逻辑回归 | ~80.1% | 7,850 |

CNN 以最少的参数量取得了最优的分类性能，验证了卷积结构在图像特征提取中的高效性。

## 数据集

Fashion-MNIST（Zalando Research, 2017）：10类服装、70,000张28×28灰度图像。PyTorch 自动下载，无需手动准备。

论文：《Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms》（Xiao et al., 2017）
