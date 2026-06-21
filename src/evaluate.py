"""
模型评估模块
功能：测试集评估、混淆矩阵、各类别指标（精确率/召回率/F1）、ROC曲线
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report,
                              roc_curve, auc)
from sklearn.preprocessing import label_binarize
import os

# 配置中文字体
for _fname in ['SimHei', 'Microsoft YaHei', 'KaiTi']:
    for _f in fm.fontManager.ttflist:
        if _f.name == _fname:
            matplotlib.rcParams['font.family'] = _f.name
            matplotlib.rcParams['font.sans-serif'] = [_f.name]
            break
    else:
        continue
    break
matplotlib.rcParams['axes.unicode_minus'] = False


@torch.no_grad()
def evaluate_on_test(model, test_loader, device):
    """在测试集上评估模型"""
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(predicted.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    return all_probs, all_preds, all_targets


def compute_metrics(all_preds, all_targets, class_names):
    """计算分类指标"""
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, average=None, labels=range(len(class_names))
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro'
    )

    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'per_class': []
    }

    for i, name in enumerate(class_names):
        metrics['per_class'].append({
            'class': name,
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        })

    return metrics


def plot_confusion_matrix(all_preds, all_targets, class_names,
                          normalize=False, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(class_names)))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        fmt = '.2f'
        title = '归一化混淆矩阵'
    else:
        fmt = 'd'
        title = '混淆矩阵'

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '比例' if normalize else '数量'},
                ax=ax, linewidths=0.5)
    ax.set_xlabel('预测类别', fontsize=13)
    ax.set_ylabel('真实类别', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {save_path}")
    plt.close()


def plot_roc_curves(all_probs, all_targets, class_names, save_path=None):
    """绘制ROC曲线（多分类，OvR策略）"""
    n_classes = len(class_names)
    targets_bin = label_binarize(all_targets, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(targets_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='随机猜测', alpha=0.5)
    ax.set_xlabel('假正率 (FPR)', fontsize=13)
    ax.set_ylabel('真正率 (TPR)', fontsize=13)
    ax.set_title('ROC曲线（OvR）', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC曲线已保存至: {save_path}")
    plt.close()


def plot_per_class_metrics(metrics, save_path=None):
    """绘制各类别指标柱状图"""
    classes = [m['class'] for m in metrics['per_class']]
    precision = [m['precision'] for m in metrics['per_class']]
    recall = [m['recall'] for m in metrics['per_class']]
    f1 = [m['f1'] for m in metrics['per_class']]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='精确率 (Precision)',
                    color='#2E86AB', edgecolor='white')
    bars2 = ax.bar(x, recall, width, label='召回率 (Recall)',
                    color='#A23B72', edgecolor='white')
    bars3 = ax.bar(x + width, f1, width, label='F1分数',
                    color='#F18F01', edgecolor='white')

    # 标注数值
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.2f}',
                ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.2f}',
                ha='center', va='bottom', fontsize=8)
    for bar in bars3:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.2f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel('分数', fontsize=13)
    ax.set_title('各类别分类指标对比', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"各类别指标图已保存至: {save_path}")
    plt.close()


def print_metrics_report(metrics, class_names):
    """打印评估报告"""
    print("\n" + "=" * 70)
    print("测试集评估报告")
    print("=" * 70)
    print(f"\n总体准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"宏平均精确率 (Macro Precision): {metrics['macro_precision']:.4f}")
    print(f"宏平均召回率 (Macro Recall): {metrics['macro_recall']:.4f}")
    print(f"宏平均F1 (Macro F1): {metrics['macro_f1']:.4f}")

    print(f"\n{'类别':<12} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'样本数':<8}")
    print("-" * 50)
    for m in metrics['per_class']:
        print(f"{m['class']:<12} {m['precision']:<10.4f} "
              f"{m['recall']:<10.4f} {m['f1']:<10.4f} {m['support']:<8}")


if __name__ == '__main__':
    print("评估模块已就绪")
