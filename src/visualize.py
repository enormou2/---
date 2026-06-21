"""
可视化模块
功能：训练曲线、特征图可视化、错误案例分析、对比实验结果图
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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

RESULTS_DIR = r"C:\Users\10959\Desktop\artificial intelligence\garbage-cnn-classification\results\figures"


def plot_training_curves(history, save_path=None):
    """绘制训练/验证损失和准确率曲线"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='训练损失')
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='验证损失')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('损失 (Loss)', fontsize=12)
    ax1.set_title('训练与验证损失曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # 准确率曲线
    ax2.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='训练准确率')
    ax2.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='验证准确率')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('准确率 (Accuracy)', fontsize=12)
    ax2.set_title('训练与验证准确率曲线', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存至: {save_path}")
    plt.close()


def visualize_feature_maps(model, input_tensor, layer_names, device,
                           max_channels=16, save_path=None):
    """可视化卷积层特征图"""
    model.eval()
    with torch.no_grad():
        _, features = model(input_tensor.unsqueeze(0).to(device), return_features=True)

    n_layers = len(layer_names)
    fig, axes = plt.subplots(n_layers, max_channels,
                              figsize=(max_channels * 1.5, n_layers * 1.8))

    if n_layers == 1:
        axes = axes.reshape(1, -1)

    for row, name in enumerate(layer_names):
        if name in features:
            fm = features[name][0].cpu()  # (C, H, W)
            n_channels = min(fm.shape[0], max_channels)
            for col in range(n_channels):
                ax = axes[row, col]
                ax.imshow(fm[col], cmap='viridis')
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(f'{name}\n({fm.shape[0]}通道)',
                                  fontsize=10, fontweight='bold')
            for col in range(n_channels, max_channels):
                axes[row, col].axis('off')
        else:
            for col in range(max_channels):
                axes[row, col].axis('off')

    plt.suptitle('卷积特征图可视化（浅层→深层）', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"特征图可视化已保存至: {save_path}")
    plt.close()


def visualize_error_cases(model, test_loader, class_names, device,
                          num_errors=10, save_path=None):
    """可视化错误分类的样本"""
    model.eval()
    errors = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs_gpu = inputs.to(device)
            outputs = model(inputs_gpu)
            _, preds = outputs.max(1)

            for i in range(len(targets)):
                if preds[i] != targets[i] and len(errors) < num_errors:
                    errors.append({
                        'image': inputs[i].cpu(),
                        'true': targets[i].item(),
                        'pred': preds[i].item()
                    })
            if len(errors) >= num_errors:
                break

    if not errors:
        print("未找到错误分类样本")
        return

    cols = min(5, len(errors))
    rows = (len(errors) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, error in enumerate(errors):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        img = error['image']
        if img.shape[0] == 1:
            ax.imshow(img.squeeze(0), cmap='gray')
        else:
            img_disp = img.permute(1, 2, 0)
            img_disp = torch.clamp(img_disp * torch.tensor([0.229, 0.224, 0.225]) +
                                    torch.tensor([0.485, 0.456, 0.406]), 0, 1)
            ax.imshow(img_disp)
        ax.set_title(f'真实:{class_names[error["true"]]}\n预测:{class_names[error["pred"]]}',
                     fontsize=10, color='red')
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(len(errors), rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis('off')

    plt.suptitle('错误分类样本分析', fontsize=15, fontweight='bold', color='red')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"错误案例图已保存至: {save_path}")
    plt.close()


def plot_hyperparameter_comparison(results_dict, param_name, title, xlabel,
                                    save_path=None):
    """绘制超参数对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x_vals = list(results_dict.keys())
    acc_vals = [results_dict[k]['test_acc'] for k in x_vals]
    f1_vals = [results_dict[k]['macro_f1'] for k in x_vals]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(x_vals)))

    # 准确率对比
    bars1 = ax1.bar(range(len(x_vals)), acc_vals, color=colors, edgecolor='black')
    for bar, val in zip(bars1, acc_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax1.set_xticks(range(len(x_vals)))
    ax1.set_xticklabels(x_vals, fontsize=11)
    ax1.set_ylabel('测试准确率', fontsize=12)
    ax1.set_title(f'{title} - 准确率对比', fontsize=13, fontweight='bold')
    ax1.set_ylim(min(acc_vals) * 0.9, max(acc_vals) * 1.05)

    # F1对比
    bars2 = ax2.bar(range(len(x_vals)), f1_vals, color=colors, edgecolor='black')
    for bar, val in zip(bars2, f1_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax2.set_xticks(range(len(x_vals)))
    ax2.set_xticklabels(x_vals, fontsize=11)
    ax2.set_ylabel('宏平均F1', fontsize=12)
    ax2.set_title(f'{title} - F1对比', fontsize=13, fontweight='bold')
    ax2.set_ylim(min(f1_vals) * 0.9, max(f1_vals) * 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"超参数对比图已保存至: {save_path}")
    plt.close()


def plot_model_comparison(models_results, save_path=None):
    """绘制CNN vs MLP vs LR 对比图"""
    names = list(models_results.keys())
    accs = [models_results[n]['test_acc'] for n in names]
    f1s = [models_results[n]['macro_f1'] for n in names]
    params = [models_results[n].get('params', 0) for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#2E86AB', '#A23B72', '#F18F01']

    # 性能对比
    x = np.arange(len(names))
    width = 0.35
    ax1.bar(x - width / 2, accs, width, label='准确率', color='#2E86AB', edgecolor='white')
    ax1.bar(x + width / 2, f1s, width, label='F1分数', color='#A23B72', edgecolor='white')
    for i in range(len(names)):
        ax1.text(i - width / 2, accs[i] + 0.005, f'{accs[i]:.4f}', ha='center', fontsize=10)
        ax1.text(i + width / 2, f1s[i] + 0.005, f'{f1s[i]:.4f}', ha='center', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=12)
    ax1.set_ylabel('分数', fontsize=12)
    ax1.set_title('模型性能对比', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 1.1)

    # 参数量对比（对数）
    ax2.bar(names, params, color=colors, edgecolor='black')
    for i, p in enumerate(params):
        ax2.text(i, p + max(params) * 0.02, f'{p:,}', ha='center', fontsize=11, fontweight='bold')
    ax2.set_ylabel('参数量（对数尺度）', fontsize=12)
    ax2.set_title('模型参数量对比', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"模型对比图已保存至: {save_path}")
    plt.close()


if __name__ == '__main__':
    print("可视化模块已就绪")
