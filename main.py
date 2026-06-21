"""CNN服装图像分类 —— 主程序入口"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.data_preprocessing import load_fashion_mnist, create_dataloaders
from src.model import GarbageCNN, count_parameters
from src.train import train_model
from src.evaluate import evaluate_on_test, compute_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print("=" * 55)
    print("  基于CNN的服装图像分类 —— Fashion-MNIST")
    print("=" * 55)
    print(f"  运行设备: {DEVICE}")

    # 1. 加载数据
    print("\n[1/4] 加载数据集...")
    train_ds, val_ds, test_ds, class_names = load_fashion_mnist(img_size=28)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=32)
    print(f"  类别数: {len(class_names)}")
    print(f"  训练集: {len(train_ds)}  验证集: {len(val_ds)}  测试集: {len(test_ds)}")

    # 2. 构建模型
    print("\n[2/4] 构建CNN模型...")
    model = GarbageCNN(num_classes=10, input_channels=1).to(DEVICE)
    total_params, trainable_params = count_parameters(model)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 3. 训练
    print("\n[3/4] 开始训练...")
    config = {
        'optimizer': 'adam', 'lr': 0.001, 'weight_decay': 1e-4,
        'epochs': 15, 'early_stop_patience': 5
    }
    history, best_val_acc, best_epoch = train_model(
        model, train_loader, val_loader, config, DEVICE)

    # 4. 测试
    print("\n[4/4] 测试集评估...")
    probs, preds, targets = evaluate_on_test(model, test_loader, DEVICE)
    metrics = compute_metrics(preds, targets, class_names)

    print("\n" + "=" * 55)
    print("  最终结果")
    print("=" * 55)
    print(f"  测试准确率 (Accuracy):  {metrics['accuracy']:.4f}")
    print(f"  宏平均F1 (Macro F1):    {metrics['macro_f1']:.4f}")
    print(f"  最佳验证准确率:          {best_val_acc:.4f} (第{best_epoch}轮)")
    print(f"  模型参数量:              {total_params:,}")
    print("\n  各类别F1分数:")
    for m in metrics['per_class']:
        print(f"    {m['class']:<10s}  {m['f1']:.4f}")
    print("\n" + "=" * 55)
    print("  运行完成!")
    print("=" * 55)

if __name__ == '__main__':
    main()
