"""
模型训练模块
功能：训练循环、早停机制、学习率调度、训练日志
"""
import torch
import torch.nn as nn
import os
import json
from datetime import datetime


class EarlyStopping:
    """早停机制 —— 监控验证损失，耐心耗尽时停止训练"""

    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improvement = self.best_score - score
        else:
            improvement = score - self.best_score

        if improvement > self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return True


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        all_preds.extend(predicted.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_targets


def train_model(model, train_loader, val_loader, config, device,
                save_dir=None):
    """
    完整训练流程
    返回训练历史记录
    """
    criterion = nn.CrossEntropyLoss()

    # 选择优化器
    optimizer_name = config.get('optimizer', 'adam')
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 1e-4)

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                      weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                     momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr,
                                         weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    early_stopping = EarlyStopping(
        patience=config.get('early_stop_patience', 15),
        mode='min'
    )

    num_epochs = config.get('epochs', 50)
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None

    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>8} {'LR':>10} {'备注':>10}")
    print("-" * 75)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # 保存最佳模型
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # 早停检查
        no_improve = early_stopping(val_loss)

        note = ''
        if is_best:
            note = '★ 最佳'
        if no_improve:
            note += ' (未改善)'

        print(f"{epoch+1:>6d} {train_loss:>12.4f} {train_acc:>10.4f} "
              f"{val_loss:>10.4f} {val_acc:>8.4f} {current_lr:>10.6f} {note:>10}")

        if early_stopping.early_stop:
            print(f"\n早停触发! 在第 {epoch+1} 轮停止训练")
            print(f"最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
            break

    if not early_stopping.early_stop:
        print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 保存训练历史
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump({
                'config': config,
                'history': history,
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch
            }, f, indent=2, ensure_ascii=False)
        print(f"训练历史已保存至: {history_path}")

    return history, best_val_acc, best_epoch


if __name__ == '__main__':
    print("训练模块已就绪")
