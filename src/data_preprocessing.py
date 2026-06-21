"""
数据加载与预处理模块
功能：加载数据集、划分训练/验证/测试集、数据增强、数据可视化
"""
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from PIL import Image

# 配置中文字体
_CN_FONT = None
for _fname in ['SimHei', 'Microsoft YaHei', 'KaiTi']:
    for _f in fm.fontManager.ttflist:
        if _f.name == _fname:
            _CN_FONT = _f
            break
    if _CN_FONT:
        break
if _CN_FONT:
    matplotlib.rcParams['font.family'] = _CN_FONT.name
    matplotlib.rcParams['font.sans-serif'] = [_CN_FONT.name]
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据集路径
DATA_DIR = r"C:\Users\10959\Desktop\artificial intelligence\garbage-cnn-classification\data"
RESULTS_DIR = r"C:\Users\10959\Desktop\artificial intelligence\garbage-cnn-classification\results\figures"

# 类别名称（垃圾分类数据集）
CLASS_NAMES_CN = ['纸板', '玻璃', '金属', '纸张', '塑料', '一般垃圾']
CLASS_NAMES_EN = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def get_transforms(img_size=128, is_grayscale=False):
    """获取数据预处理变换"""
    if is_grayscale:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2860], std=[0.3530])
        ])
        val_test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2860], std=[0.3530])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    return train_transform, val_test_transform


def denormalize(tensor):
    """反归一化以便可视化"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


def load_garbage_dataset(img_size=128):
    """
    加载垃圾分类数据集（从本地文件夹）
    如果本地数据不存在，返回None并提示下载
    """
    data_path = DATA_DIR
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    # 检查是否有包含图片的子文件夹（即按类别组织的数据集）
    valid_exts = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff', '.webp'}
    has_image_subdir = False
    for d in os.listdir(data_path):
        dpath = os.path.join(data_path, d)
        if os.path.isdir(dpath):
            for f in os.listdir(dpath):
                if os.path.splitext(f)[1].lower() in valid_exts:
                    has_image_subdir = True
                    break
        if has_image_subdir:
            break

    if not has_image_subdir:
        print(f"垃圾分类数据集未找到，请将数据集解压到: {data_path}")
        print("下载地址: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification")
        print("数据应按类别文件夹组织，如: data/cardboard/, data/glass/, ...")
        return None

    train_transform, val_test_transform = get_transforms(img_size)

    # 使用ImageFolder加载
    full_dataset = datasets.ImageFolder(root=data_path, transform=None)

    # 划分数据集: 70% 训练, 15% 验证, 15% 测试
    total = len(full_dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # 分别应用不同的transform
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_test_transform
    test_ds.dataset.transform = val_test_transform

    # 为train_ds单独设置transform（因为它是Subset，需要特殊处理）
    from torch.utils.data import Dataset

    class TransformedSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, idx):
            x, y = self.subset[idx]
            if self.transform:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.subset)

    train_dataset = TransformedSubset(train_ds, train_transform)
    val_dataset = TransformedSubset(val_ds, val_test_transform)
    test_dataset = TransformedSubset(test_ds, val_test_transform)

    class_names = full_dataset.classes
    num_classes = len(class_names)

    print(f"数据集加载成功!")
    print(f"  类别数: {num_classes} ({class_names})")
    print(f"  总样本数: {total}")
    print(f"  训练集: {train_size} 张")
    print(f"  验证集: {val_size} 张")
    print(f"  测试集: {test_size} 张")

    return train_dataset, val_dataset, test_dataset, class_names


def load_fashion_mnist(img_size=28):
    """
    加载Fashion-MNIST作为备选数据集
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2860], std=[0.3530])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2860], std=[0.3530])
    ])

    # 下载Fashion-MNIST
    full_train = datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=None
    )
    full_test = datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=None
    )

    # 从训练集分出验证集
    total = len(full_train)
    train_size = int(0.85 * total)
    val_size = total - train_size

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_train, [train_size, val_size],
                                     generator=generator)

    from torch.utils.data import Dataset

    class TransformedSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, idx):
            x, y = self.subset[idx]
            if self.transform:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.subset)

    train_dataset = TransformedSubset(train_ds, train_transform)
    val_dataset = TransformedSubset(val_ds, val_test_transform)
    test_dataset = TransformedSubset(full_test, val_test_transform)

    class_names = ['T恤', '裤子', '套头衫', '连衣裙', '外套',
                   '凉鞋', '衬衫', '运动鞋', '包', '短靴']

    print(f"Fashion-MNIST 数据集加载成功!")
    print(f"  类别数: 10 ({class_names})")
    print(f"  训练集: {train_size} 张")
    print(f"  验证集: {val_size} 张")
    print(f"  测试集: {len(full_test)} 张")

    return train_dataset, val_dataset, test_dataset, class_names


def create_dataloaders(train_ds, val_ds, test_ds, batch_size=32):
    """创建DataLoader"""
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader


def visualize_dataset(dataset, class_names, num_per_class=5, save_path=None):
    """可视化数据集样本"""
    # 收集每类样本
    class_samples = {i: [] for i in range(len(class_names))}
    for img, label in dataset:
        if len(class_samples[label]) < num_per_class:
            class_samples[label].append(img)
        if all(len(v) == num_per_class for v in class_samples.values()):
            break

    fig, axes = plt.subplots(len(class_names), num_per_class,
                              figsize=(num_per_class * 2, len(class_names) * 2))
    for i, (cls_idx, imgs) in enumerate(class_samples.items()):
        for j, img in enumerate(imgs):
            ax = axes[i, j]
            # 反归一化
            if img.shape[0] == 1:  # 灰度图
                img_disp = img.squeeze(0)
            else:
                img_disp = img.permute(1, 2, 0)
            ax.imshow(img_disp, cmap='gray' if img.shape[0] == 1 else None)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(class_names[i], fontsize=12)
            if i == 0:
                ax.set_title(f'样本{j+1}', fontsize=10)

    plt.suptitle('数据集样本展示', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"样本展示图已保存至: {save_path}")
    plt.close()


def visualize_class_distribution(dataset, class_names, save_path=None):
    """可视化类别分布"""
    labels = [dataset[i][1] for i in range(len(dataset))]
    counts = [labels.count(i) for i in range(len(class_names))]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))
    bars = ax.bar(range(len(class_names)), counts, color=colors, edgecolor='black')

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(count), ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_ylabel('样本数量', fontsize=12)
    ax.set_title('数据集类别分布', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"类别分布图已保存至: {save_path}")
    plt.close()


def visualize_augmentation(dataset, transform, class_names, num_samples=3, save_path=None):
    """可视化数据增强效果"""
    fig, axes = plt.subplots(num_samples, 4, figsize=(8, num_samples * 2.5))

    for i in range(num_samples):
        img_tensor, label = dataset[i]

        # Convert tensor to PIL for augmentation transform
        if img_tensor.shape[0] == 1:
            img_pil = transforms.ToPILImage()(img_tensor)
        else:
            img_pil = transforms.ToPILImage()(img_tensor)

        # Display original
        if img_tensor.shape[0] == 1:
            axes[i, 0].imshow(img_tensor.squeeze(0), cmap='gray')
        else:
            axes[i, 0].imshow(img_tensor.permute(1, 2, 0))
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        if i == 0:
            axes[i, 0].set_title('原图', fontsize=10)

        # Display augmented versions
        for j in range(3):
            aug_tensor = transform(img_pil)
            if aug_tensor.shape[0] == 1:
                axes[i, j + 1].imshow(aug_tensor.squeeze(0), cmap='gray')
            else:
                axes[i, j + 1].imshow(aug_tensor.permute(1, 2, 0))
            axes[i, j + 1].set_xticks([])
            axes[i, j + 1].set_yticks([])
            if i == 0:
                axes[i, j + 1].set_title(f'增强{j+1}', fontsize=10)

    plt.suptitle('数据增强效果展示', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"数据增强图已保存至: {save_path}")
    plt.close()


if __name__ == '__main__':
    # 测试数据加载
    print("=" * 60)
    print("数据预处理模块测试")
    print("=" * 60)

    # 尝试加载垃圾分类数据集，失败则使用Fashion-MNIST
    result = load_garbage_dataset(img_size=128)
    if result is None:
        print("\n切换到Fashion-MNIST数据集...")
        train_ds, val_ds, test_ds, class_names = load_fashion_mnist(img_size=32)
    else:
        train_ds, val_ds, test_ds, class_names = result

    # 创建DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=32
    )

    # 可视化
    viz_dir = RESULTS_DIR
    os.makedirs(viz_dir, exist_ok=True)

    visualize_dataset(train_ds, class_names,
                      save_path=os.path.join(viz_dir, 'dataset_samples.png'))
    visualize_class_distribution(train_ds, class_names,
                                  save_path=os.path.join(viz_dir, 'class_distribution.png'))

    print("\n数据预处理模块测试完成!")
