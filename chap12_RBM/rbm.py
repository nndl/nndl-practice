# python: 3.x
# encoding: utf-8

"""
受限玻尔兹曼机 (Restricted Boltzmann Machine, RBM)
使用对比散度(Contrastive Divergence)算法训练
使用Gibbs采样生成样本
"""

import numpy as np
import matplotlib.pyplot as plt


class RBM:
    """Restricted Boltzmann Machine."""

    def __init__(self, n_hidden=2, n_observe=784, learning_rate=0.01):
        """
        Initialize model.
        
        Args:
            n_hidden: 隐藏层单元数量
            n_observe: 可见层单元数量(观测变量维度)
            learning_rate: 学习率
        """
        self.n_hidden = n_hidden      # 隐藏层单元数
        self.n_observe = n_observe    # 可见层单元数
        self.learning_rate = learning_rate
        
        # 初始化权重矩阵 W: shape(n_observe, n_hidden)
        # 使用小的随机值初始化
        self.W = np.random.randn(n_observe, n_hidden) * 0.01
        
        # 初始化可见层偏置 b: shape(n_observe,)
        self.b = np.zeros(n_observe)
        
        # 初始化隐藏层偏置 c: shape(n_hidden,)
        self.c = np.zeros(n_hidden)
        
        print(f"RBM 初始化完成:")
        print(f"  可见层单元数: {n_observe}")
        print(f"  隐藏层单元数: {n_hidden}")
        print(f"  参数总数: {self.W.size + self.b.size + self.c.size}")

    def sigmoid(self, x):
        """Sigmoid激活函数"""
        # 数值稳定性处理
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def sample_h_given_v(self, v):
        """
        给定可见层v，采样隐藏层h
        P(h_j=1|v) = sigmoid(c_j + sum_i(v_i * W_ij))
        
        Args:
            v: 可见层状态, shape(batch_size, n_observe) 或 (n_observe,)
        
        Returns:
            h_prob: 隐藏层激活概率, shape(batch_size, n_hidden) 或 (n_hidden,)
            h_sample: 隐藏层采样值 {0,1}, 同上
        """
        # 计算激活概率
        if v.ndim == 1:
            # 单个样本
            activation = self.c + v @ self.W  # shape(n_hidden,)
        else:
            # 批量样本
            activation = v @ self.W + self.c  # shape(batch_size, n_hidden)
        
        h_prob = self.sigmoid(activation)
        
        # 伯努利采样
        h_sample = (np.random.random(h_prob.shape) < h_prob).astype(float)
        
        return h_prob, h_sample

    def sample_v_given_h(self, h):
        """
        给定隐藏层h，采样可见层v
        P(v_i=1|h) = sigmoid(b_i + sum_j(h_j * W_ij))
        
        Args:
            h: 隐藏层状态, shape(batch_size, n_hidden) 或 (n_hidden,)
        
        Returns:
            v_prob: 可见层激活概率, shape(batch_size, n_observe) 或 (n_observe,)
            v_sample: 可见层采样值 {0,1}, 同上
        """
        # 计算激活概率
        if h.ndim == 1:
            # 单个样本
            activation = self.b + h @ self.W.T  # shape(n_observe,)
        else:
            # 批量样本
            activation = h @ self.W.T + self.b  # shape(batch_size, n_observe)
        
        v_prob = self.sigmoid(activation)
        
        # 伯努利采样
        v_sample = (np.random.random(v_prob.shape) < v_prob).astype(float)
        
        return v_prob, v_sample

    def gibbs_sampling(self, v0, k=1):
        """
        Gibbs采样k步
        
        Args:
            v0: 初始可见层状态
            k: Gibbs采样步数
        
        Returns:
            vk: k步后的可见层状态
            h0_prob: 初始隐藏层概率
            hk_prob: k步后的隐藏层概率
        """
        v = v0
        
        # 第一步: v0 -> h0
        h0_prob, h0_sample = self.sample_h_given_v(v)
        h = h0_sample
        
        # Gibbs采样k步
        for _ in range(k):
            # h -> v
            v_prob, v_sample = self.sample_v_given_h(h)
            v = v_sample
            
            # v -> h
            h_prob, h_sample = self.sample_h_given_v(v)
            h = h_sample
        
        # 返回最后一步的可见层和隐藏层
        vk = v
        hk_prob = h_prob
        
        return vk, h0_prob, hk_prob

    def contrastive_divergence(self, v0, k=1):
        """
        对比散度(CD-k)算法
        
        Args:
            v0: 训练数据, shape(batch_size, n_observe)
            k: CD-k中的k值
        
        Returns:
            delta_W: 权重更新量
            delta_b: 可见层偏置更新量
            delta_c: 隐藏层偏置更新量
        """
        batch_size = v0.shape[0]
        
        # Gibbs采样k步
        vk, h0_prob, hk_prob = self.gibbs_sampling(v0, k=k)
        
        # 计算梯度（期望的差值）
        # <v*h^T>_data - <v*h^T>_model
        positive_grad = v0.T @ h0_prob  # shape(n_observe, n_hidden)
        negative_grad = vk.T @ hk_prob
        
        delta_W = (positive_grad - negative_grad) / batch_size
        delta_b = (v0 - vk).mean(axis=0)
        delta_c = (h0_prob - hk_prob).mean(axis=0)
        
        return delta_W, delta_b, delta_c

    def train(self, data, n_epochs=10, batch_size=32, k=1, verbose=True):
        """
        Train model using data.
        
        Args:
            data: 训练数据, shape(n_samples, n_rows, n_cols) 或 (n_samples, n_observe)
            n_epochs: 训练轮数
            batch_size: 批量大小
            k: CD-k中的k值
            verbose: 是否打印训练信息
        """
        # 数据预处理
        if data.ndim == 3:
            # 如果是图像格式(n_samples, rows, cols)，展平
            n_samples = data.shape[0]
            data = data.reshape(n_samples, -1)
        
        n_samples = data.shape[0]
        n_batches = n_samples // batch_size
        
        print(f"\n开始训练 RBM...")
        print(f"  训练样本数: {n_samples}")
        print(f"  批量大小: {batch_size}")
        print(f"  训练轮数: {n_epochs}")
        print(f"  CD-k: k={k}")
        print(f"  学习率: {self.learning_rate}")
        print("-" * 60)
        
        # 训练历史
        self.train_errors = []
        
        for epoch in range(n_epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)
            data_shuffled = data[indices]
            
            epoch_error = 0.0
            
            for batch_idx in range(n_batches):
                # 获取batch数据
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_data = data_shuffled[start_idx:end_idx]
                
                # CD-k算法
                delta_W, delta_b, delta_c = self.contrastive_divergence(batch_data, k=k)
                
                # 更新参数
                self.W += self.learning_rate * delta_W
                self.b += self.learning_rate * delta_b
                self.c += self.learning_rate * delta_c
                
                # 计算重构误差
                _, v_recon = self.reconstruct(batch_data)
                batch_error = np.mean((batch_data - v_recon) ** 2)
                epoch_error += batch_error
            
            # 平均误差
            epoch_error /= n_batches
            self.train_errors.append(epoch_error)
            
            if verbose:
                print(f"Epoch {epoch+1:3d}/{n_epochs}: 重构误差 = {epoch_error:.6f}")
        
        print("-" * 60)
        print("训练完成！")

    def reconstruct(self, v):
        """
        重构输入数据
        v -> h -> v'
        
        Args:
            v: 输入数据
        
        Returns:
            h_prob: 隐藏层概率
            v_recon: 重构的可见层
        """
        h_prob, _ = self.sample_h_given_v(v)
        v_recon, _ = self.sample_v_given_h(h_prob)
        return h_prob, v_recon

    def sample(self, n_samples=10, n_gibbs_steps=1000, init='random'):
        """
        Sample from trained model.
        使用Gibbs采样生成样本
        
        Args:
            n_samples: 生成样本数量
            n_gibbs_steps: Gibbs采样步数
            init: 初始化方式 'random' 或 'data'
        
        Returns:
            samples: 生成的样本, shape(n_samples, n_observe)
        """
        print(f"\n使用 Gibbs 采样生成 {n_samples} 个样本...")
        print(f"  Gibbs采样步数: {n_gibbs_steps}")
        
        samples = []
        
        for i in range(n_samples):
            # 初始化可见层
            if init == 'random':
                # 随机初始化（伯努利0.5）
                v = (np.random.random(self.n_observe) < 0.5).astype(float)
            else:
                # 从数据中随机选择一个
                v = np.random.choice([0, 1], size=self.n_observe)
            
            # Gibbs采样
            for step in range(n_gibbs_steps):
                # v -> h
                h_prob, h_sample = self.sample_h_given_v(v)
                
                # h -> v
                v_prob, v_sample = self.sample_v_given_h(h_sample)
                
                v = v_sample
            
            samples.append(v)
            
            if (i + 1) % max(1, n_samples // 10) == 0:
                print(f"  已生成 {i+1}/{n_samples} 个样本")
        
        samples = np.array(samples)
        print("采样完成！")
        
        return samples


# ================================
# 可视化函数
# ================================

def visualize_samples(samples, n_rows=2, n_cols=5, img_shape=(28, 28), 
                     title="Generated Samples", save_path=None):
    """可视化生成的样本"""
    n_samples = min(n_rows * n_cols, len(samples))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    axes = axes.flatten()
    
    for i in range(n_samples):
        img = samples[i].reshape(img_shape)
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}')
    
    # 隐藏多余的子图
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存: {save_path}")
    
    plt.show()


def visualize_weights(rbm, n_display=100, img_shape=(28, 28), save_path=None):
    """可视化权重矩阵（每个隐藏单元学到的特征）"""
    n_hidden = min(rbm.n_hidden, n_display)
    n_cols = 10
    n_rows = (n_hidden + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*1.5, n_rows*1.5))
    axes = axes.flatten()
    
    for i in range(n_hidden):
        weight = rbm.W[:, i].reshape(img_shape)
        axes[i].imshow(weight, cmap='RdBu', vmin=-weight.std()*2, vmax=weight.std()*2)
        axes[i].axis('off')
        axes[i].set_title(f'H{i}', fontsize=8)
    
    # 隐藏多余的子图
    for i in range(n_hidden, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Learned Features (Weights)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存: {save_path}")
    
    plt.show()


def visualize_reconstruction(rbm, data, n_samples=5, img_shape=(28, 28), save_path=None):
    """可视化重构结果"""
    # 随机选择样本
    indices = np.random.choice(len(data), n_samples, replace=False)
    samples = data[indices]
    
    if samples.ndim == 3:
        samples = samples.reshape(len(samples), -1)
    
    # 重构
    _, recon = rbm.reconstruct(samples)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*2, 4))
    
    for i in range(n_samples):
        # 原始图像
        axes[0, i].imshow(samples[i].reshape(img_shape), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)
        
        # 重构图像
        axes[1, i].imshow(recon[i].reshape(img_shape), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstructed', fontsize=12)
    
    plt.suptitle('Original vs Reconstructed', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存: {save_path}")
    
    plt.show()


def plot_training_curve(rbm, save_path=None):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(rbm.train_errors, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Reconstruction Error', fontsize=12)
    plt.title('RBM Training Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存: {save_path}")
    
    plt.show()


# ================================
# 主程序
# ================================

if __name__ == '__main__':
    import os
    
    # 创建输出目录
    OUTPUT_DIR = "rbm_output"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}/\n")
    
    # 检查数据文件
    if not os.path.exists('mnist_bin.npy'):
        print("警告: mnist_bin.npy 不存在！")
        print("生成模拟数据用于演示...")
        # 生成随机二值化数据作为演示
        mnist = (np.random.random((1000, 28, 28)) < 0.5).astype(float)
        np.save('mnist_bin.npy', mnist)
        print("已生成模拟数据: mnist_bin.npy\n")
    
    # load mnist dataset, no label
    print("=" * 70)
    print(" " * 20 + "受限玻尔兹曼机 (RBM)")
    print("=" * 70)
    
    mnist = np.load('mnist_bin.npy')  # shape: (n_samples, 28, 28)
    n_imgs, n_rows, n_cols = mnist.shape
    img_size = n_rows * n_cols
    
    print(f"\n数据集信息:")
    print(f"  样本数量: {n_imgs}")
    print(f"  图像尺寸: {n_rows} x {n_cols}")
    print(f"  特征维度: {img_size}")
    print(f"  数据范围: [{mnist.min()}, {mnist.max()}]")
    
    # construct rbm model
    print("\n" + "=" * 70)
    print("步骤1: 构建RBM模型")
    print("=" * 70)
    
    n_hidden = 128  # 隐藏单元数（可调整）
    rbm = RBM(n_hidden=n_hidden, n_observe=img_size, learning_rate=0.01)
    
    # train rbm model using mnist
    print("\n" + "=" * 70)
    print("步骤2: 训练RBM")
    print("=" * 70)
    
    rbm.train(
        mnist, 
        n_epochs=20,      # 训练轮数
        batch_size=64,    # 批量大小
        k=1,              # CD-k中的k值
        verbose=True
    )
    
    # 可视化训练曲线
    print("\n" + "=" * 70)
    print("步骤3: 可视化训练过程")
    print("=" * 70)
    
    plot_training_curve(rbm, save_path=os.path.join(OUTPUT_DIR, 'training_curve.png'))
    
    # 可视化学到的特征
    print("\n" + "=" * 70)
    print("步骤4: 可视化学到的特征")
    print("=" * 70)
    
    visualize_weights(rbm, n_display=min(100, n_hidden), img_shape=(n_rows, n_cols),
                     save_path=os.path.join(OUTPUT_DIR, 'learned_features.png'))
    
    # 可视化重构
    print("\n" + "=" * 70)
    print("步骤5: 可视化重构结果")
    print("=" * 70)
    
    visualize_reconstruction(rbm, mnist, n_samples=10, img_shape=(n_rows, n_cols),
                           save_path=os.path.join(OUTPUT_DIR, 'reconstruction.png'))
    
    # sample from rbm model
    print("\n" + "=" * 70)
    print("步骤6: 使用Gibbs采样生成新样本")
    print("=" * 70)
    
    n_samples = 20
    samples = rbm.sample(n_samples=n_samples, n_gibbs_steps=1000)
    
    # 可视化生成的样本
    print("\n" + "=" * 70)
    print("步骤7: 可视化生成的样本")
    print("=" * 70)
    
    visualize_samples(samples, n_rows=4, n_cols=5, img_shape=(n_rows, n_cols),
                     title="Generated Samples from RBM",
                     save_path=os.path.join(OUTPUT_DIR, 'generated_samples.png'))
    
    # 保存模型
    print("\n" + "=" * 70)
    print("步骤8: 保存模型参数")
    print("=" * 70)
    
    model_path = os.path.join(OUTPUT_DIR, 'rbm_model.npz')
    np.savez(model_path,
             W=rbm.W,
             b=rbm.b,
             c=rbm.c,
             n_hidden=rbm.n_hidden,
             n_observe=rbm.n_observe)
    print(f"模型已保存: {model_path}")
    
    print("\n" + "=" * 70)
    print("全部完成！")
    print(f"结果已保存到: {OUTPUT_DIR}/")
    print("=" * 70)