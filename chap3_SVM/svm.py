# python: 3.5.2
# encoding: utf-8

import numpy as np


def load_data(fname):
    """
    载入数据。
    """
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)


def eval_acc(label, pred):
    """
    计算准确率。
    """
    return np.sum(label == pred) / len(pred)


class SVM():
    """
    SVM模型 - 从头实现版本
    使用梯度下降法优化SVM目标函数
    """

    def __init__(self, C=1.0, learning_rate=0.001, n_iterations=1000):
        """
        初始化SVM模型参数
        
        参数:
        C: 正则化参数，控制误分类的惩罚程度
        learning_rate: 学习率
        n_iterations: 迭代次数
        """
        self.C = C
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = None  # 权重向量
        self.b = None  # 偏置项

    def train(self, data_train):
        """
        训练模型。
        使用梯度下降法最小化hinge loss
        
        SVM目标函数: min 1/2 ||w||^2 + C * sum(max(0, 1 - y_i(w^T x_i + b)))
        """
        # 分离特征和标签
        X = data_train[:, :2]  # 特征 [x1, x2]
        y = data_train[:, 2]   # 标签
        
        # 将标签转换为 {-1, 1}（如果原本是 {0, 1}）
        y = np.where(y <= 0, -1, 1)
        
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.w = np.zeros(n_features)
        self.b = 0
        
        # 梯度下降
        for iteration in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # 计算决策函数值
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    # 正确分类且在边界外，只更新w（正则化项）
                    self.w -= self.learning_rate * (2 * self.w / n_samples)
                else:
                    # 误分类或在边界内，更新w和b
                    self.w -= self.learning_rate * (2 * self.w / n_samples - self.C * y[idx] * x_i)
                    self.b -= self.learning_rate * (-self.C * y[idx])
            
            # 可选：打印训练进度
            if (iteration + 1) % 100 == 0:
                loss = self._compute_loss(X, y)
                print(f"Iteration {iteration + 1}/{self.n_iterations}, Loss: {loss:.4f}")

    def _compute_loss(self, X, y):
        """
        计算SVM的损失函数值
        Loss = 1/2 ||w||^2 + C * sum(max(0, 1 - y_i(w^T x_i + b)))
        """
        # 正则化项
        reg_term = 0.5 * np.dot(self.w, self.w)
        
        # Hinge loss
        distances = 1 - y * (np.dot(X, self.w) + self.b)
        hinge_loss = self.C * np.sum(np.maximum(0, distances))
        
        return reg_term + hinge_loss

    def predict(self, x):
        """
        预测标签。
        
        参数:
        x: 特征向量或特征矩阵
        
        返回:
        预测的标签（0或1）
        """
        # 计算决策函数 w^T x + b
        linear_output = np.dot(x, self.w) + self.b
        
        # 转换为 {0, 1} 标签
        predictions = np.where(linear_output >= 0, 1, 0)
        
        return predictions


class SVM_sklearn():
    """
    SVM模型 - 使用sklearn的简化版本
    仅供参考对比
    """
    
    def __init__(self, C=1.0):
        """
        初始化SVM模型
        """
        try:
            from sklearn.svm import LinearSVC
            self.model = LinearSVC(C=C, max_iter=1000, random_state=42)
        except ImportError:
            print("警告: sklearn未安装，请使用从头实现的SVM类")
            self.model = None
    
    def train(self, data_train):
        """
        训练模型
        """
        if self.model is None:
            raise ImportError("sklearn未安装")
        
        X = data_train[:, :2]
        y = data_train[:, 2]
        self.model.fit(X, y)
    
    def predict(self, x):
        """
        预测标签
        """
        if self.model is None:
            raise ImportError("sklearn未安装")
        
        return self.model.predict(x)


if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    
    try:
        data_train = load_data(train_file)  # 数据格式[x1, x2, t]
        data_test = load_data(test_file)
    except FileNotFoundError:
        print("错误: 找不到数据文件。请确保 data/train_linear.txt 和 data/test_linear.txt 存在。")
        print("使用模拟数据进行演示...")
        
        # 创建模拟数据
        np.random.seed(42)
        n_samples = 100
        
        # 类别1
        X1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
        y1 = np.ones(n_samples // 2)
        
        # 类别0
        X2 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
        y2 = np.zeros(n_samples // 2)
        
        # 合并数据
        data_train = np.hstack([np.vstack([X1, X2]), np.vstack([y1.reshape(-1, 1), y2.reshape(-1, 1)])])
        data_test = data_train.copy()
        np.random.shuffle(data_train)

    print("=" * 50)
    print("使用从头实现的SVM模型")
    print("=" * 50)
    
    # 使用训练集训练SVM模型
    svm = SVM(C=1.0, learning_rate=0.001, n_iterations=1000)  # 初始化模型
    svm.train(data_train)  # 训练模型

    # 使用SVM模型预测标签
    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    t_train_pred = svm.predict(x_train)  # 预测标签
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("\n最终结果:")
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
    
    print("\n模型参数:")
    print(f"w = {svm.w}")
    print(f"b = {svm.b:.4f}")