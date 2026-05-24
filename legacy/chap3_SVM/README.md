

# 支持向量机(SVM)



## 问题描述：

本次作业分为三个部分：

1. 使用基于某种核函数（线性，多项式或高斯核函数）的SVM 解决非线性可分的二分类问题，数
  据集为train_kernel.txt 及test_kernel.txt。

2. 分别使用线性分类器（squared error）、logistic 回归（cross entropy error）以及SVM（hinge error) 解
  决线性二分类问题，并比较三种模型的效果。数据集为train_linear.txt 及test_linear.txt。
  三种误差函数定义如下（Bishop P327）：
![image](http://latex.codecogs.com/gif.latex?E_%7Blinear%7D%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%28y_%7Bn%7D%20-t_%7Bn%7D%29%5E%7B2%7D&plus;%5Clambda%20%5Cleft%20%5C%7C%20%5Cmathbf%7Bw%7D%20%5Cright%20%5C%7C%5E%7B2%7D)  

![image](http://latex.codecogs.com/gif.latex?E_%7Blogistic%7D%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7Dlog%281&plus;exp%28-y_%7Bn%7Dt_%7Bn%7D%29%29%20&plus;%20%5Clambda%5Cleft%20%5C%7C%20%5Cmathbf%7Bw%7D%20%5Cright%20%5C%7C%5E%7B2%7D) 

![image](http://latex.codecogs.com/gif.latex?E_%7BSVM%7D%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5B1-y_%7Bn%7Dt_%7Bn%7D%5D&plus;%5Clambda%20%5Cleft%20%5C%7C%20%5Cmathbf%7Bw%7D%20%5Cright%20%5C%7C%5E%7B2%7D)


  ​
  其中![image](http://latex.codecogs.com/gif.latex?y_%7Bn%7D%3D%5Cmathbf%7Bw%7D%5E%7BT%7Dx_%7Bn%7D&plus;b),![image](http://latex.codecogs.com/gif.latex?t_%7Bn%7D) 为类别标签。

3. 使用多分类SVM 解决三分类问题。数据集为train_multi.txt 及test_multi.txt。（5%）





## 数据集: 

 	MNIST数据集包括60000张训练图片和10000张测试图片。图片样本的数量已经足够训练一个很复杂的模型（例如 CNN的深层神经网络）。它经常被用来作为一个新 的模式识别模型的测试用例。而且它也是一个方便学生和研究者们执行用例的数据集。除此之外，MNIST数据集是一个相对较小的数据集，可以在你的笔记本CPUs上面直接执行





## 题目要求： 

- [ ] 请使用代码模板rbm.py，补全缺失部分，尽量不改动主体部分。
- [ ] 推荐使用python 及numpy 编写主要逻辑代码，适度使用框架。
