# 线性回归



## 问题描述：

有一个函数![image](http://latex.codecogs.com/gif.latex?f%3A%20%5Cmathbb%7BR%7D%5Crightarrow%20%5Cmathbb%7BR%7D) ，使得。现 ![image](http://latex.codecogs.com/gif.latex?y%20%3D%20f%28x%29)在不知道函数 $f(\cdot)$的具体形式，给定满足函数关系的一组训练样本![image](http://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20%5Cleft%20%28%20x_%7B1%7D%2Cy_%7B1%7D%20%5Cright%20%29%2C...%2C%5Cleft%20%28%20x_%7BN%7D%2Cy_%7BN%7D%20%5Cright%20%29%20%5Cright%20%5C%7D%2CN%3D300)，请使用线性回归模型拟合出函数$y=f(x)$。

(可尝试一种或几种不同的基函数，如多项式、高斯或sigmoid基函数）




## 数据集: 

 	根据某种函数关系生成的train 和test 数据。



## 题目要求： 

- [ ] 按顺序完成 `exercise-linear_regression.ipynb`中的填空 
    1. 先完成最小二乘法的优化 (参考书中第二章 2.3节中的公式)
    1. 附加题：实现“多项式基函数”以及“高斯基函数”（可参考PRML）
    1. 附加题：完成梯度下降的优化 (参考书中第二章 2.3节中的公式)

- [ ] 使用训练集train.txt 进行训练，使用测试集test.txt 进行评估（标准差），训练模型时请不要使用测试集。

