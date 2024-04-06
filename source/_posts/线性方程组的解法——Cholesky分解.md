---
title: 线性方程组的解法——Cholesky分解
date: 2024-04-06 18:52:11
categories:
- [2.数值计算,1.线性方程组的解法]
tags:
- Cholesky分解
- 平方根法
mathjax: true
---

## Cholesky分解

在科学和工程计算中，经常需要求解形如 $\mathbf{A} \mathit{x} = \mathit{b}$ 的线性方程组，
其中 $\mathbf{A}$ 为 $n \times m$ 矩阵，称为系数矩阵， $\mathit{b}$ 为 $n$ 维列向量，称为右端向量，
$\mathit{x}$ 为待求解的 $m$ 维列向量，称为解向量。

而科学和工程的实际计算中，经常遇到系数矩阵 $\mathbf{A}$ 为对称正定矩阵的情况。若
$$
\mathbf{A}=\begin{bmatrix}
    a_{11} \\ a_{21} & a_{22}  &  & 对称\\
    a_{31} & a_{32} & a_{33} \\
    \vdots & \vdots & \vdots & \ddots \\
    a_{n1} & a_{n2} & \dots & a_{nn}
\end{bmatrix}
$$
为正定阵，则有如下三角阵
$$
\mathbf{L} = \begin{bmatrix}
    l_{11} \\
    l_{21} & l_{22} & & \mathbf{0}\\
    l_{31} & l_{32} & l_{33} \\
    \vdots & \vdots & \vdots & \ddots \\
    l_{n1} & l_{n2} & l_{n3} & \dots & l_{nn}
\end{bmatrix} \\
$$
使 $\mathbf{A} = \mathbf{L \cdot L^T}$ 成立。若 $\mathbf{L}$ 的主对角线元素取正值，则这种分解是唯一的。

将矩阵关系式 $\mathbf{A} = \mathbf{L \cdot L^T}$ 直接展开，有
$$
\begin{align*}
    a_{11} &= l_{11}^{2} \\ 
    a_{21} &= l_{21}l_{11},\quad a_{22} = l_{21}^{2}+l_{22}^{2}\\ 
    a_{31} &= l_{31}l_{11},\quad a_{32} = l_{31}l_{21}+l_{32}l_{22},\quad a_{33}=l_{31}^{2}+l_{32}^{2}+l_{33}^{2}\\
    \dots
\end{align*}\\
$$

据此可逐行求出矩阵 $\mathbf{L}$ 的元素 $l_{11} \rightarrow l_{21} \rightarrow l_{22} \rightarrow l_{31} \rightarrow l_{32} \rightarrow \dots$，计算公式为
$$
\begin{cases}
    l_{ij} &= (a_{ij} - \sum\limits_{k=1}^{j-1}l_{ik}l_{jk}) / l_{jj}, \quad & j = 1, 2, \dots, i-1 \\
    l_{ii} &= (a_{ii} - \sum\limits_{k=1}^{i-1}l_{ik}^2)^\frac{1}{2}, \quad & i = 1, 2, \dots, n \\ \end{cases} \\
$$

基于矩阵分解式 $\mathbf{A} = \mathbf{L \cdot L^T}$，对称正定方程组 $\mathbf{A} \mathit{x} = \mathit{b}$ 可归结为两个三角方程组
$\mathbf{L} \mathit{y} = \mathit{b}$ 和 $\mathbf{L}^T \mathit{x} = \mathit{y}$ 来求解。

由 $\mathbf{L} \mathit{y} = \mathit{b}$ 即
$$
\begin{cases} l_{11}y_{1} &= b_1 \\ l_{21}y_{2} + l_{22}y_{2} &= b_2 \\ \dots \dots \dots \\ l_{n1}y_{1} + l_{n2}y_{2} + \dots + l_{nn}y_{n} &= b_n \end{cases} \\
$$
可顺序计算出 $y_1 \rightarrow y_2 \rightarrow \dots \rightarrow y_n$ ：
$$
y_i = (b_i - \sum\limits_{k=1}^{i-1}l_{ik}y_{k})/l_{ii}, \quad i = 1, 2, \dots,n \\
$$

而由 $\mathbf{L}^T \mathit{x} = \mathit{y}$ 即
$$
\begin{cases}
    \begin{alignat*}{2}
    l_{11}x_1 + l_{21}x_2 + \dots + l_{n1}x_n &= y_1 \\
    l_{22}x_2 + \dots + l_{n2}x_n &= y_2 \\
    \dots \dots \dots \\
    l_{nn}x_n &= y_n \\
    \end{alignat*}
\end{cases} \\
$$
可逆序求得 $x_n \rightarrow x_{n-1} \rightarrow \dots \rightarrow x_1$：
$$
x_i = (y_i - \sum\limits_{k=i+1}^nl_{ki}x_{k})/l_{ii}, \quad i = n, n-1, \dots, 1 \\
$$

由于矩阵分解时公式含有开方运算，所以该算法称为平方根法，又叫Cholesky分解法。

## 代码实现（Fortran版）

根据上述公式，编写程序即可对方程进行求解：

``` Fortran
subroutine cholesky_full(n, a, y)
    implicit none
    
    integer, intent(in) :: n
    real, intent(inout) :: a(n, n), y(n)
    
    integer :: i, j, k
    real :: temp
    
    ! 分解矩阵，生成下三角阵L
    ! 工程问题中的很多矩阵非常庞大，所以，计算过程中的数据应该直接存放在原始数组a中，
    ! 而不是新创建一个数组
    do i = 1, n
        ! 公式中，j的取值范围为1到j-1，此处换成1到j，可以将分解式统一起来，省去一次判断。
        ! 因为j=i时，j循环虽然会执行错误的操作、生成错误的a(i,j)结果，
        ! 但a(i,j)马上就会被最外层的i循环生成的正确数据替换
        do j = 1, i
            temp = a(i, j)
            do k = 1, j-1
                temp = temp - a(i, k) * a(j, k)
            end do
            a(i, j) = temp / a(j, j)
            ! a(j, i) = 0.  ! 对角线上方d的元素赋0，可有可无
        end do
        a(i, i) = sqrt(temp)
    end do
    
    ! 根据Ly=b求解出y
    do i = 1, n
        temp = y(i)
        do j = 1, i-1
            temp = temp - a(i, j) * y(j)
        end do
        y(i) = temp / a(i, i)
    end do
    
    ! 求解出x
    do i = n, 1, -1
        temp = y(i) / a(i, i)
        y(i) = temp
        ! 公式中k的范围为i+1到n，此处为1到i-1，因为下方a(i,k)的下标和公式中交换了顺序
        do k = 1, i-1
            y(k) = y(k) - temp * a(i, k)
        end do
    end do
end subroutine
```

以上代码的Cholesky分解部分与前文公式基本上一致，很好理解，但引入了一个临时变量temp，用于存储数据。
而如果我们将j、k两层循环交换一下位置，再稍微调整一下循环计数器的取值范围，就可以不借助临时变量直接完成分解操作。
代码如下：
``` Fortran
do i = 1, n
    do k = 1, i - 1
        a(i, k) = a(i, k) / a(k, k)
        do j = k + 1, i
            a(i, j) = a(i, j) - a(i, k) * a(j, k)
        end do
    end do
    a(i, i) = sqrt(a(i, i))
end do
```

## 参考文献
[1].王能超. 高等学校教材, 数值分析简明教程, （第2版）[M]. 2003.

[2].吴建平, 王正华, 李晓梅. 稀疏线性方程组的高效求解与并行计算[M]. 湖南科学技术出版社, 2004.