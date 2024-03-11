<font face="song">

## Chapter 2 : Perceptron

<center>图灵2201 张祎迪</center>

**2.1 Minsky 和 Papert 指出:感知机因为是线性模型，所以不能表示复杂的函数，如异或(XOR)。  验证感知机为什么不能表示异或。**

对异或操作，训练数据集 $T =\{((0,0),0),((1,1),0),((0,1),1),((1,0),1)\} $ 

用反证法，如果感知机可以表示异或，那么 $\exist w = (w1, w2)^T∈ \bold{R}^2, b ∈ \bold{R} \ st.$​ 
$$
\left\{  
             \begin{array}{**lr**}  
              w·(0,0)^T +b<0 \\
              w·(1,1)^T +b<0 \\
              w·(0,1)^T +b>0 \\
              w·(1,0)^T +b<0   
             \end{array}  
\right.
$$
该方程组无解，因此对于任意的 $w ∈\bold{R}^2,b ∈ \bold{R}$ ，感知机模型 $f(x) = sign(w · x + b)$​ 都存在误分类点，即感知机不能表示异或。

**2.3 证明以下定理:样本集线性可分的充分必要条件是正实例点集所构成的凸壳与负实例点集所构成 的凸壳互不相交。**
$$
设集合S⊂ \bold{R}^n 是由 \bold{R}^n 中 k 个点所组成的集合,定义 S 的凸壳为:\\
conv(S)=\{x = \sum_{i=1}^k \lambda_ix_i | \sum_{i=1}^k \lambda_i=1,\lambda_i\ge 0,i=1,2,...k\}
$$
**充分性：**

$$dist (x_i,x_j) = ||x_i -x_j||_2\\$$

$$定义 conv(S_+) 和 conv(S_−) 之间的距离为:$$

$$dist(conv(S_+),conv(S_−))=min||s_+,s_−||(s_+ ∈conv(S+),s_− ∈conv(S−)) (1)$$

$$设 \ x_+\in conv(S_+),x_-\in conv(S_-)满足\ dist(x_+,x_-)= dist(conv(S_+),conv(S_−))$$​

$\because  conv(S_+) ∩ conv(S_−) = ∅$​

$\therefore dist(conv(S_+),conv(S_−)) >0$​

$\therefore \forall x ∈ conv(S_+)， dist(x, x_−) > dist(x, x_+)\\\forall x ∈ conv(S_-)， dist(x, x_+) > dist(x, x_-)$​

$\therefore \exist\ w=x_+ −x_−\ ,\ b=−\frac{x_+·x_+-x_-·x_-}{2} \ st$​

$\begin{align*}\forall  x ∈ conv(S_+),w^Tx+b&= (x_+ −x_−)·x −\frac{x_+·x_+-x_-·x_-}{2}\\&=\frac{||x-x_-||^2-||x-x_+||^2}{2} > 0\end{align*}$

$\begin{align*}\forall  x ∈ conv(S_-),w^Tx+b&= (x_+ −x_−)·x −\frac{x_+·x_+-x_-·x_-}{2}\\&=\frac{||x-x_-||^2-||x-x_+||^2}{2} < 0\end{align*}$

$\therefore S_+ , S_− 线性可分$



**必要性：**

若$S_+ ,S_−$ 线性可分，则$\exist\  w^Tx+b=0$将$S_+ , S_−$  分开，则$\forall \ x_i ∈S_+ , w^Tx_i +b=ε_i >0,i=1,2,...,p$

$\therefore \forall s_+∈ conv(S_+) :$

$\begin{align*}w^Ts_+ &= w^T\sum_{i=1}^p\lambda_ix_i\\ &=\sum_{i=1}^p\lambda_i (\epsilon_i -b)\\ &= \sum_{i=p}^p \lambda_i\epsilon_i - b\sum_{i=p}^p \lambda_i\\ &= \sum_{i=p}^p \lambda_i\epsilon_i - b\end{align*}$​

$\therefore w^Ts_++b>0\ \  类似地\ \  w^Ts_-+b>0 \ for\ \forall s_- \in conv(S_-)$

若 $conv(S_+) ∩ conv(S_−)\ne ∅ ,则\ \exist s\in conv(S_+) ∩ conv(S_−) \ st \ w^Ts+b>0 \ and\ w^Ts+b<0$

推出矛盾，所以可得 $conv(S_+) ∩ conv(S_−)= ∅$

</font>
