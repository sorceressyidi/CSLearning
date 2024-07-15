import math
class coin:
   def __init__(self,threshold=1e-6,max_iter=1000,theta=[0.5,0.5,0.5]):
   ############################
   #     初始化模型参数         #
   #     threshold:收敛阈值    #
   #     max_iter:最大迭代次数  #
   #     theta:模型参数的初值   #
   ############################
       self.threshold=threshold
       self.max_iter=max_iter
       self.pi,self.p,self.q=theta
   def mu(self,j):
   #############################################
   #     （E步）计算mu                          #
   #     j:观测数据y的第j个                      #
   #     返回在模型参数下观测数据yj来自掷硬币B的概率 #
   #############################################
       pro_1=self.pi*math.pow(self.p,data[j])*math.pow((1-self.p),1-data[j])
       pro_2=(1-self.pi)*math.pow(self.q,data[j])*math.pow((1-self.q),1-data[j])
       return pro_1/(pro_1+pro_2)
   def fit(self,data):
   #############################################
   #     模型迭代                               #
   #     data:观测数据                          #
   #############################################
       count=len(data)
       print("模型参数的初值：")
       print("pi={},p={},q={}".format(self.pi,self.p,self.q))
       print("EM算法训练过程:")
       for i in range(self.max_iter):
           #（E步）得到在模型参数下观测数据yj来自掷硬币B的概率
           _mu=[self.mu(j) for j in range(count)]
           #（M步）计算模型参数的新估计值
           pi=1/count*sum(_mu)
           p=sum([_mu[k]*data[k] for k in range(count)])/sum([_mu[k] for k in range(count)])
           q=sum([(1-_mu[k])*data[k] for k in range(count)])/sum([(1-_mu[k]) for k in range(count)])
           print('第{}次:pi={:.4f},p={:.4f},q={:.4f}'.format(i+1,pi,p,q))
           #计算误差值
           error=abs(self.pi-pi)+abs(self.p-p)+abs(self.q-q)
           self.pi=pi
           self.p=p
           self.q=q
           #判断是否收敛
           if error<self.threshold:
               print("模型参数的极大似然估计：")
               print("pi={:.4f},p={:.4f},q={:.4f}".format(self.pi,self.p,self.q))
               break

# 加载数据
data = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
# 模型参数的初值
init_prob = [0.46, 0.55, 0.67]

# 三硬币模型的EM模型
em = coin(theta=init_prob, threshold=1e-5, max_iter=100)
# 模型训练
em.fit(data)