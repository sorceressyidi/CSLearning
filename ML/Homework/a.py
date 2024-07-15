import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, binom

class MetropolisHastings:
    def __init__(self, proposal_dist, target_dist, burn_in=1000, num_samples=10000):
        """
        Metropolis Hastings

        :param proposal_dist: 建议分布
        :param target_dist: 目标分布
        :param burn_in: 收敛步数
        :param num_samples: 样本数量
        """
        self.proposal_dist = proposal_dist
        self.target_dist = target_dist
        self.burn_in = burn_in
        self.num_samples = num_samples

    @staticmethod
    def __calc_acceptance_ratio(q, p, x, x_prime):
        """
        计算接受概率

        :param q: 建议分布
        :param p: 目标分布
        :param x: 当前状态
        :param x_prime: 候选状态
        :return: 接受概率
        """
        prob_1 = p.prob(x_prime) * q.joint_prob(x_prime, x)
        prob_2 = p.prob(x) * q.joint_prob(x, x_prime)
        alpha = min(1., prob_1 / prob_2)
        return alpha

    def sample(self):
        """
        Metropolis Hastings 算法采样
        :return: 样本数组、样本均值、样本方差
        """
        all_samples = np.zeros(self.num_samples)
        x_0 = np.random.random()

        for i in range(self.num_samples):
            x = x_0 if i == 0 else all_samples[i - 1]
            x_prime = self.proposal_dist.sample()
            alpha = self.__calc_acceptance_ratio(self.proposal_dist, self.target_dist, x, x_prime)
            u = np.random.uniform(0, 1)

            if u <= alpha:
                all_samples[i] = x_prime
            else:
                all_samples[i] = x

        samples = all_samples[self.burn_in:]
        dist_mean = samples.mean()
        dist_var = samples.var()
        return samples, dist_mean, dist_var

    @staticmethod
    def visualize(samples, bins=50):
        """
        可视化展示
        :param samples: 抽取的样本集合
        :param bins: 直方图的分组个数
        """
        plt.figure(figsize=(10, 6))
        plt.hist(samples, bins, density=True, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Metropolis Hastings Sample Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.xlim(0, 1)
        plt.grid(True)
        plt.show()
class ProposalDistribution:
    """
    建议分布
    """

    @staticmethod
    def sample():
        """
        从建议分布中抽取一个样本
        :return: 样本值
        """
        return beta.rvs(1, 1)

    @staticmethod
    def prob(x):
        """
        P(X = x) 的概率
        :param x: 样本值
        :return: 概率
        """
        return beta.pdf(x, 1, 1)

    def joint_prob(self, x_1, x_2):
        """
        P(X = x_1, Y = x_2) 的联合概率
        :param x_1: 样本值1
        :param x_2: 样本值2
        :return: 联合概率
        """
        return self.prob(x_1) * self.prob(x_2)
class TargetDistribution:
    """
    目标分布
    """

    @staticmethod
    def prob(x):
        """
        P(X = x) 的概率
        :param x: 样本值
        :return: 概率
        """
        return binom.pmf(4, 10, x)

import warnings
warnings.filterwarnings("ignore")

# 参数设置
burn_in = 1000
num_samples = 10000

# 创建建议分布和目标分布
proposal_dist = ProposalDistribution()
target_dist = TargetDistribution()

# Metropolis-Hastings 算法实例
mh = MetropolisHastings(proposal_dist, target_dist, burn_in, num_samples)

# 采样
samples, dist_mean, dist_var = mh.sample()
print("均值:", dist_mean)
print("方差:", dist_var)

# 可视化结果
mh.visualize(samples, bins=50)
