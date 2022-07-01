
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:08:19 2020

@author: zxw
"""
import random
import numpy as np
import time


class WeightedQuickUnionUF:
    """运用WeightedQuickUnionUF数据结构将二维的sites表示为一维"""

    def __init__(self, n):
        self.count = n
        self.size = [1 for i in range(n)]  # [1, 1, ..., 1]
        self.parent = [i for i in range(n)] # [0, 1, ..., n-1]

    def validate(self, p):  # 检测P值是否有效
        if (p < 0 or p >= len(self.size)):
            raise Exception('Out of range!')

    def find(self, p):  # 找到格子p
        self.validate(p)
        while p != self.parent[p]:
            p = self.parent[p]
        return p

    def connected(self, p, q):  # 检测格子p和q的值是否相等
        self.validate(p)
        self.validate(q)
        return self.find(p) == self.find(q)    
    
    def union(self, p, q):   # 合并格子p和q
        self.validate(p)
        self.validate(q)
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ: 
            return
        if self.size[rootP] < self.size[rootQ]:
            self.parent[rootP] = rootQ
            self.size[rootQ] += self.size[rootP]
        else:
            self.parent[rootQ] = rootP
            self.size[rootP] += self.size[rootQ]
        
class Percolation:

    def __init__(self, N):
        self.Array = [[0 for i in range(N)] for j in range(N)]  # N×N 的二维数组
        self.size = N
        # 第 0 个元素和第 N×N+1 个元素为人为定义的虚元素
        # 第 k 个元素对应了二维数组中的 (i, j) (k=i×N+j+1), 这里左上为（0,0）
        self.UF = WeightedQuickUnionUF(N*N+2) 

    # is site (row i, column j) open?
    def isOpen(self, i, j):
        return self.Array[i][j] == 1

    # 每打开一个闭格，观察该闭格上下左右是否有开格。
    # 如果有，则将它们在QuickFindUF类中所对应的位置连接起来。
    def Open(self, i, j):
        if self.Array[i][j] == 1:
            # print('('+str(i)+','+str(j)+') has already been open!')
            return
        else: # open site (row i, column j) if it is not open already
            self.Array[i][j] = 1
            # print('('+str(i)+','+str(j)+') is open now!')
            if i == 0: # 如果打开的闭格在最上层，则将其与第 0 个虚元素连接起来
                self.UF.union(0, i*self.size+j+1)
            if i == self.size-1: # 如果打开的闭格在最下层，则将其与第 N×N+1 个虚元素连接起来
                self.UF.union(self.size*self.size+1, i*self.size+j+1)
            if i != 0:
                if self.isOpen(i-1, j):
                    self.UF.union(i*self.size+j+1, (i-1)*self.size+j+1)
            if i != self.size-1:
                if self.isOpen(i+1, j):
                    self.UF.union(i*self.size+j+1, (i+1)*self.size+j+1)
            if j !=0 :
                if self.isOpen(i, j-1):
                    self.UF.union(i*self.size+j+1, i*self.size+(j-1)+1)
            if j != self.size-1:
                if self.isOpen(i, j+1):
                    self.UF.union(i*self.size+j+1, i*self.size+(j+1)+1)
  
    # is site (row i, column j) full?
    def isFull(self, i, j):
        return self.UF.connected(0, i*self.size+j+1)
    
    # does the system percolate?
    def percolates(self):
        return self.UF.connected(0, self.size*self.size+1)
    
class PercolationStats:

    def __init__(self, N, T):
        self.threshold = []
        self.times = T
        self.lat = []

        for k in range(T):
            print(k)
            experiment = Percolation(N)
            count = 0
            while (not experiment.percolates()):
                block_list = []
                for i in range(N):
                    for j in range(N):
                        if not experiment.isOpen(i, j):
                            block_list.append(i*N+j)
                to_open = random.choice(block_list)
                to_open_j = int(to_open % N)
                to_open_i = int((to_open-to_open_j)/N) 
                experiment.Open(to_open_i,to_open_j)
                count = count+1
            self.threshold.append(count/(N*N))
            self.lat.append(experiment.Array)

        self.Array = np.array(self.threshold)
        self.lattice = np.array(self.lat)

    def save_data(name):
        import h5py
        with h5py.File(r".\data\{}.h5".format(name),'w') as hf:
            hf.create_dataset("elem", data=self.lattice, compression="gzip", compression_opts=9)
            hf.close()

    def mean(self):
        return np.mean(self.Array)

    def stddev(self):
        return np.std(self.Array, ddof=1)

    def confidenceLow(self):
        return self.mean()-1.96*self.stddev()/(self.times**0.5)

    def confidenceHigh(self):
        return self.mean()+1.96*self.stddev()/(self.times**0.5)

if __name__ == '__main__':
    time_start=time.time()
    test = PercolationStats(28, 1000)
    test.save_data(x2)  

    print(test.mean())
    time_end=time.time()
    print('totally cost',time_end-time_start)
