import eecs598
import torch
import torchvision
import matplotlib.pyplot as plt
import statistics
import random
import time
import math


from eecs598 import reset_seed, Solver
from convolutional_networks import DeepConvNet
from fully_connected_networks import adam

reset_seed(0)
num_inputs = 2
input_dims = (3, 8, 8)
num_classes = 10
N = 50
X = torch.randn(N, *input_dims, dtype=torch.float32, device='mps')
y = torch.randint(10, size=(N,), dtype=torch.int64, device='mps')

for reg in [0, 3.14]:
  print('Running check with reg = ', reg)
  model = DeepConvNet(input_dims=input_dims, num_classes=num_classes,
                      num_filters=[8, 8, 8],
                      max_pools=[0, 2],
                      reg=reg,
                      weight_scale=5e-2, dtype=torch.float32, device='mps')

  loss, grads = model.loss(X, y)
  # The relative errors should be up to the order of e-6
  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eecs598.grad.compute_numeric_gradient(f, model.params[name])
    print('%s max relative error: %e' % (name, eecs598.grad.rel_error(grad_num, grads[name])))
  if reg == 0: print()
  
  
  
  
  '''
        loss, dout = softmax_loss(scores, y)
        for l in range(self.num_layers):
          w = self.params[f'W{l+1}']
          loss += self.reg * torch.sum(w ** 2)
        dout,grads[f'W{self.num_layers}'],grads[f'b{self.num_layers}'] = Linear.backward(dout,caches.pop())
        for i in range(self.num_layers-1):
           if self.num_layers- 2 - i in self.max_pools:
             dout,grads[f'W{self.num_layers -1 - i}'],grads[f'b{self.num_layers -1 - i}'] = Conv_ReLU_Pool.backward(dout,caches.pop())
           else:
             dout,grads[f'W{self.num_layers -1 - i}'],grads[f'b{self.num_layers -1 - i}'] = Conv_ReLU.backward(dout,caches.pop())
        for i in range(self.num_layers):
          self.params[f'W{i+1}'] += 2*self.reg*self.params[f'W{i+1}']
          
        caches = []
        out = X
        for i in range(self.num_layers-1):
          w = self.params[f"W{i+1}"]
          b = self.params[f"b{i+1}"]
          if i in self.max_pools:
            out,cache = Conv_ReLU_Pool.forward(out,w,b,conv_param,pool_param)
            caches.append(cache)
          else :
            out,cache = Conv_ReLU.forward(out,w,b,conv_param)
            caches.append(cache)
        scores,cache = Linear.forward(out,self.params[f'W{self.num_layers}'],self.params[f'b{self.num_layers}'])
        caches.append(cache)  
          
          
          
          
  '''