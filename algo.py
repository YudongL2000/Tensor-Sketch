import numpy as np
from numpy import linalg as LA
import math
import time
import random
import csv
import pickle
import json
import os

from TensorSketch import OblvFeat, TensorInit, TensorSketch

def load_data(file):
  f = open(file, "r")
  str = ""
  for c in f.read():
    if c == 'T':
      break
    str += c
  l = str.split()
  
  vectors = []
  curr_vec = []
  for n in l:
    if len(n) == 1:
      continue
    curr_vec.append(float(n))
    if len(curr_vec) == 5:
      vectors.append(curr_vec)
      curr_vec = []
  # vectors: list of 3*n elements where each element is a 5-d vector
  assert(len(vectors) % 3 == 0)
  result = np.array(vectors)
  result = np.reshape(result, (3, -1, 5))
  result = np.transpose(result, (1, 0, 2))
  #print(result)
  return result
  # result.shape = (n, 3, 5)

#dim = batchsize * 5 for each vector
#Result 5 * 5 * 5
def brute_force_outer_product(input_vec):
  assert(input_vec.shape[1] == 3)
  assert(input_vec.shape[2] == 5)
  batches = input_vec.shape[0]
  res = np.zeros([5, 5, 5])
  for batch in range(batches):
    vec_A = input_vec[batch, 0, :]
    vec_B = input_vec[batch, 1, :]
    vec_C = input_vec[batch, 2, :]
    prod_tmp = np.outer(vec_B, vec_C)
    assert(prod_tmp.shape[0] == 5)
    assert(prod_tmp.shape[1] == 5)
    prod = np.outer(vec_A, prod_tmp).reshape(5,5,5)
    assert(prod.shape[0] == 5)
    assert(prod.shape[1] == 5)
    assert(prod.shape[2] == 5)
    res += prod
  return res


def brute_force():
  load_start = time.perf_counter()
  # os.chdir("data/")
  os.chdir("CleanedData/")
  all_tensors = []
  limit = 10000
  i = 0
  print(len(os.listdir())) # 8385
  for file in os.listdir():
    all_tensors.append(load_data(file))
    i += 1
    if (i == limit):
      break
  load_end = time.perf_counter()
  print(f"data loaded in {load_end - load_start:0.4f} seconds")

  compute_start = time.perf_counter() 
  res = np.zeros([5, 5, 5])
  for tensor in all_tensors:
    res += brute_force_outer_product(tensor)

  compute_end = time.perf_counter()
  print(f"computed in {compute_end - compute_start:0.4f} seconds")
  # print("res:")     
  # print(res)
        
if __name__ == '__main__':
  brute_force()

  