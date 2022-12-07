import numpy as np
from numpy import linalg as LA
import time
import os
from numpy.random import seed, randint
from numpy import polymul, pad
from numpy.fft import fft, ifft
from timeit import default_timer as timer
from numpy.fft import rfft, irfft

#tensor sketch implementation
def fft_prod(arr_a, arr_b):  #fft based real-valued polynomial multiplication
    L = len(arr_a) + len(arr_b)
    a_f = rfft(arr_a, L)
    b_f = rfft(arr_b, L)
    return irfft(a_f * b_f)

def countSketchInMemroy(matrixA, s):
    m, n = matrixA.shape
    matrixC = np.zeros([m, s])
    hashedIndices = np.random.choice(s, n, replace=True)
    randSigns = np.random.choice(2, n, replace=True) * 2 - 1 # a n-by-1{+1, -1} vector
    matrixA = matrixA * randSigns.reshape(1, n) # flip the signs of 50% columns of A
    for i in range(s):
        idx = (hashedIndices == i)
        matrixC[:, i] = np.sum(matrixA[:, idx], 1)
    return matrixC
    
def tensor_sketch(v1, v2, CS_DIM):
  assert(len(v1.shape)==2)
  assert(len(v2.shape)==2)
  dim1 = v1.shape[-1]
  dim2 = v2.shape[-1]
  assert(dim1 == dim2)
  sketched_vec1 = countSketchInMemroy(v1, CS_DIM)
  sketched_vec2 = countSketchInMemroy(v2, CS_DIM)
  fft_res = []
  for i in range(sketched_vec1.shape[0]):
    fft_res.append(fft_prod(sketched_vec1[i], sketched_vec2[i]))
  fft_res = np.array(fft_res)
  assert(fft_res.shape[-1] == 2 * sketched_vec1.shape[-1])
  res = []
  for i in range(fft_res.shape[0]):
    tmp = np.zeros(sketched_vec1.shape[-1])
    for j in range(sketched_vec1.shape[-1]):
      tmp[j] = fft_res[i, j] + fft_res[i, j+sketched_vec1.shape[-1]]
    res.append(tmp)
  res = np.array(res)
  return res









dim_n = 5

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
      duplicated = []
      for i in range(int(dim_n / 5)):
        duplicated.extend(curr_vec)
      vectors.append(duplicated)
      curr_vec = []
  # vectors: list of 3*n elements where each element is a 5-d vector
  # assert(len(vectors) % 3 == 0)
  result = np.array(vectors)
  result = np.reshape(result, (3, -1, dim_n))
  result = np.transpose(result, (1, 0, 2))
  return result

#dim = batchsize * 5 * 5
#Result 5 * 5 * 5
def brute_force_outer_product(inputs):
  # assert(inputs.shape[1] == 3)
  # assert(inputs.shape[2] == 5)
  batches = inputs.shape[0]
  res = np.zeros([dim_n, dim_n, dim_n])
  for batch in range(batches):
    vec_A = inputs[batch, 0, :]
    vec_B = inputs[batch, 1, :]
    vec_C = inputs[batch, 2, :]
    prod_tmp = np.outer(vec_B, vec_C)
    # assert(prod_tmp.shape[0] == 5)
    # assert(prod_tmp.shape[1] == 5)

    prod = np.outer(vec_A, prod_tmp).reshape(dim_n,dim_n,dim_n)
    # assert(prod.shape[0] == 5)
    # assert(prod.shape[1] == 5)
    # assert(prod.shape[2] == 5)
    res += prod
  return res


sketch_dim = 3 # sketch size, can be tuned
sketch_dim2 = 5 # sketch size, can be tuned

hashedIndices = np.random.choice(sketch_dim, dim_n**2, replace=True)
randSigns = np.random.choice(2, dim_n**2, replace=True) * 2 - 1 # a n-by-1{+1, -1} vector
randSigns.reshape(1, dim_n**2)

hashedIndices2 = np.random.choice(sketch_dim2, dim_n**2, replace=True)
randSigns2 = np.random.choice(2, dim_n**2, replace=True) * 2 - 1 # a n-by-1{+1, -1} vector
randSigns2.reshape(1, dim_n**2)

# reference: http://wangshusen.github.io/code/countsketch.html
def getCountSketchMatrix(matrixA):
  m, n = matrixA.shape
  matrixC = np.zeros([m, sketch_dim])
  matrixA = matrixA * randSigns # flip the signs of 50% columns of A
  for i in range(sketch_dim):
      idx = (hashedIndices == i)
      matrixC[:, i] = np.sum(matrixA[:, idx], 1)
  return matrixC

def getCountSketchMatrix2(matrixA):
  m, n = matrixA.shape
  matrixC = np.zeros([m, sketch_dim2])
  matrixA = matrixA * randSigns2 # flip the signs of 50% columns of A
  for i in range(sketch_dim2):
      idx = (hashedIndices2 == i)
      matrixC[:, i] = np.sum(matrixA[:, idx], 1)
  return matrixC

def sketch_outer_prod(u, v):
  prod = np.outer(u, v).flatten()
  return getCountSketchMatrix(prod.reshape(1, len(prod)))

def sketch_outer_prod2(u, v):
  prod = np.outer(u, v).flatten()
  return getCountSketchMatrix2(prod.reshape(1, len(prod)))

def tensor_algo(inputs):
  A1_R = np.zeros([dim_n, sketch_dim])
  A2_R = np.zeros([dim_n, sketch_dim])
  A3_T = np.zeros([dim_n, sketch_dim2])
  for input in inputs:
    # s1 = sketch_outer_prod(input[1], input[2])
    # s2 = sketch_outer_prod(input[0], input[2])
    s1 = tensor_sketch(np.reshape(input[1], (1, len(input[1]))), np.reshape(input[2], (1, len(input[2]))), sketch_dim)
    s2 = tensor_sketch(np.reshape(input[0], (1, len(input[0]))), np.reshape(input[2], (1, len(input[2]))), sketch_dim)
    print(s1.shape)
    s3 = sketch_outer_prod2(input[0], input[1])
    A1_R += np.outer(input[0], s1)
    A2_R += np.outer(input[1], s2)
    A3_T += np.outer(input[2], s3)

  ZT = []
  for i in range(sketch_dim):
    for j in range(sketch_dim):
      ZT.append(sketch_outer_prod2(A1_R[:, i], A2_R[:, j]))
  ZT = np.concatenate(ZT)
  # assert(ZT.shape[0] == sketch_dim ** 2)
  # assert(ZT.shape[1] == sketch_dim2)
  ZT_t = np.transpose(ZT)
  inv = LA.pinv(np.matmul(ZT_t, ZT))
  # assert(inv.shape[0] == sketch_dim2)
  # assert(inv.shape[1] == sketch_dim2)
  X = np.matmul(A3_T, np.matmul(inv, ZT_t))
  return np.matmul(X, ZT)

def compute():
  load_start = time.perf_counter()
  os.chdir("CleanedData/")
  inputs_vec = []
  limit = 1
  i = 0
  print(len(os.listdir())) # 8385
  for file in os.listdir():
    inputs_vec.append(load_data(file))
    i += 1
    if (i == limit):
      break
  inputs = np.concatenate(inputs_vec)
  print(inputs.shape)
  load_end = time.perf_counter()
  print(f"data loaded in {load_end - load_start:0.4f} seconds")

  compute_start = time.perf_counter() 
  res = brute_force_outer_product(inputs)
  compute_end = time.perf_counter()
  print(f"computed baseline in {compute_end - compute_start:0.4f} seconds")
  print("res:")
  print(res)

  compute_start = time.perf_counter() 
  res2 = tensor_algo(inputs)
  compute_end = time.perf_counter()
  print(f"computed my algo in {compute_end - compute_start:0.4f} seconds")
  print("res:")
  print(res2)
  norm = LA.norm(res.reshape(5, -1), ord="fro")
  norm2 = LA.norm(res2, ord="fro")
  print("correct norm: ", norm)
  print("my algo norm: ", norm2)
  print(f"error: {np.abs(norm2 - norm) / norm * 100:0.6f}%", )

if __name__ == '__main__':
  compute()

  
