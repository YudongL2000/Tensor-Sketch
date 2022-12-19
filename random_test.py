import numpy as np
from numpy import linalg as LA
import time
import os
from numpy.random import seed, randint
from numpy import polymul, pad
from numpy.fft import fft, ifft
from timeit import default_timer as timer
from numpy.fft import rfft, irfft

def generate_large_dataset(dim, total_number, low, high):
	inputs = []
	for i in range(total_number):
		parser = []
		v1 = np.random.randint(low=low, high=high, size=dim, dtype=int)
		v2 = np.random.randint(low=low, high=high, size=dim, dtype=int)
		v3 = np.random.randint(low=low, high=high, size=dim, dtype=int)
		parser.append(v1)
		parser.append(v2)
		parser.append(v3)
		parser=np.array(parser)
	inputs.append(parser)
	inputs= np.array(inputs)
	return inputs

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

"""
def countSketchInMemroy(matrixA, s, hashedIndices, randSigns):
    m, n = matrixA.shape
    matrixC = np.zeros([m, s])
    #hashedIndices = np.random.choice(s, n, replace=True)
    #randSigns = np.random.choice(2, n, replace=True) * 2 - 1 # a n-by-1{+1, -1} vector
    matrixA = matrixA * randSigns.reshape(1, n) 
    for i in range(s):
        idx = (hashedIndices == i)
        matrixC[:, i] = np.sum(matrixA[:, idx], 1)
    return matrixC

def tensor_sketch(v1, v2, CS_DIM, hashedIndices, randSigns):
  assert(len(v1.shape)==2)
  assert(len(v2.shape)==2)
  dim1 = v1.shape[-1]
  dim2 = v2.shape[-1]
  assert(dim1 == dim2)
  sketched_vec1 = countSketchInMemroy(v1, CS_DIM, hashedIndices, randSigns)
  sketched_vec2 = countSketchInMemroy(v2, CS_DIM, hashedIndices, randSigns)
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
"""


def brute_force_outer_product(inputs):
  assert(inputs.shape[1] == 3)
  dim_n = inputs.shape[2]
  batches = inputs.shape[0]
  res = np.zeros([dim_n, dim_n, dim_n])
  for batch in range(batches):
    vec_A = inputs[batch, 0, :]
    vec_B = inputs[batch, 1, :]
    vec_C = inputs[batch, 2, :]
    prod_tmp = np.outer(vec_B, vec_C)

    prod = np.outer(vec_A, prod_tmp).reshape(dim_n,dim_n,dim_n)
    res += prod
  return res


def getCountSketchMatrix(matrixA, hashedIndices, randSigns, sketch_dim):
  m, n = matrixA.shape
  matrixC = np.zeros([m, sketch_dim])
  matrixA = matrixA * randSigns # flip the signs of 50% columns of A
  for i in range(sketch_dim):
      idx = (hashedIndices == i)
      matrixC[:, i] = np.sum(matrixA[:, idx], 1)
  return matrixC


def sketch_outer_prod(u, v, hashedIndices, sketch_dim, randSigns):
  prod = np.outer(u, v).flatten()
  return getCountSketchMatrix(prod.reshape(1, len(prod)), hashedIndices, randSigns, sketch_dim)

def tensor_algo(inputs, sketch_dim1, sketch_dim2):
	dim_n = inputs.shape[-1]
	hashedIndices1 = np.random.choice(sketch_dim1, dim_n**2, replace=True)
	randSigns1 = np.random.choice(2, dim_n**2, replace=True) * 2 - 1
	randSigns1.reshape(1, dim_n**2)

	hashedIndices2 = np.random.choice(sketch_dim2, dim_n**2, replace=True)
	randSigns2 = np.random.choice(2, dim_n**2, replace=True) * 2 - 1
	randSigns2.reshape(1, dim_n**2)

	A1_R = np.zeros([dim_n, sketch_dim1])
	A2_R = np.zeros([dim_n, sketch_dim1])
	A3_T = np.zeros([dim_n, sketch_dim2])

	hashedIndices_tensor = np.random.choice(sketch_dim1, inputs.shape[-1], replace=True)
	randSigns_tensor = np.random.choice(2, inputs.shape[-1], replace=True) * 2 - 1
	for input in inputs:
		s3 = sketch_outer_prod(input[0], input[1], hashedIndices2, sketch_dim2, randSigns2)
		s1 = tensor_sketch(np.reshape(input[1], (1, len(input[1]))), np.reshape(input[2], (1, len(input[2]))), sketch_dim1)
		s2 = tensor_sketch(np.reshape(input[0], (1, len(input[0]))), np.reshape(input[2], (1, len(input[2]))), sketch_dim1)
		#s1 = tensor_sketch(np.reshape(input[1], (1, len(input[1]))), np.reshape(input[2], (1, len(input[2]))), sketch_dim1, hashedIndices_tensor, randSigns_tensor)
		#s2 = tensor_sketch(np.reshape(input[0], (1, len(input[0]))), np.reshape(input[2], (1, len(input[2]))), sketch_dim1, hashedIndices_tensor, randSigns_tensor)
		#s3 = tensor_sketch(np.reshape(input[0], (1, len(input[0]))), np.reshape(input[1], (1, len(input[1]))), sketch_dim2)
		A1_R += np.outer(input[0], s1)
		A2_R += np.outer(input[1], s2)
		A3_T += np.outer(input[2], s3)
	ZT = []
	for i in range(sketch_dim1):
		for j in range(sketch_dim1):
			ZT.append(sketch_outer_prod(A1_R[:, i], A2_R[:, j], hashedIndices2, sketch_dim2, randSigns2))
	ZT = np.concatenate(ZT)
	# assert(ZT.shape[0] == sketch_dim ** 2)
	# assert(ZT.shape[1] == sketch_dim2)
	ZT_t = np.transpose(ZT)
	inv = LA.pinv(np.matmul(ZT_t, ZT))
	# assert(inv.shape[0] == sketch_dim2)
	# assert(inv.shape[1] == sketch_dim2)
	X = np.matmul(A3_T, np.matmul(inv, ZT_t))
	return np.matmul(X, ZT)


if __name__ == '__main__':
	expanded_dim = 1000
	total_number = 2000000
	load_start = time.perf_counter()
	low = 0
	high = 10
	inputs = generate_large_dataset(expanded_dim, total_number, low, high)
	load_end = time.perf_counter()
	print(f"data loaded in {load_end - load_start:0.4f} seconds")

	sketch_dim1 = 20
	sketch_dim2 = 100

	sketch_dim11 = 5
	sketch_dim12 = 20

	sketch_dim21 = 5
	sketch_dim22 = 50
	
	sketch_dim31 = 10
	sketch_dim32 = 20

	sketch_dim41 = 10
	sketch_dim42 = 50

	sketch_dim51 = 10
	sketch_dim52 = 100

	sketch_dim61 = 20
	sketch_dim62 = 100



	compute_start = time.perf_counter() 
	res = brute_force_outer_product(inputs)
	compute_end = time.perf_counter()
	print(f"computed baseline in {compute_end - compute_start:0.4f} seconds")
	
	compute_start = time.perf_counter() 
	res12 = tensor_algo(inputs, sketch_dim11, sketch_dim12)
	compute_end = time.perf_counter()
	print(f"computed my algo tensor sketch with in {compute_end - compute_start:0.4f} seconds", sketch_dim11, sketch_dim12)

	compute_start = time.perf_counter() 
	res22 = tensor_algo(inputs, sketch_dim21, sketch_dim22)
	compute_end = time.perf_counter()
	print(f"computed my algo tensor sketch with in {compute_end - compute_start:0.4f} seconds", sketch_dim21, sketch_dim22)


	compute_start = time.perf_counter() 
	res32 = tensor_algo(inputs, sketch_dim31, sketch_dim32)
	compute_end = time.perf_counter()
	print(f"computed my algo tensor sketch with in {compute_end - compute_start:0.4f} seconds", sketch_dim31, sketch_dim32)

	compute_start = time.perf_counter() 
	res42 = tensor_algo(inputs, sketch_dim41, sketch_dim42)
	compute_end = time.perf_counter()
	print(f"computed my algo tensor sketch with in {compute_end - compute_start:0.4f} seconds", sketch_dim41, sketch_dim42)

	compute_start = time.perf_counter() 
	res52 = tensor_algo(inputs, sketch_dim51, sketch_dim52)
	compute_end = time.perf_counter()
	print(f"computed my algo tensor sketch with in {compute_end - compute_start:0.4f} seconds", sketch_dim51, sketch_dim52)

	compute_start = time.perf_counter() 
	res62 = tensor_algo(inputs, sketch_dim61, sketch_dim62)
	compute_end = time.perf_counter()
	print(f"computed my algo tensor sketch with in {compute_end - compute_start:0.4f} seconds", sketch_dim61, sketch_dim62)

	norm = LA.norm(res.reshape(inputs.shape[-1], -1), ord="fro")
	norm12 = LA.norm(res12, ord="fro")
	print("correct norm: ", norm)
	#print("my algo norm: ", norm2)

	print(f"error: {np.abs(norm12 - norm) / norm * 100:0.6f}%", sketch_dim11, sketch_dim12)

	norm22 = LA.norm(res22, ord="fro")
	print(f"error: {np.abs(norm22 - norm) / norm * 100:0.6f}%", sketch_dim21, sketch_dim22)

	norm32 = LA.norm(res32, ord="fro")
	print(f"error: {np.abs(norm32 - norm) / norm * 100:0.6f}%", sketch_dim31, sketch_dim32)

	norm42 = LA.norm(res42, ord="fro")
	print(f"error: {np.abs(norm42 - norm) / norm * 100:0.6f}%", sketch_dim41, sketch_dim42)

	norm52 = LA.norm(res52, ord="fro")
	print(f"error: {np.abs(norm52 - norm) / norm * 100:0.6f}%", sketch_dim51, sketch_dim52)

	norm62 = LA.norm(res62, ord="fro")
	print(f"error: {np.abs(norm22 - norm) / norm * 100:0.6f}%", sketch_dim61, sketch_dim62)



  





