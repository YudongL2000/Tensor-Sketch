import numpy as np
from numpy import matlib

def nchoosek(n, k):
    if k == 0:
        r = 1
    else:
        r = n/k * nchoosek(n-1, k-1)
    return round(r)

def RMFM(DATA,K,C,CS_COL):
  N = DATA.shape[0]
  D = DATA.shape[1]
  P = 2
  COEF = np.zeros(K+1)
  #print(COEF.shape)
  for i in range(K):
     COEF[i] = nchoosek(K, i) * (C**(K - i))
  DATA_SKETCH = np.zeros([N, CS_COL])
  R = np.floor(np.log2(np.divide(1, np.random.rand(CS_COL))))
  for i in range(CS_COL):
    iRan = int(R[i])
    #print(iRan, K)
    if (iRan >= K):
      continue
    else:
      const = np.sqrt(COEF[iRan + 1]) * np.sqrt(P ** (iRan + 1))
      if (iRan > 0):
        bitHash = (np.random.uniform(low=1.0, high=2.0, size=(iRan, D)) - 1.5).astype(np.double) * 2
        temp = np.matmul(DATA, bitHash.T)
        DATA_SKETCH[:, i] = const * np.prod(temp, axis=1)
      else:
        DATA_SKETCH[:, i] = const * np.ones(N)
  DATA_SKETCH = DATA_SKETCH / np.sqrt(CS_COL)
  return DATA_SKETCH

def FFT_CountSketch_k_Naive(DATA, K, CS_COL):
  N = DATA.shape[0]
  D = DATA.shape[1]
  indexHash = np.random.randint(low=1.0, high=CS_COL, size=(K, D))
  bitHASH = (np.random.uniform(low=1.0, high=2.0, size=(K, D)) - 1.5).astype(np.double) * 2
  DATA_SKETCH = np.zeros(shape=(N, CS_COL))
  P = np.zeros(shape=(K, CS_COL))
  for Xi in range(N):
    temp = DATA[Xi, :]
    P = np.zeros(shape=(K, CS_COL))
    for Xij in range(D):
      for Ki in range(K):
        iHashIndex = indexHash[Ki, Xij]
        #print(iHashIndex)
        iHashBit = bitHASH[Ki, Xij]
        P[Ki, iHashIndex] = P[Ki, iHashIndex] + iHashBit * temp[Xi]
    P = np.fft.fft(P, n=None, axis = 1)
    temp = np.prod(P, axis=0)
    DATA_SKETCH[Xi, :] = np.fft.ifft(temp)
  return DATA_SKETCH

C=1            
K=2
CS_COL=500