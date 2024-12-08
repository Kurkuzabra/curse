from scipy.spatial.distance import pdist, squareform
from itertools import permutations, combinations
from math import comb
import numpy as np

import sys
sys.path.insert(0, '/home/kurk/curse/filtration/graph_func.cpython-312-x86_64-linux-gnu.so')

import graph_func as gf
from timeit import default_timer as timer

print(gf.__file__)
print(np.__file__)
print(gf.filtrate.__doc__)

n = 150
X = np.random.normal(size=(n, 2))
A = squareform(pdist(X))

#############################################

#для фильтрации до 5 многопоточность особого значения не имеет

num_cores = 1

start = timer()
es = gf.filtrate(A, 2, num_cores)
ts = gf.filtrate(A, 3, num_cores)
qs = gf.filtrate(A, 4, num_cores)
end = timer()
print("c++ ver time on {} threads: {}".format(num_cores, end - start))

#############################################

num_cores = 4

start = timer()
es = gf.filtrate(A, 2, num_cores)
ts = gf.filtrate(A, 3, num_cores)
qs = gf.filtrate(A, 4, num_cores)
end = timer()
print("c++ ver time on {} threads: {}".format(num_cores, end - start))

#############################################

n = 50
X = np.random.normal(size=(n, 2))
A = squareform(pdist(X))

start = timer()
ps = gf.filtrate(A, 5, 1)
end = timer()
print("c++ ver time on {} threads: {}".format(1, end - start))

#############################################

start = timer()
ps = gf.filtrate(A, 5, 32)
end = timer()
print("c++ ver time on {} threads: {}".format(32, end - start))

#############################################
# start = timer()

# es = np.zeros((comb(n, 2), 3))
# ts = np.zeros((comb(n, 3), 3))
# qs = np.zeros((comb(n, 4), 3))

# for i, simplex in enumerate(combinations(range(n), 2)):
#     es[i,0] = gf.f_single_thread(A, simplex, 1)
#     es[i,1] = gf.f_single_thread(A, simplex, 2)
#     es[i,2] = gf.f_single_thread(A, simplex, np.inf)

# for i, simplex in enumerate(combinations(range(n), 3)):
#     ts[i,0] = gf.f_single_thread(A, simplex, 1)
#     ts[i,1] = gf.f_single_thread(A, simplex, 2)
#     ts[i,2] = gf.f_single_thread(A, simplex, np.inf)

# for i, simplex in enumerate(combinations(range(n), 4)):
#     qs[i,0] = gf.f_single_thread(A, simplex, 1)
#     qs[i,1] = gf.f_single_thread(A, simplex, 2)
#     qs[i,2] = gf.f_single_thread(A, simplex, np.inf)

# end = timer()
# print("c++ ver time on {} threads: {}".format(1, end - start))

#############################################

# for num_threads in [4]:

#     start = timer()

#     es = np.zeros((comb(n, 2), 3))
#     ts = np.zeros((comb(n, 3), 3))
#     qs = np.zeros((comb(n, 4), 3))

#     for i, simplex in enumerate(combinations(range(n), 2)):
#         es[i,0] = gf.f_multithread(A, simplex, 1, num_threads)
#         es[i,1] = gf.f_multithread(A, simplex, 2, num_threads)
#         es[i,2] = gf.f_multithread(A, simplex, np.inf, num_threads)

#     for i, simplex in enumerate(combinations(range(n), 3)):
#         ts[i,0] = gf.f_multithread(A, simplex, 1, num_threads)
#         ts[i,1] = gf.f_multithread(A, simplex, 2, num_threads)
#         ts[i,2] = gf.f_multithread(A, simplex, np.inf, num_threads)

#     for i, simplex in enumerate(combinations(range(n), 4)):
#         qs[i,0] = gf.f_multithread(A, simplex, 1, num_threads)
#         qs[i,1] = gf.f_multithread(A, simplex, 2, num_threads)
#         qs[i,2] = gf.f_multithread(A, simplex, np.inf, num_threads)

#     end = timer()
#     print("c++ ver time on {} threads: {}".format(num_threads, end - start))

##############################################


# def subsequences(seq):
#     return [[list(subseq) for subseq in combinations(seq, sublen)] for sublen in range(2, len(seq)+1)]

# def pairs(seq):
#     return [seq[i:i+2] for i in range(len(seq)-1)]

# def f(A, simplex, p=np.inf):

#     fs = []
#     for seq_p in permutations(simplex):

#         ds = np.zeros(0)
#         for dim, items in enumerate(subsequences(seq_p), start=1):
#             for item in items:
#                 vec = A[(*np.array(pairs(item)).T,)]
#                 ds = np.concatenate([ds, [np.linalg.norm(vec, p)]])
#         fs.append(np.max(ds))

#     return np.min(fs)

# start = timer()

# for i, simplex in enumerate(combinations(range(n), 2)):
#     es[i,0] = f(A, simplex, 1)
#     es[i,1] = f(A, simplex, 2)
#     es[i,2] = f(A, simplex, np.inf)

# for i, simplex in enumerate(combinations(range(n), 3)):
#     ts[i,0] = f(A, simplex, 1)
#     ts[i,1] = f(A, simplex, 2)
#     ts[i,2] = f(A, simplex, np.inf)

# end = timer()
# print("python ver time(for combs of 2 and 3): {}".format(end - start))