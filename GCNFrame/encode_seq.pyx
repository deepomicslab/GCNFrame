import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)

cdef str _num_transfer(str seq, str seqtype):
    if seqtype == "DNA":
        seq = seq.replace("A", "0").replace("C", "1").replace("G", "2").replace("T", "3")
    elif seqtype == "RNA":
        seq = seq.replace("A", "0").replace("C", "1").replace("G", "2").replace("U", "3")
    seq = ''.join(list(filter(str.isdigit, seq)))

    return seq    


cdef list _num_transfer_loc(str num_seq, int K):
    cdef list loc
    loc = []
    for i in range(0, len(num_seq)-K+1):
        loc.append(int(num_seq[i:i+K], 4))
    
    return loc

cdef np.ndarray[np.float64_t, ndim=1] _loc_transfer_matrix(list loc_list, list dis_list, int K, int length):
    cdef np.ndarray[np.float64_t, ndim=2] matrix
    cdef np.ndarray[np.float64_t, ndim=1] new_matrix
    cdef int num
    matrix = np.zeros((4**K, 4**K))
    num = 0
    for dis in dis_list:
        for i in range(0, len(loc_list)-K-dis):
            matrix[loc_list[i]][loc_list[i+K+dis]] += 1
        num = num + (length - 2*K - dis + 1.0)
    
    matrix = matrix / num
    
    new_matrix = matrix.flatten()
    
    return new_matrix

cdef np.ndarray[np.float64_t, ndim=1] _matrix_encoding(str seq, int K, int d, str seqtype):
    cdef int length
    cdef np.ndarray[np.float64_t, ndim=1] feature
    cdef str num_seq
    cdef list loc, dis
    seq = seq.upper()
    length = len(seq)
    num_seq = _num_transfer(seq, seqtype)
    loc = _num_transfer_loc(num_seq, K)
    dis = [list(range(0, 1)), list(range(1, 2)), list(range(2, 3)),
            list(range(3, 5)), list(range(5, 9)), list(range(9, 17)), list(range(17, 33)),
            list(range(33, 65))]
    if d == 1:
        feature = np.hstack((_loc_transfer_matrix(loc, list(range(0, 1)), K, length)))
    
    elif d == 2:
        feature = np.hstack((
            _loc_transfer_matrix(loc, list(range(0, 1)), K, length),
            _loc_transfer_matrix(loc, list(range(1, 2)), K, length)))
    else:
        feature = np.hstack((
            _loc_transfer_matrix(loc, list(range(0, 1)), K, length),
            _loc_transfer_matrix(loc, list(range(1, 2)), K, length)))
        for i in range(2, d):
            feature = np.hstack((feature, _loc_transfer_matrix(loc, dis[i], K, length)))
     
    return feature * 100

def matrix_encoding(seq, K, d, seqtype):

    return _matrix_encoding(seq, K, d, seqtype)

