'''
This file is designed to implement singular value decomposition (SVD) of complex matrices, while also providing a method for diagonalizing complex matrices and outputting the eigenvalues.
'''

from numpy import random

def EIG(matrix):
    '''
    This function is designed to perform matrix diagonalization, and the output is sorted according to the eigenvalue from largest to smallest
    '''
    matrixd, matrixv = np.linalg.eig(matrix)
    indexd = np.zeros(np.shape(matrixd));
    matrix0 = np.sort_complex(matrixd);
    matrix0 = matrix0[::-1]
    indexV = np.zeros(np.shape(matrixd));
    [m, n] = np.shape(matrix)

    j = -1;
    for i in range(m):
        indexVi = np.where(matrixd == matrix0[i]);
        if len(indexVi[0]) == 1:
            indexV[i] = indexVi[0];
        else:
            if i <= j:
                continue
            j = i
            for p in indexVi[0]:
                indexV[j] = p
                j = j + 1

    D = np.zeros(np.shape(matrixd));
    D = D.astype(np.complex64)
    V = np.zeros(np.shape(matrixv));
    V = V.astype(np.complex64)
    for i in range(len(matrixd)):
        D[i] = matrix0[i]
        V[:, i] = matrixv[:, int(indexV[i])]
    return [D, V]


# 奇异值分解
def svd(M):
    '''
    This function is designed to perform singular value decomposition (SVD) of matrices
    '''
    M_shape = np.shape(M)
    m = M_shape[0];
    n = M_shape[1];
    V_M = M.conj().T @ M

    output = EIG(V_M);
    Vd = output[0][0:min([m, n])];
    Vv = output[1];
    sigma = np.zeros(np.shape(M));
    sigma = sigma.astype(np.complex64)

    Uv = np.zeros([m, m])
    Uv = Uv.astype(np.complex64)
    U_M = M @ M.conj().T
    output = EIG(U_M);
    Ud = output[0][0:min([m, n])];
    Uv = output[1];
    Uv = Uv.T
    for i in range(min(m, n)):
        sigma[i, i] = np.sqrt(Vd[i])

    for i in range(len(Vd)):
        if sigma[i, i] != 0:
            Uv[:, i] = M @ Vv[:, i] / sigma[i, i];
    print('The matrix will be decomposed into the form of "U*sigma*V^dagger"')
    print('U, sigma, and V are respectively')
    print('U:\n', np.round(Uv, decimals=4))
    print('sigma:\n', np.round(sigma, decimals=4))
    print('V:\n', np.round(Vv, decimals=4))
    print('Verify the validity of the matrix decomposition. ')
    print('U*sigma*V.dagger\n', np.round(Uv @ sigma @ Vv.conj().T, decimals=4))
    print('Verify that U and V are unitary matrices')
    print('U.dagger*U\n', np.round(Uv @ Uv.conj().T, decimals=4))
    print('V.dagger*V\n', np.round(Vv @ Vv.conj().T, decimals=4))
    print('Verify that the singular values are positive')
    for i in range(min([m, n])):
        print('The {0} singular value is\n', np.round(sigma[i, i], decimals=4))


# 主函数部分
m = input('请输入矩阵的行数')
n = input('请输入矩阵的列数')
m = int(m)
n = int(n)
M_real = random.random(size=(m, n))
M_imag = random.random(size=(m, n))
M = M_real + M_imag * 1.j
M = np.round(M, decimals=4)
print('原矩阵为')
print(M)
svd(M)  
