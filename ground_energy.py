#Calculate the ground state energy E0 and the first excited state energy E1 
#for spin-1/2 antiferromagnetic Heisenberg chains of lengths L = 4, 6, 8, 10, 12, respectively 
#where the hamitonian is defined as H = S_i * S_i+1 in periodic boundary condition
#The energy gap for each length is defined as ∆ = E1 - E0, and caculate the gap
#In this file, U(1) symmetry wull be used to simplified calculation

import numpy as np

def qN(L):
    #caculate quantum numbers
    a = [1, -1]
    b = [1, -1]
    for n in range(L - 1):
        c = [];
        for i in a:
            for j in b:
                c.append(i + j)
        a = c;
    return a


def kronecker(A, B):
    #The calculation of the Kronecker product.
    n1 = A.shape
    n2 = B.shape
    N1 = n1[0]
    N2 = n2[0]
    N = N1 * N2
    product = np.identity(N)
    product = product.astype('complex')
    for p in range(N1):
        for q in range(N1):
            for i in range(N2):
                for j in range(N2):
                    product[p * N2 + i, q * N2 + j] = A[p, q] * B[i, j]
    return product


Sx = np.array([[0, 1], [1, 0]])
Sy = np.array([[0, -1.j], [1.j, 0]])
Sz = np.array([[1, 0], [0, -1]])
H = kronecker(Sx, Sx) + kronecker(Sy, Sy) + kronecker(Sz, Sz)
H = H.astype('int32')


def hamiltonian(L):
    # Calculate the exchange interaction terms in the Hamiltonian
    summary = np.zeros(2 ** L)
    summary = summary
    I = np.identity(2)

    for i in range(L - 1):
        list = []
        for j in range(L - 1):
            if j == i:
                list.append(H)
            else:
                list.append(I)
        product = kronecker(list[0], list[1])
        for p in range(L - 3):
            product = kronecker(product, list[p + 2])
        summary = summary + product # Periodic boundary conditions

    productx = kronecker(Sx, I)

    for i in range(L - 3):
        productx = kronecker(productx, I)
    productx = kronecker(productx, Sx)

    producty = kronecker(Sy, I)

    for i in range(L - 3):
        producty = kronecker(producty, I)
    producty = kronecker(producty, Sy)

    productz = kronecker(Sz, I)
    for i in range(L - 3):
        productz = kronecker(productz, I)
    productz = kronecker(productz, Sz)

    summary = summary + productx + producty + productz
    summary = np.real(summary)
    return summary


def commute(L, Sz, Hs):
    #Verify the commutation relations.
    summary0 = np.zeros(2 ** L)
    I = np.identity(2)
    for i in range(L):
        list = []
        for j in range(L):
            if j == i:
                list.append(Sz)
            else:
                list.append(I)
        product0 = kronecker(list[0], list[1])
        for p in range(L - 2):
            product0 = kronecker(product0, list[p + 2])
        summary0 = summary0 + product0
    Sz = summary0
    commute = Sz @ Hs - Hs @ Sz
    commute = commute.astype('float32')
    print("For a chain with length {}, the commutation of the Hamiltonian with spin operators is".format(L))
    print(commute)


for L in [4, 6, 8, 10, 12]:
    Hs = hamiltonian(L) / 4
    if L == 4:
        print("For a chain with length {}, the Hamiltonian of the system is ".format(L))
        print(Hs)

    #U(1) symmetry
    quantumNumber = qN(L)
    N = []
    for i in range(len(quantumNumber)):
        if quantumNumber[i] == 0:
            N.append(i)
    B = [];
    A = [];
    for j in range(len(N)):
        B.append(N)
        a = [N[j] for x in N]
        A.append(a)
    zeros = Hs[A, B]
    c0 = np.linalg.eigvals(zeros)
    c0.sort()
    c0 = np.real(c0)

    if L == 4:
        commute(L, Sz, Hs)

    if L == 4:
        print("For a chain with length{}，the matrix corresponding to a quantum number of 0 is".format(L))
        print(zeros)
        print("The eigenvalue is")
        print(c0)

    print("For a chain with length L = {}, the ground state energy is {}, the first excited state energy is {}, and the energy gap is {}.".
          format(L, c0[0], c0[1], c0[1] - c0[0]))
