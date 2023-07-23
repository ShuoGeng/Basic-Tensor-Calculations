# Calculate the specific heat (Cv) of the spin-1/2 antiferromagnetic Heisenberg chain with length L=8.
# The partition function Z is given by: Z = tr(e^(-βH)),
# where β = 1/T is the inverse temperature and H is the Hamiltonian.
# The specific heat Cv = 1/L*d<H>/dT, where <H> = tr(He^(-beta*H))/Z
# This file will use U(1) symmetry
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def qN(L):
    #Caculate quantum number 
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
        summary = summary + product

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

def HT(H,T):
    # The caculatoin of e^(-beta*T)
    zeros = -H/T
    w,v = np.linalg.eig(zeros)
    # print(w)
    vdagger = np.linalg.inv(v)
    [m,n]= np.shape(zeros)
    eH = np.zeros([m,n])
    gamma = np.zeros([m,n])

    for i in range(m):
        gamma[i,i] = np.exp(w[i])
    eH = v@gamma@vdagger
    Z = trace(eH)
    Hbar = trace(H@eH)/Z
    return Hbar

def trace(M):
    # The caculation of trace
    [m,n] = np.shape(M)
    output = 0
    for i in range(m):
        output = output+M[i,i]
    return output


L = 8
Hs = hamiltonian(L) / 4
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
print("with the use of U(1）symmetry")
print("For a chain with length{}，the matrix corresponding to a quantum number of 0 is".format(L))
print(zeros)

# Caculate Cv
Hbar = np.zeros([100,1])
T = np.linspace(0.0001,2,100)
dT = T[2]-T[1]
for i in range(100):
    Hbar[i,0] = HT(zeros,T[i])
dHbar = np.zeros([99,1])
for i in range(99):
    dHbar[i,0] = Hbar[i+1]-Hbar[i]
dHbar = dHbar/8/dT
plt.figure()
plt.plot(T[0:99],dHbar,linewidth=1,linestyle='-')
plt.xlabel('T')
plt.ylabel('Cv')
plt.title('T-Cv curve, where the values of T range from 0 to 2.')


Hbar = np.zeros([1000,1])
T = np.linspace(0.0001,20,1000)
dT = T[2]-T[1]
for i in range(1000):
    Hbar[i,0] = HT(zeros,T[i])
dHbar = np.zeros([999,1])
for i in range(999):
    dHbar[i,0] = Hbar[i+1]-Hbar[i]
dHbar = dHbar/8/dT
plt.figure()
plt.plot(T[0:999],dHbar,linewidth=1,linestyle='-')
plt.xlabel('T')
plt.ylabel('Cv')
plt.title('T-Cv curve, where the values of T range from 0 to 20.')

max_Cv = max(dHbar[2:999])
# print(max_Cv)
# print(dHbar)
for i in range(1000):
    if dHbar[i] == max_Cv[0]:
        p = i
        break
print('The temperature corresponding to the maximum specific heat is {0},and the maximum specific heat is {1}.'.format(T[i],dHbar[i]))
print('The variation of specific heat with temperature is:')
plt.show()

