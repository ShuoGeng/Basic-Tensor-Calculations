# Randomly generate nonzero vectors xi and eta in C^n, with ||eta|| = ||ksi||. 
# Calculate unitary matrices 𝑈 using Givens transformation and Householder transformation, respectively, such that Uksi = eta.

import numpy as np
import numpy.random as npr
from math import cos, sin, atan, acos

#生成二维向量的旋转矩阵
def rot(theta):
    # Generate the rotation matrix for a 2D vector.
    a = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    return a


def normal(xi, eta,n):
    # Convert a complex number vector into a complex diagonal matrix
    A1 = np.zeros([n, n])
    A1 = np.array(A1, dtype=complex)
    A2 = A1.copy()
    for i in range(n):
        A1[i][i] = np.linalg.norm(xi[i])/xi[i]
        A2[i][i] = np.linalg.norm(eta[i])/eta[i]
        A2[i][i] = 1/A2[i][i]
    return A1, A2


def Householder(xi,eta,n):
    # The matrix obtained from the Householder transformation.
    A = []
    [A1, A2] = normal(xi, eta,n)
    A.append(A1)

    xi2 = np.dot(A1,xi)
    xi2 = np.real(xi2)
    eta2 = abs(eta)

    for r in range(n):
        domain = list(range(n-1))
        for i in domain:
            if abs(xi2[i]) - abs(eta2[i]) > 0.00001:
                m1 = domain[i]
                del (domain[i])
                for k in domain:
                    if abs(abs(xi2[k]) - abs(eta2[k])) < 0.00001:
                        continue
                    else:
                        m2 = k
                        break

                vec1 = np.array([xi2[m1], xi2[m2]])
                l1 = np.linalg.norm(vec1)
                vec2 = vec1.copy()
                vec2[0] = eta2[m1]
                vec2[1] = (l1**2-vec2[0]**2)**0.5
                vecx = vec1 - vec2
                omega = np.zeros([n,1])
                omega[m1] = vecx[0]
                omega[m2] = vecx[1]
                omega = omega/np.linalg.norm(omega)
                H = np.identity(n) - 2*np.dot(omega,omega.T)
                xi2 = np.dot(H,xi2)
                A.append(H)
                break
    A.append(A2)

    U = np.identity(n)
    for w in A[::-1]:
        U = np.dot(U,w)

    return U


def Givens(xi, eta, n):
    # The matrix obtained from the Givens transformation.
    A = []
    [A1, A2] = normal(xi, eta,n)
    A.append(A1)

    xi2 = np.dot(A1,xi)
    xi2 = np.real(xi2)
    eta2 = abs(eta)

    for r in range(n-1):
        domain = list(range(n))
        for i in domain:
            if abs(xi2[i]) - abs(eta2[i]) > 0.00001:
                m1 = domain[i]
                del (domain[i])
                for k in domain:
                    if abs(abs(xi2[k]) - abs(eta2[k])) < 0.00001:
                        continue
                    else:
                        m2 = k
                        break

                vec = np.array([xi2[m1], xi2[m2]])
                theta_1 = atan(xi2[m2] / xi2[m1])
                theta_2 = acos(eta2[m1] / np.linalg.norm(vec))
                theta0 = theta_1 - theta_2
                A1x = rot(theta0)

                Ax = np.identity(n)
                Ax[m1][m1] = A1x[0][0]
                Ax[m1][m2] = A1x[0][1]
                Ax[m2][m1] = A1x[1][0]
                Ax[m2][m2] = A1x[1][1]

                A.append(Ax)

                xi2 = np.dot(Ax, xi2)
                break
    A.append(A2)
    U = np.identity(n)
    for w in A[::-1]:
        U = np.dot(U,w)
    return U


# Randomly generate three sets of vectors, and loop the program three times.
for cycle in range(3):
    # Enter the dimension of the vectors.
    n = input('input the dimension of the vector：')
    n = int(n)

    # Randomly generate nonzero vectors.
    eta_real = npr.uniform(-1, 1, size=(n, 1))
    eta_imag = npr.uniform(-1, 1, size=(n, 1))
    eta = eta_real + eta_imag * 1.j
    xi_real = npr.uniform(-1, 1, size=(n, 1))
    xi_imag = npr.uniform(-1, 1, size=(n, 1))
    xi = xi_real + xi_imag * 1.j
    length = 1
    eta = eta / np.linalg.norm(eta)
    xi = xi / np.linalg.norm(xi)
    eta = eta*length
    xi = xi*length

    # Output the randomly generated vectors.
    print('xi:')
    print(xi)
    print('eta')
    print(eta)


    # Call the Givens function and satisfy the condition of passing nonzero vectors as function parameters. 
    # Use the function to achieve the goal, and the output of the function will be the target matrix.
    U = Givens(xi, eta, n)
    print("The matrix U obtained through the Givens transformation is: ")
    print(U)

    # Verify that the matrix U transforms the xi vectors into the eta vectors.
    xi_test = xi.copy()
    xi_test = np.dot(U, xi_test)
    print('')
    print('After applying the matrix U, the vectors xi will be:')
    print(xi_test)

    # Take the conjugate transpose of U and verify if it is a unitary matrix with a precision of 1e-8.
    U_dag = (U.T).conjugate()
    output = np.round(np.dot(U,U_dag),8)
    print()
    print('The result of multiplying U and the conjugate transpose of U (U†) is')
    print(output)

    print('')
    print('now Householder')

    # Call the Householder function and satisfy the condition of passing nonzero vectors as function parameters. 
    # Use the function to achieve the goal, and the output of the function will be the target matrix.
    U  = Householder(xi, eta, n)
    print('The matrix U obtained through the Householder transformation is: ')
    print(U)

    # Verify that the matrix U transforms the xi vectors into the eta vectors.
    xi_test = xi.copy()
    xi_test = np.dot(U, xi_test)
    print('U*xi')
    print(xi_test)

    # Take the conjugate transpose of U and verify if it is a unitary matrix with a precision of 1e-8.
    U_dag = (U.T).conjugate()
    output = np.round(np.dot(U, U_dag), 8)
    print()
    print('The matrix U obtained through the Householder transformation is: ')
    print(output)

    print('')
