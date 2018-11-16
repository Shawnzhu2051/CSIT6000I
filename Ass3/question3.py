import numpy as np
import matplotlib.pyplot as plt

def question_a(A):
    print('Question a:')
    A = np.array(A).transpose()
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    print('U:')
    print(u)
    print('S:')
    print(s)
    print('V_t:')
    print(vt)
    print('------------------')


def question_b(A,q):
    print('Question b:')
    A = np.array(A).transpose()
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    q = np.array(q)
    dot_prod = q @ A
    inner_prod = q @ u
    print('inner product:')
    print(inner_prod)
    print('dot product:')
    print(dot_prod)
    if (list(inner_prod) == list(dot_prod)):
        print('They are same!')
    else:
        print('They are not same!')
    print('------------------')


def question_c(A,k):
    print('Question c:')
    A = np.array(A).transpose()
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    A_approx = 0
    for i in range(k):
        A_approx += s[i] * (np.array([u.transpose()[i]]).transpose() @ np.array([vt[i]]))
    print('Rank 2 approximation is: ')
    print(A_approx)
    print('------------------')


def question_d(A,q):
    plt.figure(1)
    print('Question d:')
    A = np.array(A).transpose()
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    u2 = u.transpose()[-2:]
    v2 = vt[-2:]
    plt.plot(u2[0],u2[1],'b*', v2[0],v2[1],'r*')
    #plt.show()
    print('------------------')


def question_e(A,k,q):
    print('Question e:')
    A = np.array(A).transpose()
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    A_approx = 0
    for i in range(k):
        A_approx += s[i] * (np.array([u.transpose()[i]]).transpose() @ np.array([vt[i]]))
    inner_prod = q @ A_approx
    print('inner product:')
    print(inner_prod)
    print('------------------')

if __name__ == "__main__":
    A = [
        [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0, 0, 1, 1, 0, 2, 1],
        [1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1]
    ]
    q = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1]

    question_a(A)
    question_b(A,q)
    question_c(A,2)
    question_d(A,q)
    question_e(A,2,q)
