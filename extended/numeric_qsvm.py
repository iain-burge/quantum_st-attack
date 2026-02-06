
from typing import Callable
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=150)
import matplotlib.pyplot as plt
import seaborn as sns

# import qiskit

def linearKer(x: list[float], y: list[float]):
    x = np.array(x).conj(); y = np.array(y)
    # return np.exp((np.abs(x.dot(y))+1/10))
    return np.abs(x.dot(y))

def unentState(theta: list[float]) -> np.ndarray:
    vector    = np.zeros((4), dtype=complex)
    vector[0] = 1
    
    vector[0] = np.cos(np.pi*theta[0]) * np.cos(np.pi*theta[1])
    vector[1] = np.cos(np.pi*theta[0]) * np.sin(np.pi*theta[1])
    vector[2] = -1j * np.sin(np.pi*theta[0]) * np.cos(np.pi*theta[1])
    vector[3] = -1j * np.sin(np.pi*theta[0]) * np.sin(np.pi*theta[1])

    mat = np.matmul(
        np.kron( #Ry gate top qubit
            np.array([[np.cos(np.pi*theta[2]), -np.sin(np.pi*theta[2])],
                    [np.sin(np.pi*theta[2]),  np.cos(np.pi*theta[2])]], 
                    dtype=complex),
            np.array([[1,0],
                    [0,1]],dtype=complex),
        ), 
        np.kron( #Rx gate bottom qubit
            np.array([[1,0],
                    [0,1]]),
            np.array([[np.cos(np.pi*theta[3]), -1j*np.sin(np.pi*theta[3])],
                    [-1j*np.sin(np.pi*theta[3]), np.cos(np.pi*theta[3])]]),
        )
    )

    return mat.dot(vector)

def entState(theta: list[float]) -> np.ndarray:
    vector    = np.zeros((4), dtype=complex)
    
    vector[0] = np.cos(np.pi*theta[0])
    vector[2] = -1j * np.sin(np.pi*theta[0]) * np.cos(np.pi*theta[1])
    vector[3] = -1j * np.sin(np.pi*theta[0]) * np.sin(np.pi*theta[1])

    mat = np.matmul(
        np.kron( #Ry gate top qubit
            np.array([[np.cos(np.pi*theta[2]), -np.sin(np.pi*theta[2])],
                    [np.sin(np.pi*theta[2]),  np.cos(np.pi*theta[2])]], 
                    dtype=complex),
            np.array([[1,0],
                    [0,1]],dtype=complex),
        ), 
        np.kron( #Rx gate bottom qubit
            np.array([[1,0],
                    [0,1]]),
            np.array([[np.cos(np.pi*theta[3]), -1j*np.sin(np.pi*theta[3])],
                    [-1j*np.sin(np.pi*theta[3]), np.cos(np.pi*theta[3])]]),
        )
    )

    return mat.dot(vector)

def randUnentState(n: int) -> np.ndarray:
    q1 = randEntState(n-1); q2 = randEntState(n-1)    
    return np.kron(q1,q2)
    
def randEntState(n: int) -> np.ndarray:
    smag = np.random.random(1<<n); sang = np.random.random(1<<n)
    smag = smag / np.linalg.norm(smag)
    state = smag * np.exp(2j*np.pi*sang)
    return state

def dataSet(precision: int):
    mask = (1<<precision)-1

    unent = []
    ent   = []

    for i in range(1<<(4*precision)):
        theta = [0,0,0,0]
        # print(f'{i=:>0{precision*4}b}\n')
        for var in range(4):
            theta[var] = (i & (mask << (var*precision))) >> (var*precision)
            theta[var] /= (1<<precision)

        unent.append(unentState(theta))
        ent.append(entState(theta))

    print(f'{len(unent)=}, {len(ent)}')
    return unent + ent

def kernelMatrix(ker: Callable[[np.ndarray, np.ndarray], float], 
        data: list[np.ndarray]
    ) -> np.ndarray:
    kMat = np.zeros(shape=(len(data),len(data)),dtype=float)

    for i,x in enumerate(data):
        for j,y in enumerate(data):
            kMat[i,j] = ker(x,y)

    return kMat

def labelVector(precision: int) -> np.ndarray:
    dim = 1<<(4*precision)

    bias   =  np.zeros((1,1))
    top    =  np.ones((dim,1))
    bottom = -np.ones((dim,1))

    return np.vstack([bias, top, bottom])

def biasMatrix(matrix: np.ndarray, gamma: float) -> np.ndarray:
    dim = matrix.shape[0]

    left = np.ones((dim,1))
    top  = np.ones((1,dim+1)); top[0,0] = 0

    # mat = (1-1/gamma)*matrix + (1/gamma) * np.identity(dim)
    mat = matrix + (1/gamma) * np.identity(dim)
    mat = np.hstack([left, mat])
    mat = np.vstack([top,  mat])

    # print('hello', np.max(np.abs(np.linalg.eigvals(mat))))
    # norm = np.abs(np.linalg.eigvals(mat)); norm = np.min(norm[np.nonzero(norm)])
    return mat

def classify(alpha: list[float], b: float, state: np.ndarray,
        data: list[np.ndarray], ker: Callable[[np.ndarray, np.ndarray], float]
    ) -> float:
    # result = b/2 #temp this should be b I think
    result = 0 #temp
    for i, a in enumerate(alpha):
        result += a * ker(data[i], state)
    
    return result

def testPlane(alpha: list[float], b: float, data: list[np.ndarray], 
        ker: Callable[[np.ndarray, np.ndarray], float]
    ):
    trials = 1024
    num_measurements = 3
    entRegression   = np.zeros(trials)
    unentRegression = np.zeros(trials)
    for trial in range(trials):
        # theta = np.random.rand(4)
        entRegression[trial]   = classify(alpha,b,randEntState(2),data,ker)[0]
        unentRegression[trial] = classify(alpha,b,randUnentState(2),data,ker)[0]
        # print(theta)
        # print(f'\t{entResults  =}')
        # print(f'\t{unentResults=}')
    
    entMeasurement   = np.zeros(trials)
    unentMeasurement = np.zeros(trials)
    for _ in range(num_measurements):
        entMeasurement   += 1*((entRegression+1)/2   < np.random.random(trials))
        unentMeasurement += 1*((unentRegression+1)/2 < np.random.random(trials))

    entMeasurement   = entMeasurement > num_measurements//2
    unentMeasurement = entMeasurement > num_measurements//2
    tp = entMeasurement.sum();   fp = trials - tp
    fn = unentMeasurement.sum(); tn = trials - fn
    
    # plt.scatter([i for i in range(100)], entResults, label='entangled')
    # plt.scatter([i for i in range(100)], unentResults, label='unentangled')
    
    sns.set_theme(font_scale=1.4) # for label size
    sns.kdeplot((entRegression+1)/2, label='Entangled', 
                fill=True, alpha=.6, linewidth=0)
    sns.kdeplot((unentRegression+1)/2, label='Unentangled', 
                fill=True, alpha=.6, linewidth=0)
    # plt.boxplot([entResults,unentResults])
    # plt.xticks([1,2],['Entangled', 'Unentangled'])
    plt.title('QSVM Entanglement Detection Regression')
    plt.xlabel('Regression Output')
    plt.legend()
    plt.show()

    print(f'tp={tp}\tfp={fp}\nfn={fn}\ttn={tn}')
    sns.heatmap(
        np.array([[tp,fp],[fn,tn]]), 
        annot=True, fmt='d', annot_kws={"size": 16}
    )
    plt.show()

def main():
    # entState([0,0,.5,0.5])
    # print(ker([1,1], [1,-1j]))
    # print(dataSet(precision=1))
    precision = 2
    gamma     = 10
    ker       = linearKer

    # print(randEntState(2))
    # print(randUnentState(2))

    data   = dataSet(precision=precision)
    dMat   = np.vstack(data).transpose()
    # dMat = np.random.rand(257,512) + 1j*np.random.rand(257,512)
    print(dMat)
    kerMat = np.matmul(dMat.transpose().conjugate(), dMat)
    # print(kerMat.shape)
    # kerMat = kernelMatrix(ker, data)
    fMat   = np.abs(biasMatrix(kerMat, gamma))
    # fMat   = np.random.rand(fMat.shape[0], fMat.shape[1]) #Temp
    # fMat   = np.random.rand(5, 5); fMat += fMat.transpose() #Temp
    invF   = np.linalg.inv(fMat)

    lvec   = labelVector(precision=precision)
    # plt.plot(lvec)
    result = invF.dot(lvec)
    alpha  = result[1:]
    b      = result[0,0]
    print(f'{result=}')

    testPlane(alpha,b,data,ker)

    # plt.plot(result)
    # print(b)
    # print(alpha)

    # # eigenvalues, eigenvectors = np.linalg.eigh(fMat)
    # eigenvalues, eigenvectors = np.linalg.eigh(kerMat)
    # plt.plot(eigenvalues, label='evals')
    # plt.plot(0*eigenvalues)


    # print(fMat)
    # print(labelVector(precision=precision).shape)

    # fig, ax = plt.subplots(ncols=1)
    # im0 = ax.imshow(fMat)
    # # im1 = ax[1].imshow(kerMat)
    # fig.colorbar(im0)
    # # fig.colorbar(im1)

    # test = 0
    # eigenvalues, eigenvectors = np.linalg.eigh(fMat)
    # print(eigenvalues[test])
    # print(fMat)
    # print(eigenvectors[:,test])
    # plt.plot(eigenvectors[:,test], label='evec')
    # plt.plot(eigenvectors[:,test]*eigenvalues[test], label='eval*evec')
    # plt.plot(fMat.dot(eigenvectors[:,test]), label='F*evec')
    # plt.plot(0*(eigenvectors[:,test]), color='black')
    # plt.legend()

    # print(np.linalg.eig(fMat)[1][1])
    # # plt.plot(np.linalg.eig(fMat)[0])
    # plt.plot(np.linalg.eig(fMat)[1][0])
    # plt.plot((np.linalg.eig(fMat)[1][0]).dot(fMat))

    # plt.show()

if __name__ == "__main__":
    main()
