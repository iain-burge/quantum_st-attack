from typing import Callable
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=150)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv

def moduloSquareKer(x:np.ndarray[2], y:np.ndarray[2]) -> float:
    return ((x.conj().dot(y)) * (y.conj().dot(x))).real

def randEntState(n: int) -> np.ndarray:
    smag = np.random.random(1<<n); sang = np.random.random(1<<n)
    smag = smag / np.linalg.norm(smag)
    state = smag * np.exp(2j*np.pi*sang)
    return state

def Id() -> np.ndarray[(2,2)]:
    return np.identity(2)

def Ry(theta: float) -> np.ndarray[(2,2)]:
    return np.array([
        [np.cos(np.pi*theta), -np.sin(np.pi*theta)],
        [np.sin(np.pi*theta),  np.cos(np.pi*theta)]
    ])

def Rx(theta: float) -> np.ndarray[(2,2)]:
    return np.array([
        [np.cos(np.pi*theta), -1j*np.sin(np.pi*theta)],
        [-1j*np.sin(np.pi*theta), np.cos(np.pi*theta)]
    ])

def Rz(theta: float) -> np.ndarray[(2,2)]:
    return np.array([
        [np.exp(-1j*theta), 0],
        [0, np.exp( 1j*theta)]
    ])

def ent_state(theta: list[float]) -> np.ndarray[4]:
    vec = np.zeros(4,complex); vec[0]=1

    vec[0] = np.cos(np.pi*theta[0])
    vec[2] = -1j * np.sin(np.pi*theta[0]) * np.cos(np.pi*theta[1])
    vec[3] = -1j * np.sin(np.pi*theta[0]) * np.sin(np.pi*theta[1])

    vec = np.kron(Ry(theta[2]), Id()).dot(vec)
    vec = np.kron(Id(), Rx(theta[3])).dot(vec)
    # vec = np.kron(Rz(theta[4]), Id()).dot(vec)
    # vec = np.kron(Id(), Rz(theta[5])).dot(vec)

    return vec

def unent_state(theta:list[float]) -> np.ndarray[4]:
    vec = np.zeros(4,complex); vec[0]=1

    vec[0] = np.cos(np.pi*theta[0]) * np.cos(np.pi*theta[1])
    vec[1] = np.cos(np.pi*theta[0]) * np.sin(np.pi*theta[1])
    vec[2] = -1j * np.sin(np.pi*theta[0]) * np.cos(np.pi*theta[1])
    vec[3] = -1j * np.sin(np.pi*theta[0]) * np.sin(np.pi*theta[1])

    vec = np.kron(Ry(theta[2]), Id()).dot(vec)
    vec = np.kron(Id(), Rx(theta[3])).dot(vec)
    # vec = np.kron(Rz(theta[4]), Id()).dot(vec)
    # vec = np.kron(Id(), Rz(theta[5])).dot(vec)

    return vec

def unentA(theta:list[float]) -> np.ndarray:
    vec = np.zeros(2,complex); vec[0]=1

    vec = Rx(theta[0]).dot(vec)
    vec = Ry(theta[2]).dot(vec)
    # vec = Rz(theta[4]).dot(vec)

    return vec

def unentB(theta:list[float]) -> np.ndarray:
    vec = np.zeros(2,complex); vec[0]=1

    vec = Ry(theta[1]).dot(vec)
    vec = Rx(theta[3]).dot(vec)
    # vec = Rz(theta[5]).dot(vec)

    return vec


def make_data(M: int) -> tuple[list,list]:
    #M number of datapoints of each kind
    ent_data, unent_data = [], []
    for _ in range(M):

        #Ent
        ent_state = randEntState(2)
        ent_data.append(np.kron(ent_state,ent_state))

        #Unent
        A  = randEntState(1)
        C1 = randEntState(1)
        C2 = randEntState(1)
        unent_data.append(np.kron(np.kron(A,C1),np.kron(A,C2)))


    return ent_data, unent_data

def make_synth_data(precision: int) -> tuple[list,list]:
    ent_data, unent_data = [], []
    mask = (1<<precision)-1

    M = 1<<(4*precision)
    for i in range(M):
            theta = np.zeros(4)
            for var in range(4):
                theta[var]  = (i & (mask << (var*precision))) >> (var*precision)
                theta[var] /= (1<<precision)

            entState = ent_state(theta)

            #Ent
            ent_data.append(np.kron(entState,entState))

            #Unent
            A  = unentA(theta)
            C1 = unentB(theta)
            C2 = unentB([
                t*[(2*19)**2,(3*17)**2,(5*13)**2,(7*11)**2][i]
                for i,t in enumerate(theta)
            ])

            if i%5 in [0,1]:
                unent_data.append(np.kron(entState,np.kron(A,C2)))
            elif i%5 in [2,3]:
                unent_data.append(np.kron(np.kron(A,C1),entState))
            else:
                unent_data.append(np.kron(np.kron(A,C1),np.kron(A,C2)))

    return ent_data + unent_data


def ker_matrix(
        ent_data: list, unent_data: list,
        ker: Callable[[np.ndarray, np.ndarray], float]
    ) -> np.ndarray:
    data = ent_data+unent_data
    mat  = np.zeros((len(data), len(data)))
    for i, x in enumerate(data):
        for j, y in enumerate(data):
            mat[i,j] = ker(x,y)

    return mat

def training_matrix(
        ker_mat: np.ndarray, gamma: float
    ) -> np.ndarray:
    M     = ker_mat.shape[0]
    ones  = np.ones((M,1))
    onesP = np.vstack([np.zeros((1,1)),ones]).T

    training_matrix = ker_mat + (1/gamma)*np.identity(M)
    training_matrix = np.hstack([ones, training_matrix])
    training_matrix = np.vstack([onesP, training_matrix])

    return training_matrix

def csv2Dataset(filename: csv) -> list[np.ndarray]:
    states    = []
    inputdata = []

    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            if ('REAL' in row) or ('IMAGINARY' in row) or (len(row)==0): continue
            inputdata.append(np.array([float(el) for el in row]))
    # print(inputdata[0])

    for k in range(len(inputdata)//8):
        density  =    np.zeros((4,4), complex)
        density +=    np.vstack([inputdata[8*k+j  ] for j in range(4)])
        density += 1j*np.vstack([inputdata[8*k+4+j] for j in range(4)])
        states.append(density)

    return states[::2] #Taking only every other datapoint because of redundant data

def angle( mat: np.ndarray) -> np.ndarray:
    mat = np.angle(mat)
    mat /= 2*np.pi
    return mat


def show_mat(mat: np.ndarray, title: str='', threshold: float=None) -> None:
    fig, ax = plt.subplots(2,2)
    plt.suptitle(title)

    im = []
    mats   = [mat.real, mat.imag, np.abs(mat), angle(mat)]
    widths = 3*[2*[np.max(np.abs(mat))]] + [[0,1]]
    titles = ['real', 'imag', 'abs', 'angle']

    for k in range(4):
        im.append(ax[k//2,k%2].imshow(
            mats[k], vmin=-widths[k][0], vmax=widths[k][1]
        ))
        ax[k//2,k%2].set_title(titles[k])
        # ax[k%2,k//2].set_title(title)
        fig.colorbar(im[k])

    plt.show()

def swap_gate(n: int):
    #swaps an n bit register with the next
    N = 1<<n
    swap = np.zeros((N**2,N**2))
    for k in range(N**2):
        x_k = ((N-1)&(k>>n)) + (((N-1)&k)<<n)
        # print(f'{k =:0{2*n}b}\t->\t{x_k =:0{2*n}b}')
        swap[k, x_k] = 1
    return swap

def damage_state(AB=None,C=None) -> list[np.ndarray]:
    if AB is None: AB  = randEntState(2)
    if C  is None: C   = randEntState(1)

    ABC = np.kron(AB,C)
    ACB = np.kron(np.identity(2), swap_gate(n=1)).dot(ABC)

    #Density matrix with partial trace removing B (rightmost register)
    rho_AC = np.zeros((4,4),complex)
    for k in range(8):
        for s in range(8):
            if k%2==s%2:
                rho_AC[k//2, s//2] += ACB[k].conj()*ACB[s]
    # show_mat(np.outer(ACB,ACB))
    # show_mat(rho_AC)
    # print(rho_AC)
    return rho_AC



def main():
    #Note to avoid extremely large density matrices, we are making a
    #simplification on the swap test we swap the state directly with the
    #hyperplane normal vector and then apply the normalization after.
    #Additionally, we deal with the bias term b seperately.
    #This results in an equivalent output.
    np.random.seed(67)

    #Useful matrices
    n    = 4
    N    = 1<<n
    HxI  = np.kron(np.array([[1,1],[1,-1]]), np.identity(N**2)) / np.sqrt(2)
    Proj = np.kron(np.array([[1,0],[0, 0]]), np.identity(N**2))
    swap = np.zeros((N**2,N**2))
    for k in range(N**2):
        x_k = ((N-1)&(k>>n)) + (((N-1)&k)<<n)
        # print(f'{k =:0{2*n}b}\t->\t{x_k =:0{2*n}b}')
        swap[k, x_k] = 1
    cswap = Proj + np.kron(np.array([[0,0],[0,1]]), swap)

    #expected and malicious data
    exp_data   = csv2Dataset('quisp-exported-datasets/groundtruth-teleported-matrices.csv')
    mal_data   = csv2Dataset('quisp-exported-datasets/attacked-teleported-matrices.csv')

    #states being assessed are exp/mal, mal/exp, or exp/exp
    class_data = []
    for k in range(len(exp_data)):
        class_data.append(np.kron(exp_data[k], exp_data[k]))
    for k in range(len(exp_data)):
        if   k%5 in [0,1]: class_data.append(np.kron(exp_data[k], mal_data[k]))
        elif k%5 in [2,3]: class_data.append(np.kron(mal_data[k], exp_data[k]))
        else:              class_data.append(np.kron(mal_data[k], mal_data[k]))

    num_trials = len(class_data)

    #Constructing QSVM alpha vector
    precision = 2
    M = 1<<(4*precision)
    gamma = 1
    ker   = lambda x,y: moduloSquareKer(x,y)

    labels = np.hstack([np.zeros(1), np.ones(M), -np.ones(M)])
    data = make_synth_data(precision=precision)
    ent  = data[:len(data)//2]; unent = data[len(data)//2:]
    kerMat = ker_matrix(ent_data=ent, unent_data=unent, ker=ker)
    tMat = training_matrix(kerMat, gamma)#/(2*M*(1+1/gamma))
    result = np.linalg.inv(tMat).dot(labels)
    b = result[0]; alpha = result[1:]

    classifications = np.zeros(num_trials)
    norm = np.sqrt((b**2 + np.sum(alpha**2))*(M+1))

    data_densities = []
    for d in data:
        data_densities.append(np.outer(d,d))

    for i in range(num_trials):
        print(f'\t{100*(i/num_trials):.2f}%', end='\r', flush=True)
        AB = randEntState(2)
        C  = randEntState(1)

        if i<num_trials//2: test  = np.kron(damage_state(AB,C), np.outer(AB.conj(),AB))
        else: test = np.kron(np.outer(AB.conj(),AB), np.outer(AB.conj(),AB))
        total = b

        #Note, this step is slightly modified for easier computation, the
        #results are equivalent
        for j, mu in enumerate(data):
            total += alpha[j]*(mu.conj()).dot(class_data[i]).dot(mu).real
        classifications[i] = (1 + total/norm)/2
    print(f'\t100.00%', end='\r')


    tent   = np.sum(classifications[:num_trials//2] > 0.5)
    fent   = num_trials//2-tent
    funent = np.sum(classifications[num_trials//2:] > 0.5)
    tunent = num_trials//2-funent

    print(f'Training set size = {2*M}')
    print(f'{tent  = :d}\t{funent  = :d}\n'
          f'{fent  = :d}\t{tunent= :d}'
    )
    print(f'avg(|f|) = {np.average(np.abs(2*classifications-1))}')
    print(f'std(|f|) = {np.std(np.abs(2*classifications-1))}')
    plt.plot(classifications)
    plt.plot(np.ones(num_trials)/2, color='black')
    plt.show()

    sns.heatmap(
        np.array([[tent,funent],[fent,tunent]]),
        annot=True, fmt='d', annot_kws={"size": 16},
        xticklabels=['Legitimate','Malicious'], yticklabels=['Legitimate','Malicious'],
        cbar=False
    )
    plt.xlabel('QSVM Monitor')
    plt.ylabel('Ground Truth')
    plt.savefig('quispConfusionMatrix.pdf')
    plt.show()


if __name__=="__main__":
    main()

