
from typing import Callable
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=150)
import matplotlib.pyplot as plt
import seaborn as sns

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
    for i in range(M):#1<<(6*precision)):

            # theta = np.random.rand(6)

            theta = np.zeros(4)
            # print(f'{i=:>0{precision*4}b}\n')
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

def main():
    np.random.seed(67)
    precision = 2
    M = 1<<(4*precision)
    gamma = 1#/np.log2(M)
    ker   = lambda x,y: moduloSquareKer(x,y)

    labels = np.hstack([np.zeros(1), np.ones(M), -np.ones(M)])
    # print(labels)

    data = make_synth_data(precision=precision)
    ent  = data[:len(data)//2]; unent = data[len(data)//2:]

    kerMat = ker_matrix(ent_data=ent, unent_data=unent, ker=ker)

    tMat = training_matrix(kerMat, gamma)#/(2*M*(1+1/gamma))
    
    result = np.linalg.inv(tMat).dot(labels)
    b = result[0]; alpha = result[1:]

    # print(kerMat)
    # print(tMat)
    # print(np.linalg.inv(tMat))
    # print(b,alpha)
    
    plt.plot(alpha)
    plt.show()

    num_trials = 1000
    classifications = np.zeros(num_trials)
    norm = np.sqrt(
        (b**2 + np.sum(alpha**2))
        *(M+1)
    )
    for i in range(num_trials):
        entState = randEntState(2)
        total = b

        if i < num_trials//2:
            #Ent
            state1 = np.kron(entState,entState)
            for j, state2 in enumerate(data):
                total += (alpha[j])*ker(state1, state2)

        else:
            #Unent - Need to account for impure states
            C1 = randEntState(1)
            C2 = randEntState(1)
            damagedState01 = np.kron(entState[::2], C1)
            damagedState11 = np.kron(entState[1::2], C1)
            damagedState02 = np.kron(entState[::2], C2)
            damagedState12 = np.kron(entState[1::2], C2)
            if np.random.rand() < 0.40:
                #Attack on first pair
                state01 = np.kron(damagedState01, entState)
                state11 = np.kron(damagedState11, entState)
                for j, state2 in enumerate(data):
                    total += (alpha[j])*ker(state01, state2)
                    total += (alpha[j])*ker(state11, state2)
            elif np.random.rand() < 0.66:
                #Attack on second pair
                state02 = np.kron(entState, damagedState02)
                state12 = np.kron(entState, damagedState12)
                for j, state2 in enumerate(data):
                    total += (alpha[j])*ker(state02, state2)
                    total += (alpha[j])*ker(state12, state2)
            else:
                #Attack on both pairs
                state01 = np.kron(damagedState01, damagedState02)
                state11 = np.kron(damagedState11, damagedState12)
                state02 = np.kron(damagedState01, damagedState02)
                state12 = np.kron(damagedState11, damagedState12)
                for j, state2 in enumerate(data):
                    total += (alpha[j])*ker(state01, state2)
                    total += (alpha[j])*ker(state11, state2)
                    total += (alpha[j])*ker(state02, state2)
                    total += (alpha[j])*ker(state12, state2)

        classifications[i] = (1 + total/norm)/2

    tent   = np.sum(classifications[:num_trials//2] > 0.5)
    fent =   num_trials//2-tent
    funent = np.sum(classifications[num_trials//2:] > 0.5)
    tunent = num_trials//2-funent

    print(f'Training set size = {2*M}')
    print(f'{tent  = :d}\t{fent  = :d}\n'
          f'{funent= :d}\t{tunent= :d}'
    )
    print(f'avg(|f|) = {np.average(np.abs(2*classifications-1))}')
    print(f'std(|f|) = {np.std(np.abs(2*classifications-1))}')
    plt.plot(classifications)
    plt.plot(np.ones(num_trials)/2, color='black')
    plt.show()

    sns.heatmap(
        np.array([[tent,fent],[funent,tunent]]), 
        annot=True, fmt='d', annot_kws={"size": 16}
    )
    plt.show()


if __name__=="__main__":
    main()
