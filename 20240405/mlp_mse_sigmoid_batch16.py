import numpy as np
import matplotlib.pyplot as plt
import time

# 시그모이드 함수 ------------------------
def Sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

def MLP_forward(U1, U2, P, C, x):
    N, D = x.shape #입력 차원
    zsum = np.zeros((N,P+1))
    z = np.zeros((N,P+1))
    osum = np.zeros((N,C))
    o = np.zeros((N,C))
    
    for n in range(N):
        #은닉층의 계산
        zsum[n,0] = 1.0
        z[n,0] = 1.0
        for j in range(P):
            zsum[n, j+1] = np.dot(U1[j],np.r_[1,x[n]])
            z[n,j+1] = Sigmoid(zsum[n, j+1])
        #출력층 계산
        for k in range(C):
            osum[n, k] = np.dot(U2[k],z[n])
            o[n,k] = Sigmoid(osum[n, k])
    return o, osum, z, zsum

def MLP_backward(U1, U2, P, C, x, y):
    N, D = x.shape #입력 차원
    dU1 = np.zeros_like(U1)
    dU2 = np.zeros_like(U2)
    delta = np.zeros(C)
    eta = np.zeros(P)    
    
    o, _, z, _ = MLP_forward(U1,U2,P,C,x)
    #x0 = np.ones((N,1))
    #x = np.concatenate([x0,x], axis=1)
    for n in range(N):
        # 출력층 node의 입력에서 에러값 delta 계산
        for k in range(C):
            delta[k] = (y[n,k]-o[n,k])*o[n,k]*(1-o[n,k])
        
        # 은닉층 node의 입력에서 에러값 eta 계산
        sum_err = np.zeros_like(eta)
        #t_eta[0] = 0.0
        for j in range(P):
            for k in range(C):
                sum_err[j] = sum_err[j] + U2[k,j+1]*delta[k] #은닉층 j번째 node의 출력에 유입되는 에러값을 계산
            eta[j] = z[n,j+1]*(1-z[n,j+1])*sum_err[j] #은닉층 j번째 node의 입력의 에러값 계산
        
        # 출력 node와 은닉층 node를 연결하는 edge의 weight 미분값을 계산       
        for k in range(C):
            for j in range(P+1):
                dU2[k,j] = dU2[k,j] - z[n,j]*delta[k]/N
                
        # 은닉층 node와 입력층 node를 연결하는 edge의 weight 미분값을 계산
        x_ = np.r_[1,x[n]]
        for j in range(P):
            for i in range(D+1):
                dU1[j,i] = dU1[j,i] - x_[i]*eta[j]/N

    return dU1, dU2    

def mse_cal(predict, Y):
    return np.square(Y.reshape(-1)- predict.reshape(-1)).mean() 

def accuracy_cal(predict, Y):
    return np.mean(predict.argmax(axis=1) == Y.argmax(axis=1))

def trainingMLP(epoch, U1, U2, P, C, X_train, Y_train, X_test, Y_test, rho):
    error_train = []
    error_test = []

    for e in range(epoch):
        startTime = time.time()

        print(f"The number of Epoch:{e:04d}\n")
        
        np.random.shuffle(sIdx)    #traing dataset X_train와 Y_train를 같은 순번으로 shuffing시킴
        X_train = X_train[sIdx]
        Y_train = Y_train[sIdx]
        
        for n in range(batch_num):
            n *= batch_size
            if(n < (batch_num-1)*batch_size):
                dU1, dU2 = MLP_backward(U1, U2, P, C, X_train[n:n+batch_size], Y_train[n:n+batch_size])
            else:
                dU1, dU2 = MLP_backward(U1, U2, P, C, X_train[n:], Y_train[n:])
            
            # U2 행렬을 업데이트               
            #for k in range(C):
            #    for j in range(P+1):
            #        U2[k,j] = U2[k,j] - rho*dU2[k,j]
            U2 = U2 -rho*dU2
            #U1 행렬을 업데이트
            #for j in range(P):
            #    for i in range(D+1):
            #        U1[j, i] = U1[j,i] - rho*dU1[j,i]
            U1 = U1 -rho*dU1                
        
        output_train, _, _, _ = MLP_forward(U1, U2, P, C, X_train)
        output_test, _, _, _ = MLP_forward(U1, U2, P, C, X_test)

        e_train = mse_cal(output_train, Y_train)
        e_test = mse_cal(output_test, Y_test)
    
        accuracy_train = accuracy_cal(output_train, Y_train)
        accuracy_test = accuracy_cal(output_test, Y_test)

        error_train.append(e_train)
        error_test.append(e_test)

        endTime = time.time()

        print(f"Training MSE:{e_train:0.4f}, Training Accuracy:{accuracy_train:0.4f}\n")
        print(f"Validation MSE:{e_test:0.4f}, Validation Accuracy:{accuracy_test:0.4f}\n")
        print(f"Updated U1 Matrix in learning process:\n {U1}\n")
        print(f"Updated U1 Matrix in learning process:\n {U2}\n")
        print(f"Elapsed time:{endTime-startTime:0.3f} sec\n")
        print("###########################\n")
    
    return error_train, error_test, U1, U2

np.random.seed(seed=1) # 난수를 고정
N = 1000 # 데이터의 수
K = 3 # 분포의 수
Y = np.zeros((N, K), dtype=np.uint8)
X = np.zeros((N, 2))
X_range0 = [-3, 3] # X0의 범위, 표시용
X_range1 = [-3, 3] # X1의 범위, 표시용
Mu = np.array([[-2.0, -2.0], [1.5, 2.0], [2, -1.5]]) # 분포의 중심
Sig = np.array([[0.9, 0.9], [0.8, 0.7], [0.9, 1.0]]) # 분포의 분산
Pi = np.array([0.35, 0.7, 1.0]) # 각 분포에 대한 비율
for n in range(N):
    wk = np.random.rand()
    for k in range(K):
        if wk < Pi[k]:
            Y[n, k] = 1
            break
    for k in range(2):
        X[n, k] = np.random.randn() * Sig[Y[n, :] == 1, k] + \
        Mu[Y[n, :] == 1, k]


# -------- 2 분류 데이터를 테스트 훈련 데이터로 분할
YestRatio = 0.5
X_n_training = int(N * YestRatio)
ValidRatio = 0.2
X_n_valid = X_n_training + int(N * ValidRatio)
X_train = X[:X_n_training]
X_valid = X[X_n_training:X_n_valid]
X_test = X[X_n_valid:]
Y_train = Y[:X_n_training]
Y_test = Y[X_n_valid:]
Y_valid = Y[X_n_training:X_n_valid]


# -------- 데이터를 'class_data.npz'에 저장
np.savez('class_data.npz', X_train=X_train, Y_train=Y_train,
         X_test=X_test, Y_test=Y_test,
         X_range0=X_range0, X_range1=X_range1)


def Show_MLP_Contour(U1, U2, P, C):
    xn = 60 #등고선 표시 해상도
    x0 = np.linspace(X_range0[0],X_range0[1], xn)
    x1 = np.linspace(X_range1[0],X_range1[1], xn)
    xx0,xx1 = np.meshgrid(x0,x1)
    x = np.c_[np.reshape(xx0,xn*xn), np.reshape(xx1,xn*xn)]
    output, _, _, _ = MLP_forward(U1,U2,P,C,x)
    plt.figure(1, figsize=(4,4))
    for ic in range(C):
        f = output[:,ic]
        f = f.reshape(xn, xn)
        f = f.T
        cont = plt.contour(xx0, xx1, f, levels=[0.8, 0.9],
                           colors=['cornflowerblue', 'black'])
        cont.clabel(fmt='%1.1f', fontsize=9)
    plt.xlim(X_range0)
    plt.ylim(X_range1) 

# 데이터를 그리기 ------------------------------
def Show_data(x, t):
    wk, n = t.shape
    c = [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]]
    for i in range(n):
        plt.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1],
                 linestyle='none',
                 marker='o', markeredgecolor='black',
                 color=c[i], alpha=0.8)
    plt.grid(True)


# 메인 ------------------------------------
plt.figure(1, figsize=(8, 3.7))
plt.subplot(1, 2, 1)
Show_data(X_train, Y_train)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.title('Training Data')
plt.subplot(1, 2, 2)
Show_data(X_test, Y_test)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.title('Test Data')
plt.show()

# test ---
P = 8
C = 3
D = 2

U1 = np.random.randn(P*(D+1))
U1 = U1.reshape(P,D+1)
U2 = np.random.randn(C*(P+1))
U2 = U2.reshape(C,P+1)

rho = 0.01 # learning rate
epoch = 300
N, D = X_train.shape #traing dataset size
batch_size = 16
batch_num = X_train.shape[0] // batch_size
sIdx = np.arange(X_train.shape[0])
startTime = time.time()

error_train, error_valid, U1, U2 = trainingMLP(epoch, U1, U2, P, C, X_train, Y_train, X_valid, Y_valid, rho)

calculation_time = time.time() - startTime
print(f"Calculation time:{calculation_time:0.3f} sec\n")

print(f"Final Updated U1 Matrix:{U1}\n")
print(f"Final Updated U1 Matrix:{U2}\n")

plt.figure(1,figsize=(3,3))
plt.plot(error_train, 'black', label='training')
plt.plot(error_valid,'cornflowerblue', label='validation')
plt.legend()
plt.show()


infer_train, _, _, _ = MLP_forward(U1,U2,P,C,X_train)
infer_test, _, _, _ = MLP_forward(U1,U2,P,C,X_test)

maxOutputIndex_train = np.argmax(infer_train, axis=1)
maxOutputIndex_test = np.argmax(infer_test, axis=1)

onehotOutput_train = np.eye(Y_train.shape[1])[maxOutputIndex_train]
onehotOutput_test = np.eye(Y_train.shape[1])[maxOutputIndex_test]

maxTargetIndex_train = np.argmax(Y_train, axis=1)
maxTargetIndex_test = np.argmax(Y_test, axis=1)

correct_num = 0.0
for n in range(X_train.shape[0]):
    if(maxTargetIndex_train[n] == maxOutputIndex_train[n]):
        correct_num += 1.0
accuracy_train = correct_num/X_train.shape[0]

correct_num = 0.0
for n in range(X_test.shape[0]):
    if(maxTargetIndex_test[n] == maxOutputIndex_test[n]):
        correct_num += 1.0
accuracy_test = correct_num/X_test.shape[0]

print(f'accuracy for training dataset:{accuracy_train}\n')
print(f'accuracy for test dataset:{accuracy_test}\n')


plt.figure(1, figsize=(8,4))
plt.subplot(1,2,1)
Show_data(X_train,onehotOutput_train)
Show_MLP_Contour(U1, U2, P, C)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.title('Train Data')
plt.subplot(1,2,2)
Show_data(X_test, onehotOutput_test)
Show_MLP_Contour(U1, U2, P, C)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.title('Test Data')
plt.show()