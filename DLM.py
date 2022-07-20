#### Libraries
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from numpy import linalg as LA
from copy import deepcopy
import random
import pickle

class DLM(object):

    def __init__(self, Y, imputation_value_x_coor, imputation_value_y_coor, forecasting_x_coor, forecasting_y_coor, Omega, Y_size, latent_dim, lambda_f, lambda_x,eta, max_iterations, forecasting_horizon=4, seed=None, true_imputation_value=None, true_forecasting_value=None):
        """

        """
        self.Y = Y #observable matrix
        self.imputation_x_coor, self.imputation_y_coor = imputation_value_x_coor, imputation_value_y_coor
        self.Y[np.where(Omega==0)]=0
        self.Omega = Omega #0/1 matrix, where "0" = missing value, "1" = observed value.
        self.n, self.T = Y_size

        self.k = latent_dim
        self.lambda_f = lambda_f
        self.lambda_x = lambda_x
        self.eta = eta
        self.max_its = max_iterations
        self._random_generator(seed)

        self._update_W()
        self.true_imputation_value = true_imputation_value
        self.true_forecasting_value = true_forecasting_value
        self.forecasting_x_coor, self.forecasting_y_coor = forecasting_x_coor, forecasting_y_coor
        self.forecasting_horizon = forecasting_horizon
        self.best_results = {'Imputation NRMSE': 10, 'Imputation ND': 10, 'Forecasting NRMSE': 10, 'Forecasting ND':10}

    def _random_generator(self, seed):
        if seed is not None:
            np.random.seed(seed)

        self.F = np.random.uniform(0, 10, size=(self.k, self.n)) #shape=(n, k)
        self.X = np.random.uniform(0, 10, size=(self.k, self.T)) #shape=(k, T)

    def Alternating_Minimization(self):
        """Train the TRMF model using alternating minimization."""
        Losses = {'Total':[]}


        self._sum_square_error()
        self._Frobenius_norm_F()
        self._Frobenius_norm_X()
        self._AR_X_regularizer()
        self._total_loss()

        Losses['Total'].append(self.total_loss)

        for it in range(self.max_its):
            print('iteration {} starts'.format(it+1))

            print('Update F')
            self._update_F()

            print('Update X')
            self._update_X()

            print('Update W')
            self._update_W()

            print('Compute Losses')
            self._sum_square_error()
            self._Frobenius_norm_F()
            self._Frobenius_norm_X()
            self._AR_X_regularizer()
            self._total_loss()
            self._imputation_error()
            self._forecasting_error()

            Losses['Total'].append(self.total_loss)

            print('iteration {} complete: SSE: {}, Total loss: {}, Imputation NRMSE: {}, Imputation ND: {}, Forecasting NRMSE {}, Forecasting ND {}'.format(it+1, self.SSE, self.total_loss, self.imputation_NRMSE, self.imputation_ND, self.forecasting_NRMSE, self.forecasting_ND))
            print('Best results so far is {}'.format(self.best_results))
            if Losses['Total'][-1] > Losses['Total'][-2]:
                print("Error: total loss increases and it is impossible")
                break

    def _update_F(self):
        #update the latent item feature matrix F
        for i in range(self.n):
            #Y_i corresponds to "y" in ridge regression
            Y_i = []

            #X_i corresponds to "X" in ridge regression
            X_i = []
            for t in range(self.T):
                if self.Omega[i,t]==1:
                    Y_i.append(self.Y[i,t])
                    X_i.append(self.X[:,t].tolist())

            Y_i = np.array(Y_i)
            X_i = np.array(X_i)

            #solve for f_i
            A = self.lambda_f * np.eye(self.k) + np.dot(X_i.T, X_i)
            b = np.dot(X_i.T, Y_i)
            f_i = np.linalg.solve(A, b)
            assert np.allclose(np.dot(A, f_i), b), "update F: fail"
            self.F[:,i] = f_i

    def _update_X(self):
        #update the time-dependent latent matrix X
        m = 2
        sparse_matrices = []
        for t in range(m-1, self.T):
            W_t_bar = []
            for j in range(self.T):
                l = t - j
                if l in [0,1]:
                    if l == 0:
                        W_t_bar.append(np.eye(self.k))
                    else:
                        W_t_bar.append(-self.W)
                else:
                    W_t_bar.append(np.zeros([self.k,self.k]))
            W_t_bar = np.hstack(tuple(W_t_bar))
            W_t_bar = csr_matrix(W_t_bar)
            sparse_matrices.append(W_t_bar.transpose().dot(W_t_bar))

        sum_W_t_transpose_W_t = sum(sparse_matrices)

        F_bar = []
        Y_bar = []
        for t in range(self.T):
            F_t = []
            for i in range(self.n):
                if self.Omega[i,t] == 1:
                    F_t.append(self.F[:,i])
                    Y_bar.append(self.Y[i,t])
            F_t = np.vstack(tuple(F_t))
            F_bar.append(F_t)

        F_bar = csr_matrix(block_diag(*F_bar))
        Y_bar = np.array(Y_bar)


        #solve for f_i
        A = 2.0*F_bar.transpose().dot(F_bar) + self.lambda_x*sum_W_t_transpose_W_t + self.eta*self.lambda_x*identity(self.T*self.k, format='csr')
        b = 2*F_bar.transpose().dot(Y_bar)
        x_1_to_T = spsolve(A, b)
        assert np.allclose(A.dot(x_1_to_T), b), "update X: fail"
        self.X = np.reshape(x_1_to_T, (self.k, self.T), order='F')


    def _update_W(self):
        #update lag_val W
        b = sum([np.outer(self.X[:,t], self.X[:,t-1]) for t in range(1, self.T)])
        b = b.T

        A = sum([np.outer(self.X[:,t-1], self.X[:,t-1]) for t in range(1, self.T)])
        A = A.T

        W = np.linalg.solve(A, b)
        assert np.allclose(np.dot(A, W), b), "update W: fail"
        self.W = W.T

    def _sum_square_error(self):
        Y_estimate = np.dot(self.F.T, self.X)
        self.imputation_value_estimate = Y_estimate[self.imputation_x_coor, self.imputation_y_coor]
        self.imputation_value_estimate[np.where(self.imputation_value_estimate<0)]=0
        Y_estimate[np.where(self.Omega==0)] = 0.0
        self.SSE = np.sum((self.Y - Y_estimate)**2.0)

    def _Frobenius_norm_F(self):
        self.Fro_norm_F = self.lambda_f * LA.norm(self.F, 'fro')**2.0

    def _Frobenius_norm_X(self):
        self.Fro_norm_X = 0.5 * self.eta * self.lambda_x * LA.norm(self.X, 'fro')**2.0

    def _AR_X_regularizer(self):
        self.AR_X_norm = 0.0
        m = 2
        for t in range(m-1, self.T):
            self.AR_X_norm += LA.norm(self.X[:,t] - self.W.dot(self.X[:,t-1]))**2.0
        self.AR_X_norm *= 0.5 * self.lambda_x

    def _total_loss(self):
        self.total_loss = self.SSE + self.Fro_norm_F + self.Fro_norm_X + self.AR_X_norm

    def _imputation_error(self):
        cnt = self.true_imputation_value.shape[0]
        mse = LA.norm(self.imputation_value_estimate-self.true_imputation_value)**2.0
        #mape = np.sum(np.absolute(np.divide((self.imputation_value_estimate-self.true_imputation_value), self.true_imputation_value)))
        nd = np.sum(np.absolute(self.imputation_value_estimate-self.true_imputation_value))
        nd_den = np.sum(np.absolute(self.true_imputation_value))

        self.imputation_NRMSE = np.sqrt(mse/cnt)/(nd_den/cnt)
        #self.imputation_MAPE = mape/cnt
        self.imputation_ND = nd/nd_den

        if self.best_results['Imputation NRMSE'] > self.imputation_NRMSE:
            self.best_results['Imputation NRMSE'] = self.imputation_NRMSE

        if self.best_results['Imputation ND'] > self.imputation_ND:
            self.best_results['Imputation ND'] = self.imputation_ND

    def _forecasting_error(self):
        X = deepcopy(self.X)
        cnt = self.true_forecasting_value.shape[0]
        for t in range(self.forecasting_horizon):
            new_x = self.W.dot(X[:,-1])
            X = np.hstack((X, new_x[:,np.newaxis]))
        forecasting_value_estimate = self.F.T.dot(X[:,-self.forecasting_horizon:])
        forecasting_value_estimate = forecasting_value_estimate[self.forecasting_x_coor, self.forecasting_y_coor]
        mse = LA.norm(forecasting_value_estimate-self.true_forecasting_value)**2.0
        #mape = np.sum(np.absolute(np.divide((forecasting_value_estimate-self.true_forecasting_value), self.true_forecasting_value)))
        nd = np.sum(np.absolute(forecasting_value_estimate-self.true_forecasting_value))
        nd_den = np.sum(np.absolute(self.true_forecasting_value))

        self.forecasting_NRMSE = np.sqrt(mse/cnt)/(nd_den/cnt)
        #self.forecasting_MAPE = mape/cnt
        self.forecasting_ND = nd/nd_den

        if self.best_results['Forecasting NRMSE'] > self.forecasting_NRMSE:
            self.best_results['Forecasting NRMSE'] = self.forecasting_NRMSE

        if self.best_results['Forecasting ND'] > self.forecasting_ND:
            self.best_results['Forecasting ND'] = self.forecasting_ND


def random_generate_missing_values(n, T, missing_value_x_coor, missing_value_y_coor, seed=None):
    if seed is not None:
        random.seed(0)

    x_coor = []
    y_coor = []
    for x in range(n):
        mis_y_indices = missing_value_y_coor[np.where(missing_value_x_coor==x)[0]]
        if mis_y_indices.shape[0] >= 40:
            continue
        else:
            obs_y_indices = [t for t in range(T) if t not in mis_y_indices]
            ys = random.sample(obs_y_indices, k=40-mis_y_indices.shape[0])
            for y in ys:
                x_coor.append(x)
                y_coor.append(y)

    return x_coor, y_coor

def zero_one_indicator_matrix(n,T,missing_value_x_coor, missing_value_y_coor, imputation_value_x_coor, imputation_value_y_coor):
    Omega = np.ones([n, T], dtype='int')
    Omega[missing_value_x_coor, missing_value_y_coor] = 0
    Omega[imputation_value_x_coor, imputation_value_y_coor] = 0

    return Omega


if __name__ == "__main__":
    import pdb

    with open("./dataset/kuaixiao_realdata_8_categories.pkl", "rb") as f:
        datamatrix_dict = pickle.load(f) #dict of submatrices
        Data_Matrix = np.vstack(tuple([Y_cat for Y_cat in datamatrix_dict.values()]))

    forecasting_horizon = 6
    #Data_Matrix = np.load('./dataset/1059_items_maochao_realdata.npy') #load original data matrix
    Y_train = Data_Matrix[:,:-forecasting_horizon] #Y_train is used to fit F and X
    n, T = Y_train.shape
    print('Training matrix has {0} items and {1} time steps'.format(n, T))

    missing_value_x_coor, missing_value_y_coor = np.where(Y_train==-1)
    imputation_value_x_coor, imputation_value_y_coor = random_generate_missing_values(n, T, missing_value_x_coor, missing_value_y_coor)
    true_imputation_value = Y_train[imputation_value_x_coor, imputation_value_y_coor]
    Omega = zero_one_indicator_matrix(n,T,missing_value_x_coor, missing_value_y_coor, imputation_value_x_coor, imputation_value_y_coor)
    print('{}% of entries are missing'.format(np.where(Omega==0)[0].shape[0] / Omega.size))

    forecasting_matrix = Data_Matrix[:,-forecasting_horizon:]
    forecasting_x_coor, forecasting_y_coor = np.where(forecasting_matrix != -1)
    true_forecasting_value = forecasting_matrix[forecasting_x_coor, forecasting_y_coor]

    model = DLM(Y=Y_train, imputation_value_x_coor=imputation_value_x_coor, imputation_value_y_coor=imputation_value_y_coor,
                 forecasting_x_coor=forecasting_x_coor, forecasting_y_coor=forecasting_y_coor,
                 Omega=Omega, Y_size=(n,T),
                 latent_dim=20, lambda_f=5, lambda_x=10,
                 eta=1, max_iterations=30, forecasting_horizon=forecasting_horizon, seed=None,
                 true_imputation_value=true_imputation_value, true_forecasting_value=true_forecasting_value)

    model.Alternating_Minimization()
    import matplotlib.pyplot as plt
    Y_fit = model.F.T.dot(model.X)
    Y = Data_Matrix[:,:-forecasting_horizon]
    for i in random.sample(range(n), 32):
        plt.figure(i)
        plt.plot(range(T), Y[i, :])
        plt.plot(range(T), Y_fit[i, :])
    plt.show()
