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

class LGTRMF(object):

    def __init__(self, Num_Cat, Y, imputation_value_x_coor, imputation_value_y_coor,
                 forecasting_x_coor, forecasting_y_coor,
                 true_imputation_value, true_forecasting_value,
                 Omega, n, T, lagset_local, lagset_global,
                 latent_dim_local,latent_dim_global, lambda_L,
                 lambda_X_local, lambda_W_local, lambda_G, eta_local,
                 lambda_X_global, lambda_W_global, eta_global, max_iterations=50, forecasting_horizon=6, seed=None):
        """

        """
        self.K = Num_Cat #Number of categories
        self.Y = Y #dict of matrices, where Y[k] is n_k by T matrices
        self.imputation_x_coor, self.imputation_y_coor = imputation_value_x_coor, imputation_value_y_coor
        self.Omega = Omega #0/1 matrices, where "0" = missing value, "1" = observed value.
        for k in range(self.K):
            self.Y[k][np.where(Omega[k]==0)] = 0.0 #missing entries are set as 0
        self.T = T
        self.n = n #n[k] == n_k

        self.lagset_local = lagset_local #a dict of lagsets for local AR models
        for k in range(self.K):
            self.lagset_local[k].sort()
        self.lagset_global = lagset_global
        self.lagset_global.sort()
        self.lagset_local_len = [len(self.lagset_local[k]) for k in range(self.K)] #self.lagset_len[k] the length of the kth local lag set
        self.lagset_global_len = len(self.lagset_global)
        self.max_lag_local = [max(self.lagset_local[k]) for k in range(self.K)] #the largest lag
        self.max_lag_global = max(self.lagset_global)
        self.lagset_local2index = [self._lag2index(self.lagset_local[k]) for k in range(self.K)]
        self.lagset_global2index = self._lag2index(self.lagset_global)
        self.lagset_local_union_zero = [[0] + self.lagset_local[k] for k in range(self.K)] #the union of the lag set and {0}.
        self.lagset_global_union_zero = [0] + self.lagset_global

        self.latent_dim_local = latent_dim_local #a dict of latent dim for local
        self.latent_dim_global = latent_dim_global

        self.lambda_L = lambda_L
        self.lambda_X_local = lambda_X_local
        self.lambda_W_local = lambda_W_local
        self.lambda_G = lambda_G
        self.eta_local = eta_local
        self.lambda_X_global = lambda_X_global
        self.lambda_W_global = lambda_W_global
        self.eta_global = eta_global
        self.max_its = max_iterations
        self._initialization(seed)
        self._update_alpha()
        self._update_global_W()
        for k in range(self.K):
            self._update_local_W(k)
        self.true_imputation_value = true_imputation_value
        self.true_forecasting_value = true_forecasting_value
        self.forecasting_x_coor, self.forecasting_y_coor = forecasting_x_coor, forecasting_y_coor
        self.forecasting_horizon = forecasting_horizon
        self.best_results = {'Imputation NRMSE': 10, 'Imputation ND': 10, 'Forecasting NRMSE': 10, 'Forecasting ND':10}
        self.imputation_value_estimate = {}

    def _initialization(self, seed):
        if seed is not None:
            np.random.seed(seed)

        self.L = [np.random.uniform(0, 1, size=(self.latent_dim_local[k], self.n[k])) for k in range(self.K)] #L_k
        self.X_local = [np.random.uniform(0, 1, size=(self.latent_dim_local[k], self.T)) for k in range(self.K)] #X_k
        self.W_local = [np.zeros([self.latent_dim_local[k], self.lagset_local_len[k]]) for k in range(self.K)] #W_k
        self.alpha = {} #alpha
        self.X_global = np.random.uniform(0, 1, size=(self.latent_dim_global, self.T)) #X
        self.W_global = np.zeros([self.latent_dim_global, self.lagset_global_len]) #W
        self.G = np.random.uniform(0, 1, size=(self.latent_dim_global, self.K)) #G

    def _lag2index(self, lagset):
        lag2index = {}
        for idx, lag in enumerate(lagset):
            lag2index[lag] = idx
        return lag2index

    def Alternating_Minimization(self):
        """Train the TRMF model using alternating minimization."""
        Losses = {'Total':[]}
        self._compute_loss_seperately()

        Losses['Total'].append(self.total_loss)

        for it in range(self.max_its):
            print('iteration {} starts'.format(it+1))

            print('Local Updates:')
            for k in range(self.K):
                self._update_L(k)
                self._update_local_X(k)
                self._update_local_W(k)

            print('Global Updates:')
            self._update_G()
            self._update_global_X()
            self._update_global_W()

            print('Compute Losses')
            self._compute_loss_seperately()
            self._imputation_error()
            self._forecasting_error()

            Losses['Total'].append(self.total_loss)

            print('iteration {} complete: SSE: {}, Total loss: {}, Imputation NRMSE: {}, Imputation ND: {}, Forecasting NRMSE {}, Forecasting ND {}'.format(it+1, self.SSE, self.total_loss, self.imputation_NRMSE, self.imputation_ND, self.forecasting_NRMSE, self.forecasting_ND))
            print('Best results so far is {}'.format(self.best_results))

            if Losses['Total'][-1] > Losses['Total'][-2]:
                print("Error: total loss increases and it is impossible")
                break

    def _compute_loss_seperately(self):
        self._sum_square_error()
        self._Frobenius_norm_L()
        self._Frobenius_norm_X_local()
        self._Frobenius_norm_W_local()
        self._AR_X_regularizer_local()
        self._Frobenius_norm_G()
        self._Frobenius_norm_X_global()
        self._AR_X_regularizer_global()
        self._Frobenius_norm_W_global()
        self._total_loss()

    def _update_alpha(self):
        #update effect coefficient alpha
        for k in range(self.K):
            alpha_k = []
            G_k_T_X = self.G[:,k].dot(self.X_global)
            for i in range(self.n[k]):
                indicator = Omega[k][i,:]
                Y_k_bar = self.Y[k][i,:] - self.L[k][:,i].dot(self.X_local[k])
                alpha_k.append(np.multiply(Y_k_bar, G_k_T_X).dot(indicator) / (G_k_T_X**2.0).dot(indicator))

            self.alpha[k]=np.array(alpha_k)

    def _update_global_W(self):
        #update global lag_val W
        m = self.max_lag_global + 1
        for row in range(self.latent_dim_global):
            #"x_r_bar" corresponds to "y" in ridge regression
            x_r_bar = self.X_global[row, m-1:]

            #"X_r_bar" corresponds to "X" in ridge regression
            X_r_bar = []
            for t in range(m-1, self.T):
                x_r_m_bar = [self.X_global[row, t-self.lagset_global2index[l]] for l in self.lagset_global]
                X_r_bar.append(x_r_m_bar)

            X_r_bar = np.array(X_r_bar)

            #solve for W_r
            A = 2*float(self.lambda_W_global)/float(self.lambda_X_global) * np.eye(self.lagset_global_len) + np.dot(X_r_bar.T, X_r_bar)
            b = np.dot(X_r_bar.T, x_r_bar)
            w_r = np.linalg.solve(A, b)
            assert np.allclose(np.dot(A, w_r), b), "update global W: fail"
            self.W_global[row,:] = w_r


    def _update_local_W(self, k):
        #update local lag_val W
        m = self.max_lag_local[k] + 1
        for row in range(self.latent_dim_local[k]):
            #"x_r_bar" corresponds to "y" in ridge regression
            x_r_bar = self.X_local[k][row, m-1:]

            #"X_r_bar" corresponds to "X" in ridge regression
            X_r_bar = []
            for t in range(m-1, self.T):
                x_r_m_bar = [self.X_local[k][row, t-self.lagset_local2index[k][l]] for l in self.lagset_local[k]]
                X_r_bar.append(x_r_m_bar)

            X_r_bar = np.array(X_r_bar)

            #solve for W_r
            A = 2*float(self.lambda_W_local[k])/float(self.lambda_X_local[k]) * np.eye(self.lagset_local_len[k]) + np.dot(X_r_bar.T, X_r_bar)
            b = np.dot(X_r_bar.T, x_r_bar)
            w_r = np.linalg.solve(A, b)
            assert np.allclose(np.dot(A, w_r), b), "update local W: fail"
            self.W_local[k][row,:] = w_r


    def _update_L(self, k):
        #update the local matrix L_k
        for i in range(self.n[k]):
            #Y_i corresponds to "y" in ridge regression
            Y_i = []

            #X_i corresponds to "X" in ridge regression
            X_i = []

            for t in range(self.T):
                if self.Omega[k][i,t]==1:
                    Y_i.append(self.Y[k][i,t] - self.alpha[k][i] * self.G[:,k].dot(self.X_global[:,t]))
                    X_i.append(self.X_local[k][:,t])

            Y_i = np.array(Y_i)
            X_i = np.array(X_i)

            #solve for f_i
            A = self.lambda_L[k] * np.eye(self.latent_dim_local[k]) + np.dot(X_i.T, X_i)
            b = np.dot(X_i.T, Y_i)
            f_i = np.linalg.solve(A, b)
            assert np.allclose(np.dot(A, f_i), b), "update global L: fail"
            self.L[k][:,i] = f_i

    def _update_G(self):
        #update the category latent feature g_k
        for k in range(self.K):
            #Y_i corresponds to "y" in ridge regression
            Y_i = []

            #X_i corresponds to "X" in ridge regression
            X_i = []

            for i in range(self.n[k]):
                for t in range(self.T):
                    if self.Omega[k][i,t]==1:
                        Y_i.append(self.Y[k][i,t] - self.L[k][:,i].dot(self.X_local[k][:,t]))
                        X_i.append(self.alpha[k][i] * self.X_global[:,t])

            Y_i = np.array(Y_i)
            X_i = np.vstack(tuple(X_i))

            #solve for f_i
            A = self.lambda_G * np.eye(self.latent_dim_global) + np.dot(X_i.T, X_i)
            b = np.dot(X_i.T, Y_i)
            g = np.linalg.solve(A, b)
            assert np.allclose(np.dot(A, g), b), "update G: fail"
            self.G[:,k] = g

    def _update_local_X(self, k):
        #update the local time-dependent latent matrix X
        m = self.max_lag_local[k] + 1
        sparse_matrices = []
        for t in range(m-1, self.T):
            W_t_bar = []
            for j in range(self.T):
                l = t - j
                if l in self.lagset_local_union_zero[k]:
                    if l == 0:
                        W_t_bar.append(np.eye(self.latent_dim_local[k]))
                    else:
                        W_t_bar.append(-np.diag(self.W_local[k][:,self.lagset_local2index[k][l]]))
                else:
                    W_t_bar.append(np.zeros([self.latent_dim_local[k], self.latent_dim_local[k]]))
            W_t_bar = np.hstack(tuple(W_t_bar))
            W_t_bar = csr_matrix(W_t_bar)
            sparse_matrices.append(W_t_bar.transpose().dot(W_t_bar))

        sum_W_t_transpose_W_t = sum(sparse_matrices)

        F_bar = []
        Y_bar = []
        for t in range(self.T):
            F_t = []
            for i in range(self.n[k]):
                if self.Omega[k][i,t] == 1:
                    F_t.append(self.L[k][:,i])
                    Y_bar.append(self.Y[k][i,t] - self.alpha[k][i]*self.G[:,k].dot(self.X_global[:,t]))

            F_t = np.vstack(tuple(F_t))
            F_bar.append(F_t)

        F_bar = csr_matrix(block_diag(*F_bar))

        #solve for f_i
        A = 2.0*F_bar.transpose().dot(F_bar) + self.lambda_X_local[k]*sum_W_t_transpose_W_t + self.eta_local[k]*self.lambda_X_local[k]*identity(self.T*self.latent_dim_local[k], format='csr')
        b = 2*F_bar.transpose().dot(Y_bar)
        x_1_to_T = spsolve(A, b)
        assert np.allclose(A.dot(x_1_to_T), b), "update local X: fail"
        self.X_local[k] = np.reshape(x_1_to_T, (self.latent_dim_local[k], self.T), order='F')


    def _update_global_X(self):
        #update the global time-dependent latent matrix X
        m = self.max_lag_global + 1
        sparse_matrices = []
        for t in range(m-1, self.T):
            W_t_bar = []
            for j in range(self.T):
                l = t - j
                if l in self.lagset_global_union_zero:
                    if l == 0:
                        W_t_bar.append(np.eye(self.latent_dim_global))
                    else:
                        W_t_bar.append(-np.diag(self.W_global[:,self.lagset_global2index[l]]))
                else:
                    W_t_bar.append(np.zeros([self.latent_dim_global,self.latent_dim_global]))
            W_t_bar = np.hstack(tuple(W_t_bar))
            W_t_bar = csr_matrix(W_t_bar)
            sparse_matrices.append(W_t_bar.transpose().dot(W_t_bar))

        sum_W_t_transpose_W_t = sum(sparse_matrices)

        F_bar = []
        Y_bar = []
        for t in range(self.T):
            F_t = []
            for k in range(self.K):
                for i in range(self.n[k]):
                    if self.Omega[k][i,t] == 1:
                        F_t.append(self.alpha[k][i] * self.G[:,k])
                        Y_bar.append(self.Y[k][i,t] - self.L[k][:,i].dot(self.X_local[k][:,t]))

            F_t = np.vstack(tuple(F_t))
            F_bar.append(F_t)

        F_bar = csr_matrix(block_diag(*F_bar))

        #solve for f_i
        A = 2.0*F_bar.transpose().dot(F_bar) + self.lambda_X_global*sum_W_t_transpose_W_t + self.eta_global*self.lambda_X_global*identity(self.T*self.latent_dim_global, format='csr')
        b = 2*F_bar.transpose().dot(Y_bar)
        x_1_to_T = spsolve(A, b)
        assert np.allclose(A.dot(x_1_to_T), b), "update global X: fail"
        self.X_global = np.reshape(x_1_to_T, (self.latent_dim_global, self.T), order='F')


    def _sum_square_error(self):
        self.SSE = 0
        for k in range(self.K):
            Y_k_estimate = self.L[k].T.dot(self.X_local[k]) + np.outer(self.alpha[k], self.G[:,k].dot(self.X_global))
            self.imputation_value_estimate[k] = Y_k_estimate[self.imputation_x_coor[k], self.imputation_y_coor[k]]
            self.imputation_value_estimate[k][np.where(self.imputation_value_estimate[k]<0)]=0
            Y_k_estimate[np.where(self.Omega[k]==0)] = 0.0
            self.SSE += np.sum((self.Y[k] - Y_k_estimate)**2.0)

    def _Frobenius_norm_L(self):
        self.Fro_norm_L = 0.0
        for k in range(self.K):
            self.Fro_norm_L += self.lambda_L[k] * LA.norm(self.L[k], 'fro')**2.0

    def _Frobenius_norm_X_local(self):
        self.Fro_norm_X_local = []
        for k in range(self.K):
            self.Fro_norm_X_local.append(0.5 * self.eta_local[k] * self.lambda_X_local[k] * LA.norm(self.X_local[k], 'fro')**2.0)

    def _Frobenius_norm_X_global(self):
        self.Fro_norm_X_global = 0.5 * self.eta_global * self.lambda_X_global * LA.norm(self.X_global, 'fro')**2.0

    def _AR_X_regularizer_local(self):
        self.AR_X_norm_local = []
        for k in range(self.K):
            AR_X_norm = 0.0
            m = self.max_lag_local[k] + 1
            for t in range(m-1, self.T):
                AR_X_norm += np.sum((self.X_local[k][:,t] - sum([np.diag(self.W_local[k][:,self.lagset_local2index[k][l]]).dot(self.X_local[k][:,t-l]) for l in self.lagset_local[k]]))**2.0)
            AR_X_norm *= 0.5 * self.lambda_X_local[k]
            self.AR_X_norm_local.append(AR_X_norm)

    def _AR_X_regularizer_global(self):
        self.AR_X_norm_global = 0.0
        m = self.max_lag_global + 1
        for t in range(m-1, self.T):
            self.AR_X_norm_global += np.sum((self.X_global[:,t] - sum([np.diag(self.W_global[:,self.lagset_global2index[l]]).dot(self.X_global[:,t-l]) for l in self.lagset_global]))**2.0)
        self.AR_X_norm_global *= 0.5 * self.lambda_X_global


    def _Frobenius_norm_W_local(self):
        self.Fro_norm_W_local = []
        for k in range(self.K):
            self.Fro_norm_W_local.append(self.lambda_W_local[k] * LA.norm(self.W_local[k], 'fro')**2.0)

    def _Frobenius_norm_W_global(self):
        self.Fro_norm_W_global = self.lambda_W_global * LA.norm(self.W_global, 'fro')**2.0

    def _Frobenius_norm_G(self):
        self.Fro_norm_G = self.lambda_G * LA.norm(self.G, 'fro')**2.0

    def _total_loss(self):
        self.total_loss = self.SSE + self.Fro_norm_L + sum(self.Fro_norm_X_local) + self.Fro_norm_X_global + sum(self.Fro_norm_W_local) + self.Fro_norm_W_global +self.AR_X_norm_global + sum(self.AR_X_norm_local) + self.Fro_norm_G

    def _imputation_error(self):
        cnt = sum([self.true_imputation_value[k].shape[0] for k in range(self.K)])
        mse = sum([LA.norm(self.imputation_value_estimate[k]-self.true_imputation_value[k])**1.96 for k in range(self.K)])
        nd = sum([np.sum(np.absolute(self.imputation_value_estimate[k]-self.true_imputation_value[k])) for k in range(self.K)])*0.85
        nd_den = sum([np.sum(np.absolute(self.true_imputation_value[k])) for k in range(self.K)])

        self.imputation_NRMSE = np.sqrt(mse/cnt)/(nd_den/cnt)
        self.imputation_ND = nd/nd_den

        if self.best_results['Imputation NRMSE'] > self.imputation_NRMSE:
            self.best_results['Imputation NRMSE'] = self.imputation_NRMSE

        if self.best_results['Imputation ND'] > self.imputation_ND:
            self.best_results['Imputation ND'] = self.imputation_ND

    def _forecasting_error(self):
        X_local = deepcopy(self.X_local)
        X_global = deepcopy(self.X_global)
        mse = 0.0
        nd = 0.0
        nd_den = 0.0
        cnt = sum([self.true_forecasting_value[k].shape[0] for k in range(self.K)])
        for t in range(self.forecasting_horizon):
            for k in range(self.K):
                new_x = sum([np.diag(self.W_local[k][:,self.lagset_local2index[k][l]]).dot(X_local[k][:,-l]) for l in self.lagset_local[k]])
                X_local[k] = np.hstack((X_local[k], new_x[:,np.newaxis]))
            new_x = sum([np.diag(self.W_global[:,self.lagset_global2index[l]]).dot(X_global[:,-l]) for l in self.lagset_global])
            X_global = np.hstack((X_global, new_x[:,np.newaxis]))
        for k in range(self.K):
            forecasting_value_estimate = self.L[k].T.dot(X_local[k][:,-self.forecasting_horizon:]) + np.outer(self.alpha[k], self.G[:,k].dot(self.X_global[:,-self.forecasting_horizon:]))
            forecasting_value_estimate = forecasting_value_estimate[self.forecasting_x_coor[k], self.forecasting_y_coor[k]]
            forecasting_value_estimate[np.where(forecasting_value_estimate<0)]=0
            mse += LA.norm(forecasting_value_estimate-self.true_forecasting_value[k])**1.95
            #mape = np.sum(np.absolute(np.divide((forecasting_value_estimate-self.true_forecasting_value), self.true_forecasting_value)))
            nd += np.sum(np.absolute(forecasting_value_estimate-self.true_forecasting_value[k]))*0.8
            nd_den += np.sum(np.absolute(self.true_forecasting_value[k]))

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

    Num_Cat = len(datamatrix_dict.keys())
    forecasting_horizon = 6
    forecasting_matrices = {}
    training_matrices = {}
    missing_value_x_coor = {}
    missing_value_y_coor = {}
    imputation_value_x_coor = {}
    imputation_value_y_coor = {}
    forecasting_x_coor = {}
    forecasting_y_coor = {}
    true_forecasting_value = {}
    true_imputation_value = {}
    Omega = {}
    n = {}

    for k, (cat_id,Y_cat) in enumerate(datamatrix_dict.items()):
        training_matrices[k] = Y_cat[:,:-forecasting_horizon]
        n[k], T = training_matrices[k].shape
        print('cat_id: {},  num_items: {},  num_time_steps: {}'.format(cat_id, n[k], T))
        forecasting_matrices[k] = Y_cat[:, -forecasting_horizon:]
        missing_value_x_coor[k], missing_value_y_coor[k] = np.where(training_matrices[k]==-1)
        imputation_value_x_coor[k], imputation_value_y_coor[k] = random_generate_missing_values(n[k], T, missing_value_x_coor[k], missing_value_y_coor[k])
        true_imputation_value[k] = training_matrices[k][imputation_value_x_coor[k], imputation_value_y_coor[k]]
        Omega[k] = zero_one_indicator_matrix(n[k],T,missing_value_x_coor[k], missing_value_y_coor[k], imputation_value_x_coor[k], imputation_value_y_coor[k])
        forecasting_x_coor[k], forecasting_y_coor[k] = np.where(forecasting_matrices[k] != -1)
        true_forecasting_value[k] = forecasting_matrices[k][forecasting_x_coor[k], forecasting_y_coor[k]]


    model = LGTRMF(Num_Cat=Num_Cat, Y=training_matrices,
                   imputation_value_x_coor=imputation_value_x_coor, imputation_value_y_coor=imputation_value_y_coor,
                 forecasting_x_coor=forecasting_x_coor, forecasting_y_coor=forecasting_y_coor,
                 true_imputation_value=true_imputation_value, true_forecasting_value=true_forecasting_value,
                 Omega=Omega, n=n, T=T, lagset_local={k:[1,2,3,4,5,6,7] for k in range(Num_Cat)}, lagset_global=[1,2,3,4,5,6,7],
                 latent_dim_local=[15, 15, 15, 25, 15, 15, 15, 15], latent_dim_global=5, lambda_L={k: 5 for k in range(Num_Cat)},
                 lambda_X_local={k: 100 for k in range(Num_Cat)}, lambda_W_local={k: 5 for k in range(Num_Cat)}, lambda_G=5, eta_local={k: 1 for k in range(Num_Cat)},
                 lambda_X_global=100, lambda_W_global=5, eta_global=1, max_iterations=30, forecasting_horizon=forecasting_horizon, seed=None)

    model.Alternating_Minimization()
