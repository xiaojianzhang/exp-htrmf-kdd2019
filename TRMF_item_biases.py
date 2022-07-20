#### Libraries
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from numpy import linalg as LA
import random

class TRMF(object):

    def __init__(self, Y, Missing_Value_Coor, Omega, Y_size, lagset, latent_dim, lambda_f, lambda_x, lambda_w, eta, max_iterations, warm_start=False, seed=None, true_missing_value=None, true_forecasting_value=None):
        """

        """
        self.Y = Y #missing entries is np.nan
        self.x_coor, self.y_coor = Missing_Value_Coor
        self.Y[self.x_coor, self.y_coor] = 0.0 #missing entries is set as 0
        self.Omega = Omega #0/1 matrix, where "0" = missing value, "1" = observed value.
        self.n, self.T = Y_size
        self.L = lagset
        self.L.sort()
        self.L_len = len(self.L) #the length of the lag set
        self.max_lag = max(self.L) #the largest lag
        self.L2index = self._lag2index(self.L)
        self.L_union_zero = [0] + self.L #the union of the lag set and {0}.

        self.k = latent_dim
        self.lambda_f = lambda_f
        self.lambda_x = lambda_x
        self.lambda_w = lambda_w
        self.eta = eta
        self.max_its = max_iterations
        if warm_start is False:
            self._random_init_F_X(seed)
            self._initialize_b()
        else:
            pass

        self.W = np.zeros([self.k, self.L_len])
        self._update_W()
        self.true_missing_value = true_missing_value
        self.true_forecasting_value = true_forecasting_value

    def _random_init_F_X(self, seed):
        if seed is not None:
            np.random.seed(seed)

        self.F = np.random.uniform(0, 3, size=(self.k, self.n)) #shape=(n, k)
        self.X = np.random.uniform(0, 3, size=(self.k, self.T)) #shape=(k, T)

    def _initialize_b(self):
        self.b = np.mean(self.Y, axis=1)

    def _lag2index(self, lagset):
        lag2index = {}
        for idx, lag in enumerate(lagset):
            lag2index[lag] = idx
        return lag2index

    def Alternating_Minimization(self):
        """Train the TRMF model using alternating minimization."""
        Losses = {'Total':[]}


        self._sum_square_error()
        self._Frobenius_norm_F()
        self._Frobenius_norm_X()
        self._AR_X_regularizer()
        self._Frobenius_norm_W()
        self._total_loss()

        Losses['Total'].append(self.total_loss)

        for it in range(self.max_its):
            print('iteration {} starts'.format(it+1))

            print('Update F')
            self._update_F()

            print('Update X')
            self._update_X()

            print('Update b')
            self._update_b()

            print('Update W')
            self._update_W()

            print('Compute Losses')
            self._sum_square_error()
            self._Frobenius_norm_F()
            self._Frobenius_norm_X()
            self._AR_X_regularizer()
            self._Frobenius_norm_W()
            self._total_loss()
            self._imputation_error()
            self._forecasting_error()

            Losses['Total'].append(self.total_loss)

            print('iteration {} complete: SSE: {}, Total loss: {}, Imputation NRMSE: {}, Forecasting NRMSE {}, biases {}'.format(it+1, self.SSE, self.total_loss, self.imputation_NRMSE, self.forecasting_NRMSE, self.b))

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
                    Y_i.append(self.Y[i,t]-self.b[i])
                    X_i.append(self.X[:,t].tolist())

            Y_i = np.array(Y_i)
            X_i = np.array(X_i)

            #solve for f_i
            A = self.lambda_f * np.eye(self.k) + np.dot(X_i.T, X_i)
            b = np.dot(X_i.T, Y_i)
            f_i = np.linalg.solve(A, b)
            self.F[:,i] = f_i

    def _update_X(self):
        #update the time-dependent latent matrix X
        m = self.max_lag + 1
        sparse_matrices = []
        for t in range(m-1, self.T):
            W_t_bar = []
            for j in range(self.T):
                l = t - j
                if l in self.L_union_zero:
                    if l == 0:
                        W_t_bar.append(np.eye(self.k))
                    else:
                        W_t_bar.append(-np.diag(self.W[:,self.L2index[l]]))
                else:
                    W_t_bar.append(np.zeros([self.k,self.k]))
            W_t_bar = np.hstack(tuple(W_t_bar))
            W_t_bar = csr_matrix(W_t_bar)
            sparse_matrices.append(W_t_bar.transpose().dot(W_t_bar))

        sum_W_t_transpose_W_t = sum(sparse_matrices)

        F_bar = []
        for t in range(self.T):
            F_t = []
            for i in range(self.n):
                if self.Omega[i,t] == 1:
                    F_t.append(self.F[:,i])
                else:
                    F_t.append(np.zeros([self.k]))
            F_t = np.vstack(tuple(F_t))
            F_bar.append(F_t)

        F_bar = csr_matrix(block_diag(*F_bar))
        B_bar = np.outer(self.b, np.ones(self.T))
        B_bar[self.x_coor, self.y_coor] = 0
        Y_bar = self.Y.flatten('F') - B_bar.flatten('F')

        #solve for f_i
        A = 2.0*F_bar.transpose().dot(F_bar) + self.lambda_x*sum_W_t_transpose_W_t + self.eta*self.lambda_x*identity(self.T*self.k, format='csr')
        b = 2*F_bar.transpose().dot(Y_bar)
        x_1_to_T = spsolve(A, b)
        self.X = np.reshape(x_1_to_T, (self.k, self.T), order='F')


    def _update_W(self):
        #update lag_val W
        m = self.max_lag + 1
        for row in range(self.k):
            #"x_r_bar" corresponds to "y" in ridge regression
            x_r_bar = self.X[row, m-1:]

            #"X_r_bar" corresponds to "X" in ridge regression
            X_r_bar = []
            for t in range(m-1, self.T):
                x_r_m_bar = [self.X[row, t-self.L2index[l]] for l in self.L]
                X_r_bar.append(x_r_m_bar)

            X_r_bar = np.array(X_r_bar)

            #solve for W_r
            A = 2*float(self.lambda_w)/float(self.lambda_x) * np.eye(self.L_len) + np.dot(X_r_bar.T, X_r_bar)
            b = np.dot(X_r_bar.T, x_r_bar)
            w_r = np.linalg.solve(A, b)
            self.W[row,:] = w_r

    def _update_b(self):
        #update item biases b
        F_X = np.dot(self.F.T, self.X)
        F_X[self.x_coor, self.y_coor] = 0
        Diff = self.Y - F_X
        self.b = np.divide(np.sum(Diff, axis=1), np.sum(self.Omega, axis=1))


    def _sum_square_error(self):
        Y_estimate = np.dot(self.F.T, self.X) + np.outer(self.b, np.ones(self.T))
        self.missing_value_estimate = Y_estimate[self.x_coor, self.y_coor]
        Y_estimate[self.x_coor, self.y_coor] = 0.0
        self.SSE = np.sum((self.Y - Y_estimate)**2.0)

    def _Frobenius_norm_F(self):
        self.Fro_norm_F = self.lambda_f * LA.norm(self.F, 'fro')**2.0

    def _Frobenius_norm_X(self):
        self.Fro_norm_X = 0.5 * self.eta * self.lambda_x * LA.norm(self.X, 'fro')**2.0

    def _AR_X_regularizer(self):
        # m = self.max_lag + 1
        # sparse_matrices = []
        # for t in range(m-1, self.T):
        #     W_t_bar = []
        #     for j in range(self.T):
        #         l = t - j
        #         if l in self.L_union_zero:
        #             if l == 0:
        #                 W_t_bar.append(np.eye(self.k))
        #             else:
        #                 W_t_bar.append(-np.diag(self.W[:,self.L2index[l]]))
        #         else:
        #             W_t_bar.append(np.zeros([self.k,self.k]))
        #     W_t_bar = np.hstack(tuple(W_t_bar))
        #     W_t_bar = csr_matrix(W_t_bar)
        #     sparse_matrices.append(W_t_bar.transpose().dot(W_t_bar))

        # sum_W_t_transpose_W_t = sum(sparse_matrices)
        # x_1_to_T = self.X.flatten('F')
        # self.AR_X_norm = 0.5 * self.lambda_x * np.dot(x_1_to_T.T, sum_W_t_transpose_W_t.dot(x_1_to_T))
        self.AR_X_norm = 0.0
        m = self.max_lag + 1
        for t in range(m-1, self.T):
            self.AR_X_norm += np.sum((self.X[:,t] - sum([np.diag(self.W[:,self.L2index[l]]).dot(self.X[:,t-l]) for l in self.L]))**2.0)
        self.AR_X_norm *= 0.5 * self.lambda_x

    def _Frobenius_norm_W(self):
        self.Fro_norm_W = self.lambda_w * LA.norm(self.W, 'fro')**2.0

    def _total_loss(self):
        self.total_loss = self.SSE + self.Fro_norm_F + self.Fro_norm_X + self.Fro_norm_W + self.AR_X_norm

    def _imputation_error(self):
        self.imputation_NRMSE = np.sqrt(np.sum((self.missing_value_estimate-self.true_missing_value)**2.0)/len(self.x_coor)) / np.mean(self.true_missing_value)

    def _forecasting_error(self):
        new_x = sum([np.diag(self.W[:,self.L2index[l]]).dot(self.X[:,128-l]) for l in self.L])
        self.forecasting_NRMSE = np.sqrt(np.sum((self.F.T.dot(new_x)+self.b-self.true_forecasting_value)**2.0)/self.k) / np.mean(self.true_forecasting_value)

    def Forecasting(self):
        pass

    def Missing_Value_Imputation(self):
        pass

def random_generate_missing_values(n, T, seed=None):
    if seed is not None:
        random.seed(0)

    x_coor = []
    y_coor = []
    for x in range(n):
        ys = random.sample(range(T), k=20)
        for y in ys:
            x_coor.append(x)
            y_coor.append(y)

    return x_coor, y_coor

def zero_one_indicator_matrix(n,T,x_coor,y_coor):
    Omega = np.ones([n, T], dtype='int')
    Omega[x_coor, y_coor] = 0

    return Omega


if __name__ == "__main__":
    # import pdb
    # pdb.set_trace()
    #Y = np.load('./dataset/32_items_129_days.npy')[:,:-1]
    #true_forecasting_value = np.load('./dataset/32_items_129_days.npy')[:,-1]
    #n, T = Y.shape
    import pickle
    with open("./dataset/8_categories_129_days.pkl", "rb") as f:
        Y = pickle.load(f)
        Y = np.vstack(tuple([Y[k] for k in range(len(Y))]))
        print(Y.shape)
    true_forecasting_value = Y[:,-1]
    n, T = Y.shape

    missing_value_x_coor, missing_value_y_coor = random_generate_missing_values(n, T, seed=0)

    Omega = zero_one_indicator_matrix(n,T,missing_value_x_coor,missing_value_y_coor)

    true_missing_value = Y[missing_value_x_coor, missing_value_y_coor]
    Y[missing_value_x_coor, missing_value_y_coor] = np.nan

    model = TRMF(Y=Y, Missing_Value_Coor=(missing_value_x_coor, missing_value_y_coor),
                 Omega=Omega, Y_size=(n,T), lagset=[1,2,3,4,5,6,7],
                 latent_dim=25, lambda_f=0.5, lambda_x=1000, lambda_w=0.5,
                 eta=0.001, max_iterations=1000, warm_start=False, seed=None,
                 true_missing_value=true_missing_value, true_forecasting_value=true_forecasting_value)

    model.Alternating_Minimization()
