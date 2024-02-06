import pickle
from abc import abstractmethod
from gurobipy import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np


class BaseModel(object):
    """
    Base class for models, to be used as coding pattern skeleton.
    Can be used for a model on a single cluster or on multiple clusters"""

    def __init__(self):
        """Initialization of your model and its hyper-parameters"""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Fit function to find the parameters according to (X, Y) data.
        (X, Y) formatting must be so that X[i] is preferred to Y[i] for all i.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # Customize what happens in the fit function
        return

    @abstractmethod
    def predict_utility(self, X):
        """Method to call the decision function of your model

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # Customize what happens in the predict utility function
        return

    def predict_preference(self, X, Y):
        """Method to predict which pair is preferred between X[i] and Y[i] for all i.
        Returns a preference for each cluster.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of preferences for each cluster. 1 if X is preferred to Y, 0 otherwise
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return (X_u - Y_u > 0).astype(int)

    def predict_cluster(self, X, Y):
        """Predict which cluster prefers X over Y THE MOST, meaning that if several cluster prefer X over Y, it will
        be assigned to the cluster showing the highest utility difference). The reversal is True if none of the clusters
        prefer X over Y.
        Compared to predict_preference, it indicates a cluster index.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, ) index of cluster with highest preference difference between X and Y.
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return np.argmax(X_u - Y_u, axis=1)

    def save_model(self, path):
        """Save the model in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the file in which the model will be saved
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(clf, path):
        """Load a model saved in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the path to the file to load
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


class RandomExampleModel(BaseModel):
    """Example of a model on two clusters, drawing random coefficients.
    You can use it to understand how to write your own model and the data format that we are waiting for.
    This model does not work well but you should have the same data formatting with TwoClustersMIP.
    """

    def __init__(self):
        self.seed = 444
        self.weights = self.instantiate()

    def instantiate(self):
        """No particular instantiation"""
        return []

    def fit(self, X, Y):
        """fit function, sets random weights for each cluster. Totally independant from X & Y.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        np.random.seed(self.seed)
        indexes = np.random.randint(0, 2, (len(X)))
        num_features = X.shape[1]
        weights_1 = np.random.rand(num_features)
        weights_2 = np.random.rand(num_features)

        weights_1 = weights_1 / np.sum(weights_1)
        weights_2 = weights_2 / np.sum(weights_2)
        self.weights = [weights_1, weights_2]
        return self

    def predict_utility(self, X):
        """Simple utility function from random weights.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        return np.stack([np.dot(X, self.weights[0]), np.dot(X, self.weights[1])], axis=1)


class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n°clusters: int
            Number of clusters to implement in the MIP.
        """
        self.L = n_pieces
        self.K = n_clusters
        self.n = 4
        self.seed = 123
        self.epsilon = 0.00001
        self.P = 2000
        self.M = 5
        self.criterion_utilite = {}
        self.sum_utilite = {}
        self.model = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables - To be completed."""
        # self.bigM = 100
        m = Model("Simple PL modelling")
        self.criteria = [[[m.addVar(name=f"u_{k}_{i}_{l}",vtype=GRB.CONTINUOUS, lb=0, ub=1) for l in range(self.L+1)] for i in range(self.n)] for k in range(self.K)]
        self.sigma_plus_x = [m.addVar(name=f"sigmax+_{j}",vtype=GRB.CONTINUOUS) for j in range(self.P)]
        self.sigma_plus_y = [m.addVar(name=f"sigmay+_{j}",vtype=GRB.CONTINUOUS) for j in range(self.P)]
        self.sigma_moins_x = [m.addVar(name=f"sigmax-_{j}",vtype=GRB.CONTINUOUS) for j in range(self.P)]
        self.sigma_moins_y = [m.addVar(name=f"sigmay-_{j}",vtype=GRB.CONTINUOUS) for j in range(self.P)]

        self.binary = [[m.addVar(vtype=GRB.BINARY, name=f"binary_{k}_{j}") for k in range(self.K)] for j in range(self.P)]

        m.update()
        
        return m

    def li(self,x):
        # print(x,mins[i],maxs[i])
        return int(self.L*x + 1)

    def xl(self,l):
        return l/self.L
    
    def u_i(self,i,x,k, evaluate = False):
        get_val = (lambda x: x.X) if evaluate else (lambda x: x)
        # print(j,i,X[j,i])   
        l = self.li(x[i])
        x_l = self.xl(l-1)
        x_l1 = self.xl(l)
        # print((self.criteria[k][i][l] + ((X[j, i]-x_l)/(x_l1-x_l))*(self.criteria[k][i][l+1]-self.criteria[k][i][l])))
        return (get_val(self.criteria[k][i][l-1]) + ((x[i]-x_l)/(x_l1-x_l))*(get_val(self.criteria[k][i][l])-get_val(self.criteria[k][i][l-1])))
    
    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
          
        for k in range(self.K):
            for j in range(self.P):
                x = X[j]
                y = Y[j]
                self.sum_utilite[(0,k,j)] = quicksum([self.u_i(i,x,k) for i in range(self.n)]) # X is equivalent to 0
                self.sum_utilite[(1,k,j)] = quicksum([self.u_i(i,y,k) for i in range(self.n)]) # Y is equivalent to 1
                # print(self.sum_utilite[(1,k,j)])
                # print(self.sum_utilite[(0,k,j)])
                # for i in range(self.n):
                #     print(u_i(j,i,X,k))
                #     self.criterion_utilite[(0,k,j,i)] = u_i(j,i,X,k) # X is equivalent to 0
                #     self.criterion_utilite[(1,k,j,i)] = u_i(j,i,Y,k) # Y is equivalent to 1

        
        # for k in range(self.K):
        #     for j in range(self.P):
        #         self.sum_utilite[(0,k, j)] = quicksum(self.criterion_utilite[0,k,j])
        #         self.sum_utilite[(1,k, j)] = quicksum(self.criterion_utilite[1,k,j])
         
        for j in range(self.P):
            for k in range(self.K):         
                self.model.addConstr(self.sum_utilite[0,k,j] - self.sigma_plus_x[j] + self.sigma_moins_x[j] - self.sum_utilite[1,k, j] + self.sigma_plus_y[j] - self.sigma_moins_y[j]>= -self.M*(1-self.binary[j][k])+self.epsilon)
                # self.model.addConstr(self.sum_utilite[0,k,j] - self.sigma_plus_x[j] + self.sigma_moins_x[j] - self.sum_utilite[1,k, j] + self.sigma_plus_y[j] - self.sigma_moins_y[j]<= self.M*self.binary[j][k] - self.epsilon )

        for k in range(self.K):
            for i in range(self.n):
                for l in range(self.L): # self.L-1? L?
                    self.model.addConstr(self.criteria[k][i][l+1] - self.criteria[k][i][l]>=0)#self.epsilon)
        
        for k in range(self.K):
            for i in range(self.n):
                self.model.addConstr(self.criteria[k][i][0] == 0)

        for k in range(self.K):
            self.model.addConstr(quicksum([self.criteria[k][i][self.L] for i in range(self.n)]) == 1)
        
        for j in range(self.P):
            self.model.addConstr(quicksum(self.binary[j]) >= 1)
        
        self.model.setObjective(quicksum(self.sigma_plus_x) + quicksum(self.sigma_moins_x) + quicksum(self.sigma_plus_y) + quicksum(self.sigma_moins_y), GRB.MINIMIZE)
        # self.model.setObjective(quicksum(self.sigma_plus_x[j] + self.sigma_moins_x[j] + self.sigma_plus_y[j] + self.sigma_moins_y[j] for j in range(self.P)), GRB.MINIMIZE)

        self.model.optimize()
        
        if self.model.status == GRB.INFEASIBLE:
            print("Pas de solution")
        elif self.model.status == GRB.UNBOUNDED:
            print("Non borné")
        else:
            print("Solution!")
        
        return self

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        P = []
        for x in X:
            K = []
            for k in range(self.K):
                K.append(sum([self.u_i(i,x,k, evaluate = True) for i in range(self.n)]))
            P.append(K)
        return np.array(P)



class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self):
        """Initialization of the Heuristic Model.
        """
        self.L = 5
        self.K = 3
        self.n = 10
        self.seed = 123
        self.epsilon = 0.001
        self.P = 40002
        self.criterion_utilite = {}
        self.sum_utilite = {}
        self.model = self.instantiate()

    def prior_cluster(self,X,Y):
        data =X-Y
        kmeans = KMeans(n_clusters=self.K, random_state=42)
        return  kmeans.fit_predict(data)
        
    def instantiate(self):
        """Instantiation of the MIP Variables - To be completed."""
        m = Model("Complex PL modelling")
        self.criteria = [[[m.addVar(name=f"u_{k}_{i}_{l}",vtype=GRB.CONTINUOUS, lb=0, ub=1) for l in range(self.L+1)] for i in range(self.n)] for k in range(self.K)]
        self.sigma_plus_x = [m.addVar(name=f"sigmax+_{j}",vtype=GRB.CONTINUOUS) for j in range(self.P)]
        self.sigma_plus_y = [m.addVar(name=f"sigmay+_{j}",vtype=GRB.CONTINUOUS) for j in range(self.P)]
        self.sigma_moins_x = [m.addVar(name=f"sigmax-_{j}",vtype=GRB.CONTINUOUS) for j in range(self.P)]
        self.sigma_moins_y = [m.addVar(name=f"sigmay-_{j}",vtype=GRB.CONTINUOUS) for j in range(self.P)]

        # self.binary = [[m.addVar(vtype=GRB.BINARY, name=f"binary_{k}_{j}") for k in range(self.K)] for j in range(self.P)]

        m.update()
        
        return m

    def li(self,x):
        # print(x,mins[i],maxs[i])
        return int(self.L*x + 1)

    def xl(self,l):
        return l/self.L
    
    def u_i(self,i,x,k, evaluate = False):
        get_val = (lambda x: x.X) if evaluate else (lambda x: x)
        # print(i,k) 
        if x[i] == 1:
            return get_val(self.criteria[k][i][-1]) 
        l = self.li(x[i])
        x_l = self.xl(l-1)
        x_l1 = self.xl(l)
        # print(x[i])
        # print(get_val(self.criteria[k][i][l]))
        return (get_val(self.criteria[k][i][l-1]) + ((x[i]-x_l)/(x_l1-x_l))*(get_val(self.criteria[k][i][l])-get_val(self.criteria[k][i][l-1])))
    
    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        clusters = self.prior_cluster(X,Y)
        for c,j in zip(clusters,range(self.P)):
            x = X[j]
            y = Y[j]
            self.sum_utilite[(0,c,j)] = quicksum([self.u_i(i,x,c) for i in range(self.n)]) # X is equivalent to 0
            self.sum_utilite[(1,c,j)] = quicksum([self.u_i(i,y,c) for i in range(self.n)]) # Y is equivalent to 1
         
        for c,j in zip(clusters,range(self.P)):       
            self.model.addConstr(self.sum_utilite[0,c,j] - self.sigma_plus_x[j] + self.sigma_moins_x[j] - self.sum_utilite[1,c, j] + self.sigma_plus_y[j] - self.sigma_moins_y[j]>=self.epsilon)
            # self.model.addConstr(self.sum_utilite[0,k,j] - self.sigma_plus_x[j] + self.sigma_moins_x[j] - self.sum_utilite[1,k, j] + self.sigma_plus_y[j] - self.sigma_moins_y[j]<= self.M*self.binary[j][k] - self.epsilon )

        for k in range(self.K):
            for i in range(self.n):
                for l in range(self.L): # self.L-1? L?
                    self.model.addConstr(self.criteria[k][i][l+1] - self.criteria[k][i][l]>=0)#self.epsilon)
        
        for k in range(self.K):
            for i in range(self.n):
                self.model.addConstr(self.criteria[k][i][0] == 0)

        for k in range(self.K):
            self.model.addConstr(quicksum([self.criteria[k][i][self.L] for i in range(self.n)]) == 1)
        
        
        self.model.setObjective(quicksum(self.sigma_plus_x) + quicksum(self.sigma_moins_x) + quicksum(self.sigma_plus_y) + quicksum(self.sigma_moins_y), GRB.MINIMIZE)
        # self.model.setObjective(quicksum(self.sigma_plus_x[j] + self.sigma_moins_x[j] + self.sigma_plus_y[j] + self.sigma_moins_y[j] for j in range(self.P)), GRB.MINIMIZE)

        self.model.optimize()
        
        if self.model.status == GRB.INFEASIBLE:
            print("Pas de solution")
        elif self.model.status == GRB.UNBOUNDED:
            print("Non borné")
        else:
            print("Solution!")
        
        return self

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        P = []
        for x in X:
            K = []
            for k in range(self.K):
                K.append(sum([self.u_i(i,x,k, evaluate = True) for i in range(self.n)]))
            P.append(K)
        return np.array(P)
