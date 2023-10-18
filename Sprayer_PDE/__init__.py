import numpy as np
np.set_printoptions(suppress=True, threshold=np.inf)
import pandas as pd
pd.set_option('display.float_format',lambda x : '%.2f' % x)
import pde
import itertools
import json
from scipy import interpolate
from scipy.stats import multivariate_normal

f2 = open('Sprayer_PDE/info.json', 'r')
info_data = json.load(f2)
sprayer_model_mean = [0.0, 0.0]

diffusivity_K = info_data["diffusion_k"]
# diffusivity_K = 10

sprayer_model_variance_num = info_data["sprayer_model_variance_num"]
sprayer_model_variance = [sprayer_model_variance_num, sprayer_model_variance_num]

lambda_S = info_data["sprayer_lambda"]
hyper_dt = 0.01
# lambda_S = 0.01

class Diffusion_Model():
    def __init__(self, diffusivity=1, x_range = 10, y_range = 10,\
                 initial_field = np.zeros((10, 10)), R_field = np.zeros((10, 10)), data_sprayer_train = [], t_start = 0):
        self.diffusivity = diffusivity
        self.x_range = x_range
        self.y_range = y_range
        self.grid = pde.UnitGrid([x_range, y_range])
        self.R_field = pde.ScalarField(grid = self.grid, data = R_field)
        self.initial_field = pde.ScalarField(grid = self.grid, data = initial_field)
        self.PHY_PDE = DiffusionPDE_withR(diffusivity=diffusivity, R = self.R_field, data_sprayer_train = data_sprayer_train, t_start = t_start)
        
    def solve(self, t):
        return self.PHY_PDE.solve(self.initial_field, t_range = t, dt = hyper_dt).data

class DiffusionPDE_withR(pde.PDEBase):
    def __init__(self, diffusivity = diffusivity_K, bc="auto_periodic_neumann", bc_laplace="auto_periodic_neumann"\
                    , R = 0, data_sprayer_train = [], t_start = 0):
        """ initialize the class with a diffusivity and boundary conditions
        for the actual field and its second derivative """
        self.sprayer_model_variance = sprayer_model_variance
        self.diffusivity = diffusivity
        self.bc = bc
        self.bc_laplace = bc_laplace
        self.R = R
        self.data_sprayer_train = data_sprayer_train
        self.t_start = t_start
        self.grid_x = self.R.data.shape[0]
        self.grid_y = self.R.data.shape[1]
        
    def evolution_rate(self, state, t=0):
        if int(t * 100) % 50 == 0:
            # print("t")
            # print(t)
            data_result = itertools.product([t + self.t_start],range(self.grid_x),range(self.grid_y))
            data_result = pd.DataFrame([x for x in data_result])
            self.S_S = self.sprayer(I = data_result)
            self.S_matrix = np.zeros((self.grid_x, self.grid_y))
            i_count = 0
            for i_x in range(self.grid_x):
                for i_y in range(self.grid_y):
                    self.S_matrix[i_x][i_y] = self.S_S[i_count]
                    i_count += 1
            self.S_data = lambda_S * np.multiply(np.square(np.clip(state.data, 0, 2 ** 12)), self.S_matrix)
            # print("state")
            # print(state.data.max())
            # print(state.data.min())
            # print("S_matrix")
            # print(self.S_matrix.max())
            # print(self.S_matrix.min())
            # print("S_data")
            # print(self.S_data.max())
            # print(self.S_data.min())
            # self.S_data = lambda_S * (state.data ** 2) * self.S_matrix
            self.S = pde.ScalarField(grid = pde.UnitGrid([self.grid_x, self.grid_y]), data = self.S_data)
        
        # print(self.S)
        # print(self.S.data)
        # S_field = state.grid
        # S_field = pde.ScalarField(grid = state.grid, data = S_field)
        """ numpy implementation of the evolution equation """
        state_lapacian = state.laplace(bc=self.bc)
        state_gradient = state.gradient(bc=self.bc)
        return (self.diffusivity * state_lapacian
                + self.R - self.S - 0.02*state)

    def sprayer(self, I):
        data_sprayer_train = self.data_sprayer_train
        I.columns = ["time","x_I","y_I"]
        I = I.assign(S = pd.Series([0 for j in range(len(I.index))]).values)
        for i in range(len(data_sprayer_train)):
            sprayer_z = pd.DataFrame(data_sprayer_train[i])
            # self.sprayer_z = sprayer_z
            if sprayer_z.shape[0] < 2:
                continue
                
            I_spray_compute = I.loc[(I.time < sprayer_z.time.max())\
                                      & (I.time >= sprayer_z.time.min()),:]
            I_spray_compute["x_drift"] = interpolate.interp1d(sprayer_z.time,sprayer_z["x"])(I_spray_compute.time) - I_spray_compute.x_I
            I_spray_compute["y_drift"] = interpolate.interp1d(sprayer_z.time,sprayer_z["y"])(I_spray_compute.time) - I_spray_compute.y_I
            I_spray_compute["spray_volume"] = interpolate.interp1d(sprayer_z.time,sprayer_z["spray_volume"])(I_spray_compute.time)
            I_spray_compute["S"] = multivariate_normal.pdf(I_spray_compute.loc[:,["x_drift","y_drift"]],\
                                           mean=np.array(sprayer_model_mean),\
                                           cov=np.diag(np.array(self.sprayer_model_variance)**2)) * I_spray_compute["spray_volume"]
            I.loc[I_spray_compute.index,"S"] = I.loc[I_spray_compute.index,"S"] + I_spray_compute["S"]
        # S = torch.tensor(I["S"], dtype=torch.float, requires_grad=True, device = device_name)
#         print(S)
        return I.S