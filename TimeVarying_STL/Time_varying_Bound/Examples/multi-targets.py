import numpy as np
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobipy import GRB
import os
import sys
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
from Time_varying_Bound.GurobiSolver_BezierCurve_time_varying_1norm import GurobiPlan

from Time_varying_Bound.formula_time_varying import Node
from Time_varying_Bound.vis2D import vis as vis

import pandas as pd
import time


scale = 10.
T = 50
def test():

    x0 = scale*np.array([-0.8, 0.0, 0., 0., ])
    goal = scale*np.array([1.0, 0.0, 0., 0., ])

    vmax = 1
    amax = 1
    num_segs = 20
    rho_min = 0.3
    Q=1e3
    R=1e3
    W=1



    wall_half_width = 0.02
    A = np.array([[-1, 0], [1, 0], [0,-1], [0, 1]])
    walls = []

    walls.append(scale*np.array([-0.9, -0.9, -1., 1.], dtype = np.float64))
    walls.append(scale*np.array([1.1, 1.1, -1., 1.], dtype = np.float64))
    walls.append(scale*np.array([-0.9, 1.1, -1., -1.], dtype = np.float64))
    walls.append(scale*np.array([-0.9, 1.1, 1., 1.], dtype = np.float64))

    obs = []
    for wall in walls:
        if wall[0]==wall[1]:
            wall[0] -= wall_half_width
            wall[1] += wall_half_width
        elif wall[2]==wall[3]:
            wall[2] -= wall_half_width
            wall[3] += wall_half_width
        else:
            raise ValueError('wrong shape for axis-aligned wall')
        wall *= np.array([-1,1,-1,1])
        obs.append((A, wall))



    obstacles_centers = [
        [-0.35, -0.3],   # Obs1,1
        [-0.35, 0.3],   # Obs1,2
        [0.4, -0.3],  # Obs2,1
        [0.4, 0.3],  # Obs2,2
    ]

    min_lim = np.array([-0.2, -0.2], dtype=np.float64)
    max_lim = np.array([0.2, 0.2], dtype=np.float64)
    Bs = []
    
    for center in obstacles_centers:
        center = np.array(center, dtype=np.float64)
        x_center, y_center = center
        b = scale * np.array([
            -(x_center + min_lim[0]),  # 左边界
            (x_center + max_lim[0]),   # 右边界
            -(y_center + min_lim[1]),  # 下边界
            (y_center + max_lim[1])    # 上边界
        ], dtype=np.float64)
    
        Bs.append(b)


    obs_poly = []
    obs_poly.extend((A, b) for b in Bs)

    bloat = 0.01
    s = scale*np.array([0.9-bloat, 1.1-bloat, 1.-bloat, 1.-bloat], dtype = np.float64)
    S = (A, s)
    goal_poly = scale*np.array([-0.82, 1.02, 0.1, 0.1], dtype = np.float64)
    G = (A, goal_poly)

    # the locations of charges
    charging_stations_centers = [
        [0.02, -0.2],  # chg1,1
        [0.02, -0.4],  # chg1,2
        [0.02, 0.2],  # chg1,3
        [0.02, 0.4],  # chg1,4
        [0.75, -0.2],  # chg2,1
        [0.75, 0.2],  # chg2,2
        [0.75, 0.4],  # chg2,2
    ]
    min_lim = np.array([-0.085, -0.085], dtype=np.float64)
    max_lim = np.array([0.085, 0.085], dtype=np.float64)
    Cs = []
    for center in charging_stations_centers:
        center = np.array(center, dtype=np.float64)
        x_center, y_center = center
        b = scale * np.array([
            -(x_center + min_lim[0]),  
            (x_center + max_lim[0]),   
            -(y_center + min_lim[1]),  
            (y_center + max_lim[1])    
        ], dtype=np.float64)
    
        Cs.append(b)


    charging_stations = []
    charging_stations.extend((A, b) for b in Cs)

    avoid_obs = Node('and', deps=[Node('negmu', info={'A':A, 'b':b}) for A, b in obs_poly])
    phi_1 = Node('A', deps=[avoid_obs, ], info={'int':[0,T]})

    tc1 = 3.
    charge_1 = Node('or', deps=[Node('mu', info={'A':A, 'b':b}) for A,b in charging_stations[:4]])
    charging1_in_tc =  Node('A', deps=[charge_1], info={'int':[0, tc1]})
    phi_2 = Node('F', deps=[charging1_in_tc, ], info={'int':[0, T]})
    
    tc2 = 3
    charge_2 = Node('or', deps=[Node('mu', info={'A':A, 'b':b}) for A,b in charging_stations[4:]])
    charging2_in_tc =  Node('A', deps=[charge_2], info={'int':[0, tc2]})
    phi_3 = Node('F', deps=[charging2_in_tc, ], info={'int':[0, T]})

    InS = Node('mu', info={'A':S[0], 'b':S[1]})
    phi_4 = Node('A', deps=[InS,], info={'int':[0,T]})

    Ingoal = Node('mu', info={'A':G[0], 'b':G[1]})
    reach_goal = Node('F', deps=[Ingoal,], info={'int':[0,T]})
    spec = Node('and', deps=[phi_1, phi_2, phi_3, phi_4])



    x0s = [x0, ]
    specs = [spec, ]
    goals = [goal, ]
    # goals = None

    # space constraints
    bloat = 0.02
    min_lim= scale*np.array([-0.9+bloat, -1.+bloat],dtype =np.float64)
    max_lim= scale*np.array([1.1-bloat, 1.-bloat],dtype =np.float64)
    limits = [min_lim, max_lim]


    plots = [[obs_poly, 'r'], [charging_stations, 'g'],[obs, 'k']]#, [[G], 'y']]



    cost_param=[Q, R, W]
    log_name = f"multi_targets_BezierPoints_1norm_T{T}"
    solver = GurobiPlan(x0s, specs, limits, goals, num_segs=num_segs, tmax=T, MIPGap=0.3, # 0.1 if T = 20
                        vel_max=vmax, acc_max=amax, rho_min=rho_min, cost_param=cost_param, 
                        log_name=log_name)
    PWP, rho, Bezier = solver.Solve()

    if Bezier[0] is not None: 
        dataframe=pd.DataFrame(columns=['index', 'dt', 'rho', 'x', 'y']) 
        k = 0
        dt = T/num_segs
        for i in range(len(Bezier)):
            for j in range(len(Bezier[0])):
                data = np.array([i, dt, rho[i][0], Bezier[i][j][0],Bezier[i][j][1]], dtype=np.float64)
                dataframe.loc[k] = data
                k += 1

       

        dataframe.to_csv( f"multi_targets_BezierPoints_1norm_T{T}.csv",index=True,sep=',')



    if PWP[0] is not None: 
        dataframe=pd.DataFrame(columns=['index', 't', 'rho','x', 'y']) 
        k = 0
        dt = T/num_segs
        real_rho = []
        for i in range(len(PWP)):
            t = PWP[i][1]
            if i == 0: 
                data = np.array([i, t, rho[i][0], PWP[i][0][0], PWP[i][0][1]], dtype=np.float64)
                real_rho.append(rho[i][0])
            elif i == len(PWP)-1:
                rho_ = min(rho[i-2][0], rho[i-1][0])
                data = np.array([i, t, rho_, PWP[i][0][0], PWP[i][0][1]], dtype=np.float64)
                real_rho.append(rho_)
            else: 
                rho_ = min(rho[i-1][0], rho[i][0])
                data = np.array([i, t, rho_, PWP[i][0][0], PWP[i][0][1]], dtype=np.float64)
                real_rho.append(rho_)

            dataframe.loc[k] = data
            k += 1


        dataframe.to_csv(f"multi_targets_PWP_1norm_T{T}.csv",index=True,sep=',')
            


    
    return x0s, plots, PWP, rho, Bezier


if __name__ == '__main__':

    png_path =  f"multi_targets_BezierPoints_1norm_T{T}.png"
    results = vis(test, png_path=png_path)

