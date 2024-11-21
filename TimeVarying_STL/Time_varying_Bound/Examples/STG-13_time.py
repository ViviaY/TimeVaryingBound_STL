import numpy as np
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobipy import GRB

# from STL import plan, Node
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
    # x0 = [-1., -1, -1]
    x0 = np.array([scale*-0.6, scale*-0.6, 0., 0., ]) # x,y,vx, vy
    goal = np.array([scale*0.6, scale*0.6, 0., 0., ])


    # T = 50
    vmax = 1.
    amax = 1.
    num_segs = 15
    rho_min= 0.4
    # cost weights
    Q=1e4
    R=1e4
    W=1

    wall_half_width = 0.02
    A = np.array([[-1, 0], [1, 0], [0,-1], [0, 1]])
    walls = []

    walls.append(scale*np.array([-0.75, -0.75, -0.75, 0.75], dtype = np.float64))
    walls.append(scale*np.array([0.75, 0.75, -0.75, 0.75], dtype = np.float64))
    walls.append(scale*np.array([-0.75, 0.75, -0.75, -0.75], dtype = np.float64))
    walls.append(scale*np.array([-0.75, 0.75, 0.75, 0.75], dtype = np.float64))

    # walls.append(scale*np.array([-0.7, -0.7, -0.8, 0.8], dtype = np.float64))
    # walls.append(scale*np.array([0.7, 0.7, -0.8, 0.8], dtype = np.float64))
    # walls.append(scale*np.array([-0.7, 0.7, -0.8, -0.8], dtype = np.float64))
    # walls.append(scale*np.array([-0.7, 0.7, 0.8, 0.8], dtype = np.float64))

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


    b1 = scale*np.array([0.5, -0.1, 0.5, -0.2], dtype = np.float64)
    B1 = (A, b1)
    b2 = scale*np.array([-0.1, 0.55, 0.5, -0.1], dtype = np.float64)
    B2 = (A, b2)
    b3 = scale*np.array([-0.1, 0.5, -0.25, 0.5], dtype = np.float64)
    B3 = (A, b3)
    b4 = scale*np.array([0.55, -0.05, 0.05, 0.35], dtype = np.float64)
    B4 = (A, b4)
    bloat = 0.01
    s = scale*np.array([0.75-bloat,0.75-bloat, 0.75-bloat, 0.75-bloat], dtype = np.float64)
    S = (A, s)


    plots = [[[B1,], 'y'], [[B2,], 'r'], [[B3,], 'g'], [[B4,], 'b'], [obs, 'k']]

    notB4 = Node('negmu', info={'A':B4[0], 'b':B4[1]})
    notB1 = Node('negmu', info={'A':B1[0], 'b':B1[1]})
    notB2 = Node('negmu', info={'A':B2[0], 'b':B2[1]})
    B3 = Node('mu', info={'A':B3[0], 'b':B3[1]})
    B2 = Node('mu', info={'A':B2[0], 'b':B2[1]})
    InS = Node('mu', info={'A':S[0] , 'b':S[1]})


    phi_1 = Node('F', deps=[Node('A', deps=[B2,], info={'int':[0,2]}),], info={'int':[0,T-2]})
    phi_2 = Node('F', deps=[Node('A', deps=[B3,], info={'int':[0,2]}),], info={'int':[0,T-2]})
    phi_3 = Node('A', deps=[notB4,], info={'int':[0,T]})
    phi_4 = Node('A', deps=[notB1,], info={'int':[0,T]})
    phi_5 = Node('A', deps=[InS,], info={'int':[0,T]})
    spec = Node('and', deps=[phi_1, phi_2, phi_3, phi_4, phi_5]) 

    x0s = [x0, ]
    specs = [spec, ]
    goals = [goal, ]

    # space constraints
    bloat = 0.01
    min_lim= scale*np.array([-0.75+bloat, -0.75+bloat],dtype =np.float64)
    max_lim= scale*np.array([0.75-bloat, 0.75-bloat],dtype =np.float64)
    limits = [min_lim, max_lim]


    cost_param=[Q, R, W]
    log_name = f"stlcg-1_3d_time_varying_BezierPoints_1norm_T{T}"

    solver = GurobiPlan(x0s, specs, limits, goals, num_segs=num_segs, tmax=T, 
                        vel_max=vmax, acc_max=amax, rho_min=rho_min, cost_param=cost_param,log_name=log_name)

    time_start=time.time()
    PWP, rho, Bezier = solver.Solve()
    time_end=time.time()
    # print('time cost',time_end-time_start,'s')
    # print("PWP = ", PWP)
    # print("Bezier = ", Bezier)

    if Bezier[0] is not None: 
        dataframe=pd.DataFrame(columns=['index', 'dt', 'rho', 'x', 'y']) 
        k = 0
        dt = T/num_segs
        for i in range(len(Bezier)):
            for j in range(len(Bezier[0])):
                data = np.array([i, dt, rho[i][0], Bezier[i][j][0],Bezier[i][j][1]], dtype=np.float64)
                dataframe.loc[k] = data
                k += 1
        

        dataframe.to_csv( f"stlcg-1_3d_time_varying_BezierPoints_1norm_T{T}.csv",index=True,sep=',')


    if PWP[0] is not None: 
        dataframe=pd.DataFrame(columns=['index', 't', 'rho','x', 'y']) 
        k = 0
        dt = T/num_segs
        
        for i in range(len(PWP)):
            t = PWP[i][1]
            if i == 0: 
                data = np.array([i, t, rho[i][0], PWP[i][0][0], PWP[i][0][1]], dtype=np.float64)
            elif i == len(PWP)-1:
                data = np.array([i, t, rho[i-1][0], PWP[i][0][0], PWP[i][0][1]], dtype=np.float64)
            else: 
                rho_ = min(rho[i-1][0], rho[i][0])
                data = np.array([i, t, rho_, PWP[i][0][0], PWP[i][0][1]], dtype=np.float64)

            dataframe.loc[k] = data
            k += 1
        

        dataframe.to_csv(f"stlcg-1_3d_time_varying_PWP_1norm_T{T}.csv",index=True,sep=',')

    print("**************** verifying the robustness of the Bezier Curves ****************")
    # verify(Bezier, T=T, scale=scale, dt=T/num_segs, IsBezier=True)

    ## print("**************** verifying the robustness of the linear segments ****************")
    ## verify(PWP, T=T, scale=scale,dt=T/num_segs, IsBezier=False)



    return x0s, plots, PWP, rho, Bezier


if __name__ == '__main__':
    png_path =  f"stlcg-1_3d_time_varying_BezierPoints_1norm_T{T}"

    results = vis(test,png_path) 

