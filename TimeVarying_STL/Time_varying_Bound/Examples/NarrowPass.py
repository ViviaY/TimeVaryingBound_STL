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


T = 50
def test():

    x0 = np.array([3.0,3.6,0,0])
    goal = np.array([10.5, 6, 0., 0., ])


    # T = 50.
    vmax = 0.6
    amax = 0.2
    num_segs = 14
    rho_min = 0.15
    Q=1e3
    R=1e3
    W=1

    wall_half_width = 0.02
    A = np.array([[-1, 0], [1, 0], [0,-1], [0, 1]])
    walls = []

    walls.append(np.array([0., 0., 0., 12], dtype = np.float64))
    walls.append(np.array([12, 12, 0., 12], dtype = np.float64))
    walls.append(np.array([0, 12, 0, 0], dtype = np.float64))
    walls.append(np.array([0, 12, 12, 12], dtype = np.float64))

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

    # (xmin, xmax, ymin, ymax)
    obstacles = np.array([(-2,5,-4,6),
                        (-5.5,9,-3.8,5.7),
                        (-4.6,8,-0.5,3.5),
                        (-2.2,4.4,-6.4,10)], dtype = np.float64)


    charges = np.array([(-6,7,-7,8),
                        (-8.5,9.5,-2,3)], dtype = np.float64)


    bloat = 0.02
    s = np.array([0-bloat, 12-bloat,0-bloat, 12-bloat], dtype = np.float64)
    S = (A, s)

    charges_poly = [(A, charges[0]), (A, charges[1])]
    obs_poly = []
    for obs_ in obstacles:
        obs_poly.append((A, obs_))


    avoids = [Node('negmu', info={'A':A, 'b':b}) for A, b in obs_poly]
    avoid_obs = Node('and', deps=avoids)
    always_avoid_obs = Node('A', deps=[avoid_obs,], info={'int':[0,T]})

    InS = Node('mu', info={'A':S[0], 'b':S[1]})
    always_InS = Node('A', deps=[InS,], info={'int':[0,T]})


    reach_goal = [Node('mu', info={'A':A, 'b':b}) for A, b in charges_poly]
    OR_reach =  Node('or', deps=reach_goal)
    finally_reach_charg = Node('F', deps=[OR_reach,], info={'int':[0, T]})

    spec = Node('and', deps = [always_avoid_obs, always_InS, finally_reach_charg]) 
    specs = [spec,]
    x0s = [x0, ]
    goals = [goal, ]
    # goals = None
    

    # space constraints
    bloat = 0.01
    min_lim= np.array([0+bloat, 0+bloat],dtype =np.float64)
    max_lim= np.array([12-bloat, 12-bloat],dtype =np.float64)
    limits = [min_lim, max_lim]

    plots = [[charges_poly, 'g'], [obs_poly, 'r'], [obs, 'k']]


    cost_param=[Q, R, W]
    log_name = f"narrow_pass_BezierPoints_1norm_T{T}"
    solver = GurobiPlan(x0s, specs, limits, goals=goals, num_segs=num_segs, tmax=T, 
                        vel_max=vmax, acc_max=amax, rho_min=rho_min, cost_param=cost_param, 
                        MIPGap=0.1, log_name=log_name) 
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

        dataframe.to_csv(f"narrow_pass_BezierPoints_1norm_T{T}.csv",index=True,sep=',')

    
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


        dataframe.to_csv(f"narrow_pass_PWP_1norm_T{T}.csv",index=True,sep=',')
    

    
    return x0s, plots, PWP, rho, Bezier


if __name__ == '__main__':
    # x0s, plots, PWP = test()

    png_path =  f"narrow_pass_BezierPoints_1norm_T{T}.png"
    results = vis(test, png_path=png_path)
    
    # results = drone_vis(test)
