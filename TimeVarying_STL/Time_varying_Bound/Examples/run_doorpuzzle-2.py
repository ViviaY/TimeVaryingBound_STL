import sys
import numpy as np

# from PWLPlan import plan, Node
# from vis import vis

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


T = 50
scale = 2
def test():
    x0 = scale*np.array([5.0,4.5,0,0])

    # # T = 50.
    vmax = 3
    amax = 2
    num_segs = 25
    rho_min = 0.20
    Q=1e6
    R=1e6
    W=1


    A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

    walls = []
    wall_half_width = 0.01

    walls.append(scale*np.array([-0.01, -0.01, -0.01, 10.01], dtype = np.float64))
    walls.append(scale*np.array([15.01, 15.01, -0.01, 10.01], dtype = np.float64))
    walls.append(scale*np.array([-0.01, 15.01, -0.01, -0.01], dtype = np.float64))
    walls.append(scale*np.array([-0.01, 15.01,10.01, 10.01], dtype = np.float64))


    wall_bounds = []
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
        wall_bounds.append((A, wall))


    # Obstacle locations
    obs_bounds = scale*np.array([(8,15.01,-0.01,4), 
                            (8,15.01,6,10.01),
                            (3.5,5,-0.01,2.5),
                            (-0.01,2.5,4,6),
                            (3.5,5,7.5,10.01)], dtype = np.float64)

    # Door locations (set to overlap obstacles slightly)
    door_bounds = scale*np.array([(12.8,14,3.99,6.01),
                            (11.5,12.7,3.99,6.01),
                            (10.2,11.4,3.99,6.01),
                            (8.9,10.1,3.99,6.01)], dtype = np.float64)

    # key_bounds = np.array([(1,2,1,2),
    #                         (1,2,8,9),
    #                         (6,7,8,9),
    #                         (6,7,1,2)], dtype = np.float64)

    key_bounds = scale*np.array([(0.5,2.5,0.5,2.5),
                        (0.5,2.5,7.5,9.5),
                        (5.5,7.5,7.5,9.5),
                        (5.5,7.5,0.5,2.5)], dtype = np.float64)
    
    obs = []
    obs.extend((A, B*np.array([-1,1,-1,1])) for B in obs_bounds)
    doors = []
    doors.extend((A, B*np.array([-1,1,-1,1])) for B in door_bounds)
    keys = []
    keys.extend((A, B*np.array([-1,1,-1,1])) for B in key_bounds)


    avoid_obs = Node('and', deps=[Node('negmu', info={'A':A, 'b':b}) for A, b in obs])
    always_avoid_obs = Node('A', deps=[avoid_obs, ], info={'int':[0,T]})

    avoid_doors = [Node('negmu', info={'A':A, 'b':b}) for A, b in doors]
    pick_keys = [Node('mu', info={'A':A, 'b':b}) for A, b in keys]
    # pick_keys = [Node('A', deps=[pickkey, ], info={'int':[0,2]}) for pickkey in pickkeys]

    untils = [Node('U', deps=[avoid_door, pick_key], info={'int':[0,T]}) for avoid_door, pick_key in zip(avoid_doors, pick_keys)]
    
    goal_bounds = scale*np.array([14.1,14.9,4.1,5.9], dtype = np.float64)*np.array([-1,1,-1,1])
    goal =  (A, goal_bounds)
    reach_goal = Node('mu', info={'A':goal[0], 'b':goal[1]})
    finally_reach_goal = Node('F', deps=[reach_goal,], info={'int':[0,T]})

    spec = Node('and', deps = untils + [always_avoid_obs, finally_reach_goal]) 



    x0s = [x0,]
    # goals = [goal_, ]
    goals = None

    specs = [spec,]
    bloat = 0.001
    min_lim= scale*np.array([0+bloat, 0+bloat],dtype =np.float64)
    max_lim= scale*np.array([15-bloat, 10-bloat],dtype =np.float64)
    limits = [min_lim, max_lim]
    
    cost_param=[Q, R, W]
    log_name = f"run_doorpuzzle-2_BezierPoints_1norm_T{T}"
    solver = GurobiPlan(x0s, specs, limits, goals, num_segs=num_segs, tmax=T, 
                    vel_max=vmax, acc_max=amax, rho_min=rho_min, cost_param=cost_param, 
                    log_name=log_name, MIPGap=0.99)

    PWP, rho, Bezier = solver.Solve()
    plots = [[[goal,], 'b'], [keys, 'g'], [doors, 'r'], [obs, 'k'], [wall_bounds, 'k']]


    if Bezier[0] is not None: 
        dataframe=pd.DataFrame(columns=['index', 'dt', 'rho', 'x', 'y']) 
        k = 0
        dt = T/num_segs
        for i in range(len(Bezier)):
            for j in range(len(Bezier[0])):
                data = np.array([i, dt, rho[i][0], Bezier[i][j][0],Bezier[i][j][1]], dtype=np.float64)

                dataframe.loc[k] = data
                k += 1

        dataframe.to_csv(f"run_doorpuzzle-2_BezierPoints_1norm_T{T}.csv",index=True,sep=',')

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
        

        dataframe.to_csv(f"run_doorpuzzle-2_PWP_1norm_T{T}.csv",index=True,sep=',')



    return x0s, plots, PWP, rho, Bezier


if __name__ == '__main__':
    # png_path = 'run_doorpuzzle-2_BezierPoints.png'
    png_path =  f"run_doorpuzzle-2_BezierPoints_1norm_T{T}.png"

    results = vis(test, png_path=png_path)
