import numpy as np

import gurobipy as gp
from gurobipy import GRB, quicksum
# import formula 
import time

import os
import sys
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_file_path)))

import Time_varying_Bound.formula_time_varying as formula 
import math


class GurobiPlan(object):

    def __init__(self, x0s, specs, limits, goals, tmax=15., acc_max=0.1, vel_max=0.5, 
        rho_min=0.015, num_segs=15, M=1e3, cost_param=[1e1, 1e2, 1e3],MIPGap=1e-3,
        presolve=True, verbose=True, log_name="gurobi_output.log"): # tmax=tmax, vmax=vmax, goals=goals, 
        assert M > 0, "M should be a (large) positive scalar"

        self.M = float(M)
        self.presolve = presolve
        self.num_segs = num_segs
        self.num_states = num_segs+1
        self.dims = 2 # x,y
        self.order = 6
        self.x0 = x0s[0]
        self.limits = limits
        self.specs = specs
        self.T = tmax
        
        self.verbose = verbose
        self.presolve = presolve
        self.dt = tmax/num_segs
        self.rho_min = rho_min
        self.vel_max = vel_max
        self.acc_max = acc_max
        self.Q, self.R, self.W = cost_param



        if goals is None:
            self.goal = None
        else: 
            self.goal = goals[0]
        # Set up the optimization problem
        self.model = gp.Model("STL_MICP")
        self.model.Params.NonConvex = 2
        # 设置日志文件路径
        log_file = log_name+".log"
        self.model.setParam('LogFile', log_file)


        self.model.setParam('PreCrush', 1)
        self.model.setParam('Presolve', 2)
        self.model.setParam("MIPFocus", 1) 
        self.model.setParam("NumericFocus", 1)
        self.model.setParam("ScaleFlag", 1)
        self.model.setParam("Cuts", 2)

        self.model.setParam('Heuristics', 0.15)
        self.model.setParam('FeasibilityTol', 1e-9)
        T_MIN_SEP = 1e-1
        IntFeasTol  = 1e-1 * T_MIN_SEP /self.M
        self.model.setParam(GRB.Param.IntFeasTol, IntFeasTol)
        self.model.setParam(GRB.Param.MIPGap, MIPGap)


        # Store the cost function, which will added to self.model right before solving
        self.cost = 0.0

        # Set some model parameters
        if not self.presolve:
            self.model.setParam('Presolve', 0)
        if not self.verbose:
            self.model.setParam('OutputFlag', 0)

        
        if self.verbose:
            print("Setting up optimization problem...")
            st = time.time()  # for computing setup time

        # Create optimization variables
        self.PWP = [] # end points of Bezier Curves: x,y,t
        for i in range(self.num_states):
            self.PWP.append([self.model.addVars(self.dims, lb=-GRB.INFINITY), self.model.addVar()]) 

        # Create optimization variables
        self.Bezier = [] # control points of the Bezier curves
        for i in range(self.num_segs):
            self.Bezier.append([self.model.addVars(self.dims, lb=-GRB.INFINITY) for i in range(self.order+1)])       

        self.acc = [] # inputs control  = [ddx, ddy]
        for i in range(self.num_states-1):
            self.acc.append(self.model.addVars(self.dims, lb=-GRB.INFINITY))
        
        self.vel = [] # inputs control  = [ddx, ddy]
        for i in range(self.num_states-1):
            self.vel.append(self.model.addVars(self.dims, lb=-GRB.INFINITY))

        self.rho = self.model.addVars(self.num_segs, name="rho", lb=0.0)  #lb sets minimu m robustness
        self.rho_Bezier = self.model.addVars(self.num_segs, name="rho_Bezier", lb=0.0)  #lb sets minimu m robustness
        self.deviation_a = self.model.addVars(self.num_segs, name="deviation_a", lb=0.0)  #lb sets minimu m robustness
        self.deviation_v = self.model.addVars(self.num_segs, name="deviation_v", lb=0.0)  #lb sets minimu m robustness

        



        self.model.update()

        # Add cost and constraints to the optimization problem
        self.GenerateBezier()
        self.AddRobustnessConstraint(rho_min=rho_min) 
        self.AddStateBounds()
        self.AddConvexCost(Q=self.Q, R=self.R)
        self.AddRobustnessCost(W=self.W)
        self.AddDynamicsConstraintsFixedDt()

        size = 0.0
        formula.dt = self.dt
        for spec in self.specs:
            formula.clearSpecTree(spec)
        formula.handleSpecTree(self.specs[0], self.PWP, size, self.rho)
        formula.add_CDTree_Constraints(self.model, self.specs[0].zs[0], self.rho)

        if self.verbose:
            print(f"Setup complete in {time.time()-st} seconds.")



    # add the bounds of positionsx and y
    def AddStateBounds(self):
        for n in range(self.num_segs):
            for k in range(self.order+1): 
                for i in range(2):
                    self.model.addConstr(self.Bezier[n][k][i] >= self.limits[0][i], name=f"position>-p_min_{n}_{k}_{i}")
                    self.model.addConstr(self.Bezier[n][k][i] <= self.limits[1][i], name=f"position<p_max_{n}_{k}_{i}")

    def AddRobustnessConstraint(self, rho_min=0.015): 
        for i in range(self.num_segs):
            self.model.addConstr(self.rho_Bezier[i] >= rho_min, name=f"robustConstr")

        
    def AddRobustnessCost(self, W=1e5):
        for i in range(self.num_segs):
            self.cost += -self.rho_Bezier[i]*W

    def AddConvexCost(self,Q=1e2, R = 1e2):

        self.vel_abs = [] # inputs control  = [ddx, ddy]
        self.acc_abs = [] # inputs control  = [ddx, ddy]

        for i in range(self.num_states-1):
            self.vel_abs.append(self.model.addVars(self.dims, lb=0))
            self.acc_abs.append(self.model.addVars(self.dims, lb=0))

        self.model.update()


        for n in range(self.num_segs):
            for i in range(self.dims):
                self.model.addConstr(self.vel_abs[n][i] == gp.abs_(self.vel[n][i]), name=f"vel_abs_pos_{n}_{i}")
                self.model.addConstr(self.acc_abs[n][i] == gp.abs_(self.acc[n][i]), name=f"acc_abs_pos_{n}_{i}")
    
                self.model.addConstr(self.vel_abs[n][i] <= self.vel_max, name=f"vel<vel_max_{n}")

                self.model.addConstr(self.acc_abs[n][i] <= 8*self.deviation_a[n]/(math.sqrt(self.order)*self.dt**2), name=f"acc_m>acc_min_{n}")

                self.model.addConstr(self.acc_abs[n][i] <= self.acc_max, name=f"acc<acc_max_{n}")

            self.model.addConstr(self.rho_Bezier[n] == self.rho[n] - self.deviation_a[n], name=f"rho&de")      


                

        for n in range(self.num_segs):
            for i in range(2):
                # self.cost +=  (self.vel[n][i])*Q*(self.vel[n][i])
                self.cost +=  (self.vel_abs[n][i])*Q

        for n in range(self.num_states-1):
            for i in range(2):
                # self.cost += (self.acc[n][i])*R*(self.acc[n][i])
                self.cost += (self.acc_abs[n][i])*R
        

    def GenerateBezier(self):
        n = self.order
        if self.goal is None:
            print("the goal states is not provided.")
        else:
            self.model.addConstrs((self.Bezier[self.num_segs-1][n][k] == self.goal[k] for k in range(self.dims)), name="goal")
        self.model.addConstrs((self.Bezier[0][0][k] == self.x0[k] for k in range(self.dims)), name="x0")
        # if the initial and the final velocity are 0
        # self.model.addConstrs((self.Bezier[0][1][k] == self.Bezier[0][0][k] for k in range(self.dims)), name="v0")
        # self.model.addConstrs((self.Bezier[self.num_segs-1][n][k]==self.Bezier[self.num_segs-1][n-1][k] for k in range(self.dims)), name="vf")


        for i in range(self.num_segs): 

            for j in range(n):
                # # constraints for velocity
                if j == 0 or j==n-1:
                    self.model.addConstrs((self.Bezier[i][j+1][k] - self.Bezier[i][j][k] <= self.acc[i][k]*self.dt**2/(n) for k in range(self.dims)), name=f'constr_vel_{i}{0}')
                    self.model.addConstrs((self.Bezier[i][j+1][k] - self.Bezier[i][j][k] >= -self.acc[i][k]*self.dt**2/(n) for k in range(self.dims)), name=f'constr_vel_{i}{0}')
                else:
                    self.model.addConstrs((self.Bezier[i][j+1][k] - self.Bezier[i][j][k] <= self.vel[i][k]*self.dt/(n) for k in range(self.dims)), name=f'constr_vel0_{i}{j}')
                    self.model.addConstrs((self.Bezier[i][j+1][k] - self.Bezier[i][j][k] >= -self.vel[i][k]*self.dt/(n) for k in range(self.dims)), name=f'constr_vel1_{i}{j}')
        

            for j in range(n-1):
                # constraints for accelarations
                self.model.addConstrs((self.Bezier[i][j+2][k] - 2*self.Bezier[i][j+1][k] + self.Bezier[i][j][k] 
                                        <= self.acc[i][k]*self.dt**2/((n-1) *(n)) for k in range(self.dims)), name=f'constr_acc0_{j}')

                self.model.addConstrs((self.Bezier[i][j+2][k] - 2*self.Bezier[i][j+1][k] + self.Bezier[i][j][k]
                                        >= -self.acc[i][k]*self.dt**2/((n-1)*(n)) for k in range(self.dims)), name=f'constr_acc1_{j}')


            # for the continuity: 
            if i != self.num_segs-1:
                # for C^0: 
                self.model.addConstrs((self.Bezier[i][n][k] == self.Bezier[i+1][0][k] for k in range(self.dims)),name=f'constr_C^1_{i}')
                # for C^1: p^k_{n-2} - p^k_{n-1} = p^{k+1}_{1} - p^{k+1}_{0}
                self.model.addConstrs(((self.Bezier[i][n][k] - self.Bezier[i][n-1][k])
                                         == (self.Bezier[i+1][1][k] - self.Bezier[i+1][0][k]) for k in range(self.dims)),name=f'constr_C^1_{i}')
    
                # for C^2: p^k_{n-2}-2*p^k_{n-1}+ p^k_n= p^{k+1}_2-2*p^{k+1}_1 + p^{k+1}_0
                self.model.addConstrs(((self.Bezier[i][n-2][k] - 2*self.Bezier[i][n-1][k] + self.Bezier[i][n][k])
                                        == (self.Bezier[i+1][2][k] - 2*self.Bezier[i+1][1][k] + self.Bezier[i+1][0][k]) for k in range(self.dims)),name=f'constr_C^2_{i}')

    def AddDynamicsConstraintsFixedDt(self):
        n = self.order+1
        for k in range(self.num_segs):
            self.model.addConstrs((self.PWP[k][0][i] ==  self.Bezier[k][0][i] for i in range(2)), name="PWP_{k}")
            self.model.addConstr((self.PWP[k][1] == k*self.dt), name="PWP_time_{k}")

        self.model.addConstrs((self.PWP[self.num_states-1][0][i] ==  self.Bezier[self.num_segs-1][n-1][i] for i in range(2)), name="PWP_end")
        self.model.addConstr((self.PWP[self.num_states-1][1] == self.num_segs*self.dt), name="PWP_time_end")


    def Solve(self):
        # Set the cost function now, right before we solve.
        # This is needed since model.setObjective resets the cost.
        self.model.setObjective(self.cost, GRB.MINIMIZE) #GRB.MAXIMIZE  

        # Do the actual solving
        self.model.update()
        self.model.optimize()

        if self.model.Status in [GRB.TIME_LIMIT, GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
            try:
                # compute the IIS
                self.model.computeIIS()
                # write the IIS to a file
                self.model.write("model.ilp")
                # get the constraints and variable bounds in the IIS
                cst_num = 0
                constrain_list = self.model.getConstrs()
                for c in self.model.getConstrs():
                    if c.IISConstr:
                        print("Constraint", c, "is in the IIS")
                        print(self.model.getRow(c), c.sense, c.rhs)
                    for v in self.model.getVars():
                        if v.IISLB > 0:
                            print("Variable", v, "lower bound is in the IIS")
                        if v.IISUB > 0:
                            print("Variable", v, "upper bound is in the IIS")
                self.model.dispose()
            except GurobiError as e:
                print("Error code " + str(e.errno) + ": " + str(e))
                self.model.dispose()

        
        traj_output = []
        acc_output = []
        vel_output = []
        Bezier_output = []
        final_cost = 0.
        Bezier_rho_output = []
        rho_output = []
        deviation_output = []
        mode_mode_output = []
        if self.model.status == GRB.OPTIMAL:
            self.model.printQuality()
            # if self.verbose:
            #     print("\n Optimal Solution Found! \n")
            for Z in self.PWP:
                traj_output.append([[Z[0][i].X for i in range(len(Z[0]))], Z[1].X])

            for B in self.Bezier:
                Bezier_output.append([[B[k][i].X for i in range(2)] for k in range(self.order+1)])

            for i in range(self.num_segs):
                Bezier_rho_output.append([self.rho_Bezier[i].X])

            for i in range(self.num_segs):
                rho_output.append([self.rho[i].X])

            # for i in range(self.num_segs):
            #     deviation_output.append([self.deviation[i].X])

                
            for n in range(self.num_states-1):
                acc_output.append([self.acc[n][i].X for i in range(2)])
                vel_output.append([self.vel[n][i].X for i in range(2)])

            # rho_output = self.rho.X
            # Bezier_rho_output = self.rho_Bezier.X
            # deviation_output = self.deviation.X
            final_cost = self.model.objVal
            print(f'Optimal Cost: {final_cost}')
            # print("acc_output = ", acc_output)
            # print("vel_output = ", vel_output)

            print("Bezier_rho_output = ", Bezier_rho_output)
            # print("deviation_output = ", deviation_output)
            print("rho_output = ", rho_output)
            self.model.dispose()



        else:
            # if self.verbose:
            #     print(f"\nOptimization failed with status {self.model.status}.\n")
            flagz_output = None
            rho_output = -np.inf
            self.model.dispose()

        return traj_output, Bezier_rho_output, Bezier_output


 