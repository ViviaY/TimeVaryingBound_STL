import numpy as np
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobipy import GRB

# EPS = 1e-3
M = 1000
dt = 0.1
dl = 0.5
class Conjunction(object):
    # conjunction node
    def __init__(self, deps = []):
        super(Conjunction, self).__init__()
        self.deps = deps
        self.constraints = []
 
        

class Disjunction(object):
    # disjunction node
    def __init__(self, deps = []):
        super(Disjunction, self).__init__()
        self.deps = deps
        self.constraints = []
   


def mu(i, PWP, A, b, rho):
    num_edges = len(b)
    conjunctions = []
    for e in range(num_edges):
        a = A[e,:]
        for j in [i, i+1]:
            x = PWP[j][0]
            conjunctions.append(b[e] - np.linalg.norm(a) * (rho[i]) - sum([a[k]*(x[k]) for k in range(len(a))]))

    return Conjunction(conjunctions)

def negmu(i, PWP, A, b, rho):
    # this segment is outside Ax<=b (bloated)
    # b = b.reshape(-1)
    num_edges = len(b)
    disjunctions = []
    for e in range(num_edges):
        a = A[e,:]
        conjunctions = []
        for j in [i, i+1]:
            x = PWP[j][0]
            conjunctions.append(sum([a[k]*(x[k]) for k in range(len(a))]) - (b[e]+np.linalg.norm(a) * (rho[i])))
        
        disjunctions.append(Conjunction(conjunctions))
    return Disjunction(disjunctions)


def until(i, a, b, zphi1s, zphi2s, PWP):
    ti =  i * dt
    nf = int((ti + a)/dt)
    nr = int((ti + b)/dt)
    
    disjunction = []
    for n in range(nf, nr):
        conjunctions = []
        if n < len(PWP)-1:
            conjunctions.append(zphi2s[n])
            for k in range(i,n):
                conjunctions.append(zphi1s[k])
        disjunction.append(Conjunction(conjunctions))

    return Disjunction(disjunction)



def eventually(i, a, b, zphis, PWP):
    ti = i*dt
    nf = int((ti + a)/dt)
    nr = int((ti + b)/dt)
    disjunctions = []
    for j in range(nf, nr+1):
        if j < len(PWP)-1:
            disjunctions.append(zphis[j])
    return Disjunction(disjunctions)

def always(i, a, b, zphis, PWP):
    ti = i*dt
    nf = int((ti + a)/dt)
    nr = int((ti + b)/dt)+1
    conjuctions = []
    for j in range(nf, nr):
        if j < len(PWP)-1:
            conjuctions.append(zphis[j])        
    return Conjunction(conjuctions)

class Node(object):
    """docstring for Node"""
    def __init__(self, op, deps = [], zs = [], info = []):
        super(Node, self).__init__()
        self.op = op
        self.deps = deps
        self.zs = zs
        self.info = info

def clearSpecTree(spec):
    for dep in spec.deps:
        clearSpecTree(dep)
    spec.zs = []


def handleSpecTree(spec, PWP, size, rho):
    for dep in spec.deps:
        handleSpecTree(dep, PWP, size, rho)
    if len(spec.zs) == len(PWP)+1:
        return
    elif len(spec.zs) > 0:
        raise ValueError('incomplete zs')
    if spec.op == 'mu':
        spec.zs = [mu(i, PWP, spec.info['A'], spec.info['b'], rho) for i in range(len(PWP)-1)]
    elif spec.op == 'negmu':
        spec.zs = [negmu(i, PWP, spec.info['A'], spec.info['b'], rho) for i in range(len(PWP)-1)]
    elif spec.op == 'and':
        spec.zs = [Conjunction([dep.zs[i] for dep in spec.deps]) for i in range(len(PWP)-1)]
    elif spec.op == 'or':
        spec.zs = [Disjunction([dep.zs[i] for dep in spec.deps]) for i in range(len(PWP)-1)]
    elif spec.op == 'U':
        spec.zs = [until(i, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, spec.deps[1].zs, PWP) for i in range(len(PWP)-1)]
    elif spec.op == 'F':
        spec.zs = [eventually(i, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, PWP) for i in range(len(PWP)-1)]
        # spec.zs = [eventually(0, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, PWP)]
    elif spec.op == 'A':
        spec.zs = [always(i, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, PWP) for i in range(len(PWP)-1)]
    else:
        raise ValueError('wrong op code')

def add_CDTree_Constraints(model, root, rho):
    constrs = gen_CDTree_constraints(model, root, rho)
    for con in constrs:
        model.addConstr(con >= 0)

def gen_CDTree_constraints(model, root, rho):
    if not hasattr(root, 'deps'):
        return [root,]
    else:
        if len(root.constraints)>0:
            return root.constraints
        dep_constraints = []
        for dep in root.deps:
            dep_constraints.append(gen_CDTree_constraints(model, dep, rho))
        
        zs = []
        # depcon = []
        for dep_con in dep_constraints:
            if isinstance(root, Disjunction):
                z = model.addVar(vtype=GRB.BINARY)
                zs.append(z) 
                dep_con = [con + M * (1 - z) for con in dep_con]
            root.constraints += dep_con
        if len(zs)>0:
            root.constraints.append(sum(zs)-1)
            
        model.update()
        return root.constraints