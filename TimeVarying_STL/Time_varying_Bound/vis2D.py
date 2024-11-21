from gurobipy import *
import numpy as np
from collections import namedtuple
import time
import pickle
import pypoman as ppm
import matplotlib.pyplot as plt; 
import matplotlib.path as path
import matplotlib.lines as Lines
import matplotlib.patches as patches
import scipy.special

def bernstein_poly(n, i, t):
    """
    Bernstein polynom.

    :param n: (int) polynom degree
    :param i: (int)
    :param t: (float)
    :return: (float)
    """
    return scipy.special.comb(n, i) * t ** i * (1 - t) ** (n - i)

def bezier(t, control_points):
    """
    Return one point on the bezier curve.

    :param t: (float) number in [0, 1]
    :param control_points: (numpy array)
    :return: (numpy array) Coordinates of the point
    """
    n = len(control_points) - 1
    return np.sum([bernstein_poly(n, i, t) * control_points[i] for i in range(n + 1)], axis=0)

def calc_bezier_path(control_points, n_points=100):
    """
    Compute bezier path (trajectory) given control points.

    :param control_points: (numpy array)
    :param n_points: (int) number of points in the trajectory
    :return: (numpy array)
    """
    traj = []
    for t in np.linspace(0, 1, n_points):
        traj.append(bezier(t, control_points))

    return np.array(traj)


def plot_line_Triangle(ax, vertices, color='red'):
    vertices = np.vstack([vertices, vertices[0]])
    plt.plot(vertices[:, 0], vertices[:, 1], color=color)

def plot_rectangle(ax, vertices, color='red'):

    vertices = np.concatenate(np.array([vertices],dtype=object), axis=0)
    x, y= vertices.min(axis=0)
    dx, dy = vertices.max(axis=0) - vertices.min(axis=0)
    kwargs = {'alpha': 1, 'color': color}
    rectangle = plt.Rectangle((x, y), dx, dy, fill=None, edgecolor=color, linewidth=2)
    ax.add_patch(rectangle)



def vis(test, png_path=None, limits=None, equal_aspect=True):
    _, plots, PWPs, rho, Bezier = test()

    # print(PWPs)
    plt.rcParams["figure.figsize"] = [6.4, 6.4]
    plt.rcParams['axes.titlesize'] = 20
    fig,ax = plt.subplots()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 移除图表周围的空白
    fig.add_axes(ax)
    ax.axis('off')

    vertices = []
    for plot in plots:
        for A, b in plot[0]:
            vs = ppm.duality.compute_polytope_vertices(A, b) 
            if len(vs) == 3:
                plot_line_Triangle(ax, vs, color=plot[1])
                vertices.append(vs)
            else:
                plot_rectangle(ax, vs, color=plot[1])
                vertices.append(vs)


    if limits is not None:
        ax.xlim(limits[0])
        ax.ylim(limits[1])
        # ax.zlim(limits[2])
    else:
        vertices = np.concatenate(vertices, axis=0)
        xmin, ymin = vertices.min(axis=0)
        xmax, ymax = vertices.max(axis=0)
        ax.set_xlim([xmin - 0.1, xmax + 0.1])
        ax.set_ylim([ymin - 0.1, ymax + 0.1])

    if equal_aspect:
        plt.gca().set_aspect('equal', adjustable='box')

    if PWPs is None or len(PWPs) == 0: #PWPs[0] is None: #  or 
        plt.show()
        return

    if len(PWPs) <= 4:
        colors = ['k', np.array([153,0,71],dtype=object)/255, np.array([6,0,153],dtype=object)/255, np.array([0, 150, 0],dtype=object)/255]
    else:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in np.linspace(0, 0.85, len(PWPs))]


    # ax.plot([PWPs[i][0][0] for i in range(len(PWPs))], [PWPs[i][0][1] for i in range(len(PWPs))],'.-', color = colors[1])
    
    ax.plot(PWPs[-1][0][0], PWPs[-1][0][1], '*', color = 'g')
    ax.plot(PWPs[0][0][0], PWPs[0][0][1], 'o', color = 'g')

    #Plot Bezier cuves
    for j in range(len(Bezier)):
        cp = np.array(Bezier[j])
        points = [cp[0], cp[-1]]
        path = calc_bezier_path(cp)
        # ax.plot(path.T[0], path.T[1], label="Bezier Path", color=colors[j])
        ax.plot(path.T[0], path.T[1], label="Bezier Path", color='b')

        # ax.plot(cp.T[0], cp.T[1],'--o', label="Control Points", color=colors[j])
        # ax.plot(np.array(points).T[0], np.array(points).T[1],'--o', label="data Points")
        
        # the bounding box
        # xmin = np.min(cp.T[0])
        # ymin = np.min(cp.T[1])
    
        # xmax = np.max(cp.T[0])
        # ymax = np.max(cp.T[1])

        # vertices = [[xmin, ymin], [xmax, ymax]]
        # plot_rectangle(ax, vertices)

            
    plt.show()


     # Save the chart as PDF and specify the DPI
    if png_path != None: 
        save_png = png_path + ".png"
        fig.savefig(save_png, format='png', dpi=None)  # 
        save_eps = png_path + ".eps"
        fig.savefig(save_eps, format='eps', dpi=None)  #  





def vis_file(test, png_path=None, limits=None, equal_aspect=True):
    _, plots, PWPs, rho, Bezier, track_path = test()

    # print(PWPs)
    # plt.rcParams["figure.figsize"] = [6.4, 6.4]
    plt.rcParams["figure.figsize"] = [9.4, 6.4]
    plt.rcParams['axes.titlesize'] = 20
    fig,ax = plt.subplots()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 移除图表周围的空白
    fig.add_axes(ax)
    ax.axis('off')

    vertices = []
    for plot in plots:
        for A, b in plot[0]:
            vs = ppm.duality.compute_polytope_vertices(A, b) 
            if len(vs) == 3:
                plot_line_Triangle(ax, vs, color=plot[1])
                vertices.append(vs)
            else:
                plot_rectangle(ax, vs, color=plot[1])
                vertices.append(vs)


    if limits is not None:
        ax.xlim(limits[0])
        ax.ylim(limits[1])
        # ax.zlim(limits[2])
    else:
        vertices = np.concatenate(vertices, axis=0)
        xmin, ymin = vertices.min(axis=0)
        xmax, ymax = vertices.max(axis=0)
        ax.set_xlim([xmin - 0.1, xmax + 0.1])
        ax.set_ylim([ymin - 0.1, ymax + 0.1])

    if equal_aspect:
        plt.gca().set_aspect('equal', adjustable='box')

    if PWPs is None or len(PWPs) == 0: #PWPs[0] is None: #  or 
        plt.show()
        return

    if len(PWPs) <= 4:
        colors = ['k', np.array([153,0,71],dtype=object)/255, np.array([6,0,153],dtype=object)/255, np.array([0, 150, 0],dtype=object)/255]
    else:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in np.linspace(0, 0.85, len(PWPs))]


    # ax.plot([PWPs[i][0][0] for i in range(len(PWPs))], [PWPs[i][0][1] for i in range(len(PWPs))],'.-', color = colors[1])
    
    # ax.plot(PWPs[-1][0], PWPs[-1][1], '*', color = 'g')
    # ax.plot(PWPs[0][0], PWPs[0][1], 'o', color = 'g')

    Bezier_line_style = {
    'color': 'blue',
    'linewidth': 3,
    'linestyle': '-',
    'zorder':0
    # 'linestyle': (0, (3, 2))
    # 'marker': 'o',
    # 'markersize': 10,
    # 'markerfacecolor': 'blue'
    # 'alpha': 0.6          
    }

    Bezier_endcps_style = {
    'c': 'blue',          
    's': 30,             
    'marker': 'o',        
    'alpha': 0.7,          
    'zorder':2
    }

    tracking_line_style = {
    'color': 'red',
    'linewidth': 3.5,
    # 'linestyle': '-',
    'linestyle': (0, (3, 2)),  
    'zorder':0
    # 'marker': 'o',
    # 'markersize': 10,
    # 'markerfacecolor': 'blue'
    }


  
        # the bounding box
        # xmin = np.min(cp.T[0])
        # ymin = np.min(cp.T[1])
    
        # xmax = np.max(cp.T[0])
        # ymax = np.max(cp.T[1])

        # vertices = [[xmin, ymin], [xmax, ymax]]
        # plot_rectangle(ax, vertices)

    
        #Plot Bezier cuves
    for j in range(len(Bezier)):
        cp = np.array(Bezier[j])
        points = [cp[0], cp[-1]]
        path = calc_bezier_path(cp)
        # ax.plot(path.T[0], path.T[1], label="Bezier Path", color=colors[j])
        ax.plot(path.T[0], path.T[1], label="Bezier Path", **Bezier_line_style)
        # ax.plot(cp.T[0], cp.T[1],'--o', label="Control Points", color=colors[j])
        # ax.scatter(np.array(points).T[0], np.array(points).T[1], label="end Points", **Bezier_endcps_style)

    if track_path != None:
        if len(track_path) == 1: 
            ax.plot(track_path[0][0], track_path[0][1], label="Tracking Path", **tracking_line_style)
        else: 
            for i in range(len(track_path)):
                ax.plot(track_path[i][0], track_path[i][1], label="Tracking Path", color=colors[i])

    

    for j in range(len(Bezier)):
        cp = np.array(Bezier[j])
        points = [cp[0], cp[-1]]
        ax.scatter(np.array(points).T[0], np.array(points).T[1], label="end Points", **Bezier_endcps_style)

    
               
    plt.show()


     # Save the chart as PDF and specify the DPI
    if png_path != None: 
        save_png = png_path + ".pdf"
        fig.savefig(save_png, format='pdf', dpi=None)  # 
        save_eps = png_path + ".eps"
        fig.savefig(save_eps, format='eps', dpi=None)  #  




