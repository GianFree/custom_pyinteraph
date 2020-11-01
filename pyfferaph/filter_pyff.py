import networkx as nx
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.special import expit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda



# Parametric sigmoid function for fitting
def sigmoid(x, x0, k, m, n): 
    y = m / (1 + np.exp(k*(x-x0))) + n
    return y

# Parametric analytic second derivative of sigmoid 
def seconddevsigmoid(x, x0, k, l, m): 
    y = ( k**2 * l * np.exp(k*(x+x0)) * ( np.exp(k*x)-np.exp(k*x0) )  )   /   ( np.exp(k*x0) + np.exp(k*x) )**3    
    return y    



def cluster_plotter(network, fig_name, stride=0.1):
    
    ''' plots the size of the biggest cluster
    in network as a function of increasing
    persistence values, from 0 to 100 default
    stride is 0.1 '''
        
    vals = np.arange(0, 100, stride)
    maxclustsize = []

    for val in vals:       
        boolmats = np.array([i>val for i in network])
        G=nx.Graph(boolmats)
#     print(len(max(nx.connected_components(G), key=len)))
        maxclustsize.append(len(max(nx.connected_components(G), key=len)))
#     print(maxclustsize)

    x = vals
    y = maxclustsize
    plt.plot(x,y, '.')
    #plt.title(fig_name)
    plt.xlabel('$p_{min}$')
    plt.ylabel('size of the biggest cluster')
    plt.savefig(fig_name)


def macroINN_generator(matrices_list, p_crit):
    
    ''' returns a macro interaction network containing
    all the instances from each network in
    matrices list with a value greater than p_crit '''

    boolmats = [i > p_crit for i in matrices_list]
    macroINN = boolmats[0]

    for i in range (1,len(boolmats)):
        macroINN = np.logical_or(macroINN, boolmats[i])
    
    return macroINN


def network_generator(network_matrix, topology_file):
    
    '''returns a graph G based on the adjacency 
    network_matrix. Each node in the graph is 
    named after the corrisponding residue in 
    the topology_file '''
    
    u = mda.Universe(topology_file)
    G = nx.Graph(network_matrix)
    identifiers = ["%s%d" % (r.resname,r.resnum) for r in u.residues]
    #print(identifiers)
    node_names = dict(zip(range(0,network_matrix.shape[0]),identifiers))
    #print(node_names)
    nx.relabel_nodes(G, node_names, copy=False)
    
    return G

def shortest_paths(G, source, target):

    '''Returns all shorteset paths found 
    in network G connecting source and target'''

    try:
        nx.shortest_path(G,source,target)
    except:
        print('no path between {} and {}'.format(source, target))
        print('\n')
        return None
    
    paths = [k for k in nx.all_shortest_paths(G,source,target)]

    return paths


def path_scores(paths):

    '''Returns a dictionary with the following structure {'path':score}.
    A score is assigned to each path in paths; the score
    is computed summing the frequencies, calculated over all the 
    paths connecting source and target, of each residue in the path;'''

    frequency = dict()
    for path in paths:
        for position in range(len(path)):
            if path[position] not in frequency:
                frequency[path[position]] = 1
            else:
                frequency[path[position]] += 1
    #print(frequency)
           
    scores = dict()
    for path in paths:
        path_score = sum(frequency[i] for i in path)    
        scores[str(path)] = path_score

    return scores

def communication_robustness(paths, threshold):

    '''returns the communication robustness (cr) index 
    for the pathway (list of paths) source-target (s-t) as:
    cr(s-t) = (paths s-t) * threshold / lenght_shortest_path '''

    n = len(paths)
    l = len(paths[0])
    cr = round(n*threshold/(l*100), 2)

    return cr

def selective_betweenness(G, source, target, sb_residue):
    
    ''' returns the selective betweenness value for
    sb_residue computed over all the shortest path 
    connecting source and target in G '''
    
    paths = [k for k in nx.all_shortest_paths(G,source,target)]

    sb = 0
    for path in paths:
        #print(path)
        if sb_residue in path:
            sb += 1
    sb = round(sb / len(paths), 2)
    print(f'source: {source} target: {target}\nselective betweenness for {sb_residue}: {sb}')
    return sb


def all_shortest_paths(G, source_res, target_res, threshold):
    
    '''prints to file all shortest paths in netework G connecting each
    source-target residues pair in source_res and target_res.
    A score is assigned to each path; the corresponding cr 
    index is assigned to each pathway'''
        
    for res in source_res:
        file_name = res + '.dat'
        lines = 0
        with open(file_name, 'w') as paths_file:
            for int_res in target_res:
                paths = shortest_paths(G, res, int_res)
                #print(paths)
                cr_index = communication_robustness(paths, threshold)
                #print(cr_index)
                scores = path_scores(paths)
                max_score = max(scores.values())
                best_paths = []
                #print(scores)
                path_info = f'source node:\t{res}\ntarget node:\t{int_res}\n'
                paths_file.write(path_info)
                paths_file.write(f'cr value:\t{str(cr_index)}\n')
                paths_file.write(f'Number of shortest paths:\t{len(paths)}\n')
                if paths != None:
                    paths_file.write(f'Length of the shortest paths:\t{len(paths[0])}\n\n')
                    path_i = 0
                    for path in paths:
                        path_i += 1
                        lines += 1
                        paths_file.write(f'Path\t{path_i}:\n{str(path)}\n')
                        score = scores[str(path)]
                        #print(score)
                        if score == max_score:
                            best_paths.append(path)
                        paths_file.write(f'score\t{str(score)}\n\n')
                    paths_file.write(f'Best shortest paths (score {max_score}):\n')
                    for k in best_paths:
                        paths_file.write(f'{k}\n')
                    paths_file.write('\n\n')
        print(f'{lines} paths written to {file_name}')
