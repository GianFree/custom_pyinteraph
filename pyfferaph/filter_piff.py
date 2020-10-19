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
    plt.xlabel('p_{min}')
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


def all_shortest_path(G, source, target, threshold, terminal=False):
    
    ''' prints to file all shortest paths in netework G connecting
    source and target. A score is assigned to each path; the score
    is computed summing the frequencies, calculated over all the 
    paths connecting source and target, of each residue in the path;
    returns the communication robustness (cr) index for the pathway 
    source-target (s-t) as:
    cr(s-t) = (paths s-t) * treshold / lenght_shortest_path '''
        
    try:
        nx.shortest_path(G,source,target)
    except:
        print('no path between {} and {}'.format(source, target))
        print('\n')
        return None
    
    paths = [k for k in nx.all_shortest_paths(G,source,target)]
    #for j in paths:
    #    print(j)
    #    print(len(j))
    
    frequency = dict()
    for path in paths:
            for position in range(len(path)):
                    if path[position] not in frequency:
                            frequency[path[position]] = 1
                    else:
                            frequency[path[position]] += 1
    #print(frequency)
    file_name = 'shortest_path_' + source + '_' + target + '.dat'
           
    scores = dict()
    with open(file_name, 'w') as out_paths:
        out_paths.write('Paths:\n')
        for path in paths:
            path_score = sum(frequency[i] for i in path)
            if path_score not in scores:
                scores[path_score] = [path]
            else:
                scores[path_score].append(path)
            if terminal:
                print(path,'\tscore=', path_score)
            out_paths.write(f'{path}\tscore={path_score}\n')

            
    max_score = max(scores.keys())
    
    n = len(paths)
    l = len(path)
    cr = round(n*threshold/(l*100), 2)
    
    
    with open(file_name, 'a') as out_paths:
        out_paths.write(f'Number of shortest paths: {n}\n')
        out_paths.write(f'Length of the shortest path: {l}\n')
        out_paths.write(f'CR index = {cr:.2f}\n')
        out_paths.write(f'Best shortest path:\n')
        
        for el in scores[max_score]:
            string2file = str(el)+'\n'
            out_paths.write(string2file)
            
        out_paths.write(f'source: {source}\ntarget: {target}\n')
        #out_paths.write('\n')

    if terminal:
            print('Number of shortest paths:',n)
            print('Length of the shortest path:',l)
            print('CR index = %.2f' % cr)
            print('Best shortest path:')
            for el in scores[max_score]:
                    print(el)
            print(source, target)
            print('\n')

    #return paths, cr


def paths_to_interface(G, interface_residues, source_residues, threshold):
    
    ''' prints to file all the shortest path for each inteface-source
    residues pair found in network G. Calculates the cr index for each
    inteface-source pathway. '''
        
    interactions = {k:[] for k in interface_residues}
    #print(interactions)
    for res in interface_residues:
        	for int_res in source_residues:
                    temp = all_shortest_path(G, res, int_res, threshold, terminal=False)
                    if temp != None:
                            #print(temp)
                            interactions[res].append(temp)


    #print(len(interactions['GLU85']))
    for key in interactions.keys():
            #ref_structure_suffix = os.path.splitext(args.top)[1]
            file_name = key + '.dat' #args.top.rstrip(ref_structure_suffix) + '_' + key + '.dat'
            #path_to_file = os.getcwd() + '/' + file_name
            #print(path_to_file)
            lines = 0
            with open(file_name, 'w') as paths_file:
                    for tupla in interactions[key]:
                            #print(tupla)
                            path_info = f'source node\t{key}\ntarget node\t{str(tupla[0][0][-1])}\n'
                            cr_val = 'cr_value\t' + str(tupla[-1])+'\n'
                            paths_file.write('\n')
                            paths_file.write(path_info)
                            paths_file.write(cr_val)
                            for path in tupla[0]:
                                    paths_file.write(str(path)+'\n')
                                    lines += 1
                    print(f'{lines} paths written to {file_name}')


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




