import itertools
import numpy as np
import gzip, sys
import numba as nb
from numba import jit, njit, types
import random
from graph_tool.all import *
from line_profiler import LineProfiler, profile
import copy
import json
from pysam import VariantFile
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import argparse
from hap_utils import open_bcf_to_read


def worker(queue, result_queue, free_fixed_found, core_components, draw, cores_to_process, basis_genotype):
    while queue:
        core_to_process = queue.get()
        if core_to_process is None:  # Check for sentinel value
            return result_queue

        start_time = time.time()
        new_cores = process_core(core_to_process, free_fixed_found, core_components, draw, cores_to_process, basis_genotype)
        end_time = time.time()
        for new_core in new_cores:
            print("finished core with size ",len(new_core[0].get_vertices())," and ",len(genotypes)," genotypes"," in ",end_time-start_time," seconds", file=sys.stderr)
            result_queue.put(new_core)


@njit(types.uint8(types.uint8,types.uint8))
def pconsistent_val(g1,g2):
    if g1==2 and g2==2:
        return 2
    elif g1==2:
        return g2
    elif g2==2:
        return g1
    elif g1==g2:
        return g1
    else: 
        return 255


def complement_genotype(genotype1,genotype2):
    complement = np.zeros((M),dtype=np.uint8)
    for i in range(len(genotype1)):
        comp = complement_allele(genotype1[i],genotype2[i])
        if comp == -1:
            return None
        else:
            complement[i]=comp
    return complement

@njit(types.uint8(types.uint8,types.uint8))
def complement_allele(genotype_allele1,genotype_allele2):
    if genotype_allele1==2 and genotype_allele2==2:
        return 2
    elif genotype_allele1==2:
        return np.mod(genotype_allele2+1,2)
    elif genotype_allele2==2:
        return np.mod(genotype_allele1+1,2)
    elif genotype_allele1==genotype_allele2:
        return genotype_allele1
    else:
        return -1    

@njit(types.uint8[:](types.Array(types.uint8, 1, 'C', readonly=False),types.Array(types.uint8, 1, 'C', readonly=True),types.int64))
def fast_complement_genotype(genotype1,genotype2,M):
    complement = np.zeros((M),dtype=np.uint8)
    for i in range(len(genotype1)):
        complement[i] = complement_allele(genotype1[i],genotype2[i])
    return complement


def check_intersection(template1,template2):
    for i in range(len(template1)):
        inter_val = pconsistent_val(template1[i],template2[i])
        if inter_val == 255:
            return False
    return True

@njit(types.boolean(types.Array(types.uint8, 1, 'C', readonly=False),types.Array(types.uint8, 1, 'C', readonly=True)))
def fast_check_intersection(template1,template2):
    for i in range(len(template1)):
        inter_val = pconsistent_val(template1[i],template2[i])
        if inter_val == 255:
            return False
    return True

@jit(types.uint8[:](types.Array(types.uint8, 1, 'C'),types.Array(types.uint8, 1, 'C', readonly=True)))
def fast_intersect(template1,template2):
    # assumes that the two templates are compatible
    intersection = np.zeros((len(template1)),dtype=np.uint8)
    for i in range(len(template1)):
        intersection[i]=pconsistent_val(template1[i],template2[i])
    return intersection

def intersect(template1,template2):
    intersection = np.zeros((len(template1)),dtype=np.uint8)
    for i in range(len(template1)):
        inter_val = pconsistent_val(template1[i],template2[i])
        if inter_val == -1:
            return None
        else:
            intersection[i]=inter_val
    return intersection


def consistent(template1,template2):
    for i in range(len(template1)):
        if not pconsistent(template1[i],template2[i]):
            return False
    return True


def gconsistent(h,h2,g):
    if g==0 and h==0 and h2==0:
        return 0
    elif g==1 and h==1 and h2==1: 
        return 1
    elif g==2:
        if (h==1 and h2==1) or (h==0 and h2==0):
            return None
        else:
            return min(h,h2)
    else: 
        return None


def pconsistent(g1,g2):
    if g1==2 or g2==2 or g1==g2:
        return True
    else: 
        return False




def get_template(clique,P):
    itemplate = np.zeros((M),dtype=np.uint8)
    for i in range(M):
        consensus = 2
        for vertex in clique:
            if P[vertex][i]!=2:
                consensus = P[vertex][i]
        itemplate[i]=consensus
    return itemplate


def generate_haps_from_genotype(original_genotype, node_genotype, haplotype_to_freq_map, gen_to_haplotypes):
    if 2 not in node_genotype:
        haplotype_to_freq_map[node_genotype]=0
        gen_to_haplotypes[original_genotype].append(node_genotype)
    else:
        unique, counts = np.unique(np.frombuffer(node_genotype, dtype=np.uint8), return_counts=True)
        num_twos = dict(zip(unique, counts))[2]
        if num_twos<30:
            haplotypes = np.zeros((np.power(2,num_twos), M), dtype=np.uint8)
            truth_table = np.array(list(itertools.product([0, 1], repeat=num_twos)), dtype=np.uint8)
            idx = 0
            for i in range(len(node_genotype)):
                if node_genotype[i]==1:
                    haplotypes[:,i]=1
                elif node_genotype[i]==2:
                    haplotypes[:,i]=truth_table[:,idx]
                    idx+=1
            for haplotype in haplotypes:
                haplotype_to_freq_map[haplotype.data.tobytes()]=0
                gen_to_haplotypes[original_genotype].append(haplotype.data.tobytes())
        else:
            print("skipping genotype with too many twos:",''.join(map(str, np.frombuffer(node_genotype, dtype=np.uint8))), file=sys.stderr)

# head is the template for the starting set of the path explored so far
# tail is the parity of the number of genotype twos traversed at each position

def find_paths_r(G, head, tail, used, last):
    call_stack = []
    call_stack.append((G, head, tail, used, last, 0,True, None))
    it = 0
    while len(call_stack)>0:   
        it+=1
        if it%10000==0: print(it,len(call_stack),start_idx, file=sys.stderr)
        (G, head, tail, used, last, start_idx,self_loop,lastNTail)=call_stack.pop()
        yield head

        if not self_loop:
            used.remove(lastNTail.data.tobytes())

        for idx in range(start_idx,len(G)):
            g = G[idx]
            if arrays_equal(g,last): # avoid backtracking
                continue
            self_loop = True
            nhead = head.copy()
            ntail = tail.copy()
            self_loop,nhead,ntail = getnHeadnTail(g,head,tail,nhead,ntail)

            cyclic = (ntail is not None) and (ntail.data.tobytes() in used) and not self_loop # if this is a self-loop then tail=ntail
            if ntail is not None and not cyclic:
                used.add(ntail.data.tobytes())
                call_stack.append((G, head, tail, used, last, idx+1,self_loop,ntail))
                call_stack.append((G, nhead, ntail, used, g, 0, True,None))
                break

@nb.jit(nopython=True)
def getnHeadnTail(g,head,tail,nhead,ntail):
    for i in range(M):
        if g[i] == 2: # g has a two, so flip the tail value
            self_loop = False # this isn't a self-loop if g has any twos
            ntail[i] = 1 - ntail[i]
        elif head[i] == 2: # g doesn't have a two, but the template does
            if tail[i] == 0: # head and tail should be the same
                nhead[i] = g[i]
            else: # head and tail should be different
                nhead[i] = 1 - g[i]
        elif (tail[i] == g[i]) != (head[i] == 0): # Mismatch!
            ntail = None
            break
    return self_loop,nhead,ntail

@nb.jit(nopython=True)
def arrays_equal(a, b):
    if a is None or b is None:
        return False
    for ai, bi in zip(a.flat, b.flat):
        if ai != bi:
            return False
    return True


def compatible(h,h2,g):
    comp_gen = np.zeros((M),dtype=np.uint8)
    for i in range(len(h)):
        con = gconsistent(h[i],h2[i],g[i])
        if con is None:
            return None
        else:
            comp_gen[i]=con
    return comp_gen


def get_basis_allele(base,index):
    return base[index]


def get_non_basis_allele(base,index):
    return np.mod(base[index]+1,2)


def annotate_template(core_component,genotypes,free_fixed_found, basis_genotype, root_vertex=0): 
    # theorem 4.1 with a template graph that only has edge labels. Fill in the free-fixed positions and templates
    # does not extend any graph, simply annotates the vertices

    # get an edge adjacent to the root vertex
    random_genotype_index = core_component.ep["genotype"][core_component.edge(root_vertex,core_component.get_all_neighbors(root_vertex)[0])]
    random_genotype = genotypes[random_genotype_index]

    fingerprint = core_component.new_vp("vector<int8_t>")     # creates a VertexPropertyMap of type string
    root_distance = core_component.new_vp("int8_t")     # creates a VertexPropertyMap of type string
    node_updated = core_component.new_vp("bool")     # creates a VertexPropertyMap of type string
    # contains a vector of integers that is twice the length of a genotype
    # the first half of the vector is the free and fixed positions, the second is the fingerprint
    core_component.vp["fingerprint"] = fingerprint
    core_component.vp["root_distance"] = root_distance
    core_component.vp["updated"] = node_updated


    # add fingerprint for the root
    core_component.vp["fingerprint"][core_component.vertex(root_vertex)]=np.zeros((M),dtype=np.uint8)
    core_component.vp["root_distance"][core_component.vertex(root_vertex)]=0

    seen_edge = set()
    bfs_queue = []
    bfs_queue.append(core_component.vertex(root_vertex))

    finger_to_vertex = {}
    finger_to_vertex[core_component.vp["fingerprint"][root_vertex].get_array().tobytes()]=root_vertex

    while len(bfs_queue)>0:
        vertex = bfs_queue.pop(0)
        # for each vertex adjacent to vertex
        verts = np.unique(core_component.get_all_neighbours(vertex))
        skip_edge_removed=set()
        for v in verts:
            if v in skip_edge_removed or (v,vertex) in seen_edge:
                continue
            genotype_idx = core_component.ep["genotype"][core_component.edge(vertex,v)]
            genotype = genotypes[genotype_idx]

            # get free_fixed positions from the vertex we are coming from
            free_fixed = np.zeros((M),dtype=np.uint8)
            free_fixed = fast_compute_genotype(core_component.vp["fingerprint"][vertex].get_array(), basis_genotype, free_fixed) # gets free fixed for target node

            # check if the free_fixed positions of the vertex we are coming from are compatible with the newly added genotype
            is_compat = fast_check_intersection(free_fixed,genotype)

            seen_edge.add((v,vertex))     

            if not is_compat:
                # kill this branch
                core_component.remove_edge(core_component.edge(vertex,v))
                continue

            # if we do not have a self loop, then we must do additional processing
            if not (v == vertex and vertex in core_component.get_all_neighbours(v)):
                # not a self-loop
                if core_component.edge(vertex,v) is None:
                    # don't go backwards and do not process an edge we've removed
                    continue
                # there already are edge labels, add the appropriate vertex labels
                seen_edge.add((vertex,v))

                compat_g = fast_intersect(free_fixed,genotype) # this gets the free-fixed haplotype for the node we are coming from

                
                basis_genotype = update_basis(basis_genotype, core_component.vp["fingerprint"][core_component.vertex(vertex)].get_array(), compat_g)

                # now compute fingerprint
                distance_from_root = core_component.vp["root_distance"][vertex]+1
                free_fixed = np.zeros((M),dtype=np.uint8)
                free_fixed = fast_compute_genotype(core_component.vp["fingerprint"][core_component.vertex(root_vertex)].get_array(), basis_genotype, free_fixed) # gets free fixed for target node
                comp_gen = fast_complement_genotype(compat_g,genotype,M) # this gets the complement free-fixed haplotype given the shared genotype
                if free_fixed.tobytes() in free_fixed_found or comp_gen.tobytes() in free_fixed_found:
                    return True
                
                fingerprint = fast_get_fingerprint(comp_gen, free_fixed,distance_from_root, M)

                if fingerprint.tobytes() in finger_to_vertex:
                    # if we are pointing to the vertex that already exists, then this edge just reinforces what's already there, keep it
                    # otherwise, clear the edge
                    if finger_to_vertex[fingerprint.tobytes()]!=v:
                        core_component.remove_edge(core_component.edge(vertex,v))  #vertex->v
                        # erase v in case it is connected through another path
                        core_component.vp["fingerprint"][v].clear()
                        core_component.vp["root_distance"][v] = 255

                        finger_vertex = finger_to_vertex[fingerprint.tobytes()]
                        if core_component.edge(vertex,finger_vertex) is None:
                            core_component.add_edge(core_component.vertex(vertex), finger_vertex)
                            seen_edge.add((finger_vertex,vertex))
                            seen_edge.add((vertex,finger_vertex))
                            label_edge(core_component, vertex, finger_vertex, genotype_idx)        
                else:
                    finger_to_vertex[fingerprint.tobytes()]=v
                    label_vertex_full(core_component, v, fingerprint,distance_from_root)
                    basis_genotype = update_basis(basis_genotype, fingerprint, comp_gen)
                    bfs_queue.append(v)

    return False


def find_initial_connected_component_core(genotypes, seed_node_ff, genotype_index): # theorem 4.1
    # select basis genotype

    core_component = Graph(directed=False) 
    core_component.set_fast_edge_removal(fast=True)

    edge_labels = core_component.new_ep("int32_t")    # creates an EdgePropertyMap of type double
    # stores the genotype index, the genotype can be retrieves from "genotypes" object
    fingerprint = core_component.new_vp("vector<int8_t>")     # creates a VertexPropertyMap of type string
    root_distance = core_component.new_vp("int8_t")     # creates a VertexPropertyMap of type string
    # contains a vector of integers that is twice the length of a genotype
    # the first half of the vector is the free and fixed positions, the second is the fingerprint
    core_component.vp["fingerprint"] = fingerprint
    core_component.vp["root_distance"] = root_distance


    # contains a vector of integers that is twice the length of a genotype
    # the first half of the vector is the free and fixed positions, the second is the fingerprint

    node_updated = core_component.new_vp("bool") 
    core_component.vp["updated"] = node_updated

    vprop = core_component.new_vertex_property("int")
    core_component.vp["root"] = vprop

    basis_genotype = np.zeros((M),dtype=np.uint8)
    np.copyto(basis_genotype,seed_node_ff) # 2 if unset, 0 if basis is 0, 1 if basis is 1


    core_component.ep["genotype"] = edge_labels


    # select root as one of its adjacent vertices
    root_vertex = 0
    if 2 in seed_node_ff:
        core_component.add_vertex(2) # root is vertex at index 0
        other_vertex = 1
    else:
        # self loop
        core_component.add_vertex(1) # root is vertex at index 0
        other_vertex = 0

    core_component.vp["root"][other_vertex]=0
    core_component.vp["root"][root_vertex]=1

    core_component.add_edge(core_component.vertex(root_vertex), core_component.vertex(other_vertex))
    core_component.ep["genotype"][core_component.edge(root_vertex,other_vertex)]=genotype_index

    # this is the basis genotype
    core_component.vp["fingerprint"][core_component.vertex(root_vertex)]=np.zeros((M),dtype=np.uint8)
    core_component.vp["root_distance"][core_component.vertex(root_vertex)]=0
 
    core_component.vp["fingerprint"][core_component.vertex(other_vertex)]=np.zeros((M),dtype=np.uint8)
    core_component.vp["root_distance"][core_component.vertex(other_vertex)]=1
    
    free_fixed = compute_genotype(core_component.vp["fingerprint"][core_component.vertex(other_vertex)].get_array(), basis_genotype)
    free_fixed_root = compute_genotype(core_component.vp["fingerprint"][core_component.vertex(root_vertex)].get_array(), basis_genotype)
    
    fingerprint = fast_get_fingerprint(free_fixed,free_fixed_root,1,M)
    core_component.vp["fingerprint"][core_component.vertex(other_vertex)]=fingerprint
    core_component.vp["root_distance"][core_component.vertex(other_vertex)]=1

    
    (root_vertex, core_component), basis_genotype = find_connected_component_core(genotypes, core_component, 0, set(), basis_genotype)
    return core_component, basis_genotype


def find_connected_component_core(genotypes, core_component, root_vertex, free_fixed_found, basis_genotype): 
    # theorem 4.1 which maximizes a partially resolved core

    genotype_idx_to_adj_vertex = {}
    for gidx in range(len(genotypes)):
        genotype_idx_to_adj_vertex[gidx]=set()

    added_vertex = True
    vertices_to_process = set()
    vertices_to_process.update(core_component.get_vertices())

    finger_to_vertex = {}

    # map fingerprints to vertices
    for v in core_component.get_vertices():
        for ajv in core_component.get_all_neighbours(v):
            genotype_idx_to_adj_vertex[core_component.ep["genotype"][core_component.edge(v,ajv)]].add(v)
            genotype_idx_to_adj_vertex[core_component.ep["genotype"][core_component.edge(v,ajv)]].add(ajv)
        if len(core_component.vp["fingerprint"][v])>0:
            finger_to_vertex[core_component.vp["fingerprint"][v].get_array().tobytes()]=v
            

    while vertices_to_process:
        vertices_to_add_next_round = set()
        v = vertices_to_process.pop()

        for idx,genotype in enumerate(genotypes):
                if v not in genotype_idx_to_adj_vertex[idx]:
                    free_fixed = np.zeros((M),dtype=np.uint8)
                    free_fixed = fast_compute_genotype(core_component.vp["fingerprint"][v].get_array(), basis_genotype, free_fixed) # gets free fixed for target node

                    is_compat = fast_check_intersection(free_fixed,genotype) # check to see if we are able to add this genotype to the node
                    if is_compat:
                        compat_g = fast_intersect(free_fixed,genotype)   # this gets the free-fixed haplotype for the node we are coming from
                        comp_gen = fast_complement_genotype(compat_g,genotype,M)    # this gets the complement free fixed positions

                        # check if compat=comp-gen, then self loop!
                        if np.array_equal(compat_g, comp_gen):
                            core_component.add_edge(core_component.vertex(v), core_component.vertex(v))
                            # core_component.vp["adj_genos"][v].append(idx)
                            label_edge(core_component, v,v, idx)
                            added_vertex_idx = v
                            genotype_idx_to_adj_vertex[idx].add(v)
                        else:
                            # otherwise, we need to add a new vertex
                            # update graph first, then fingerprint!
                            # update vertex we're coming from
                            added_vertex_idx = add_vertex_and_edge_to_component(core_component,v,idx)

                            genotype_idx_to_adj_vertex[idx].add(v)
                            genotype_idx_to_adj_vertex[idx].add(added_vertex_idx)

                            # and we're we are going to
                            label_edge(core_component, v,added_vertex_idx, idx)

                            # now compute fingerprint
                            distance_from_root = core_component.vp["root_distance"][v]+1

                            # must update basis genotype first
                            basis_genotype = update_basis_root(basis_genotype, comp_gen, distance_from_root)

                            # root free_fixed must be updated!
                            free_fixed_root = compute_genotype(core_component.vp["fingerprint"][core_component.vertex(root_vertex)].get_array(), basis_genotype)
                            fingerprint = fast_get_fingerprint(comp_gen,free_fixed_root,distance_from_root,M)

                            if fingerprint.tobytes() in finger_to_vertex:
                                genotype_idx_to_adj_vertex[core_component.ep["genotype"][core_component.edge(v,added_vertex_idx)]].remove(v)
                                genotype_idx_to_adj_vertex[core_component.ep["genotype"][core_component.edge(v,added_vertex_idx)]].remove(added_vertex_idx)
                                core_component.remove_edge(core_component.edge(v,added_vertex_idx))
                                if core_component.edge(v,finger_to_vertex[fingerprint.tobytes()]) is None:
                                    # if the edge is already in the graph, don't add a duplicate edge
                                    core_component.add_edge(core_component.vertex(v), finger_to_vertex[fingerprint.tobytes()])
                                    genotype_idx_to_adj_vertex[idx].add(v)
                                    genotype_idx_to_adj_vertex[idx].add(finger_to_vertex[fingerprint.tobytes()])
                                    label_edge(core_component, v,finger_to_vertex[fingerprint.tobytes()], idx)
                            else: 
                                label_vertex_full(core_component, added_vertex_idx, fingerprint,distance_from_root)
                                vertices_to_add_next_round.add(added_vertex_idx)
                                finger_to_vertex[fingerprint.tobytes()]=added_vertex_idx

                                # a new vertex added or changed and fingerprint added, so we might have to update the basis
                                basis_genotype = update_basis(basis_genotype, fingerprint, comp_gen)

 
        vertices_to_process.update(vertices_to_add_next_round)    
    
    for v in core_component.get_vertices():
        free_fixed = compute_genotype(core_component.vp["fingerprint"][v].get_array(), basis_genotype)
        if free_fixed.tobytes() in free_fixed_found:
            return (None, None), basis_genotype

    return clean_component(core_component, root_vertex), basis_genotype

def update_basis(basis_genotype, fingerprint, comp_gen):
    # fingerprint: 0 when node matches root. 1 when opposite. If free, it's the length of walk mod 2
    for i in range(len(basis_genotype)):
        if basis_genotype[i] == 2: # if we still need to update the basis
            if comp_gen[i] != 2: # if the genotype is not a two, then it fixes a position
                if fingerprint[i] == 0:
                    basis_genotype[i] = comp_gen[i]
                elif fingerprint[i] == 1:
                    basis_genotype[i] = np.mod(comp_gen[i]+1,2)
    return basis_genotype

def update_basis_root(basis_genotype, comp_gen, distance):
    # fingerprint: 0 when node matches root. 1 when opposite. If free, it's the length of walk mod 2
    for i in range(len(basis_genotype)):
        if basis_genotype[i] == 2: # if we still need to update the basis
            if comp_gen[i] != 2: # if the genotype is not a two, then it fixes a position
                number_of_2s_from_root = distance-1 # because we are just now fixing the root and all nodes between the root and here must have been 2s
                if  np.mod(number_of_2s_from_root,2) == 0: 
                    basis_genotype[i] = comp_gen[i]
                else:
                    basis_genotype[i] = np.mod(comp_gen[i]+1,2)
    return basis_genotype


def update_graph(added_vertex_idx, adj_vertex_idx, graph, genotypes, root_vertex, free_fixed_found,basis_genotype):
    # search through vertices we've updated and update all free/fixed positions for adjacent vertices recursively
    updated = set()
    updated.add(adj_vertex_idx)  
    updated.add(added_vertex_idx)
    to_update_neighbors = set()      
    to_update_neighbors.add(added_vertex_idx)  
    to_update_neighbors.add(adj_vertex_idx)         
    counter=0

    while len(to_update_neighbors)>0:
        edges_to_remove = set() 
        adjv = to_update_neighbors.pop()
        for adjv2 in graph.get_all_neighbours(adjv):
            if len(graph.vp["fingerprint"][graph.vertex(adjv2)].get_array()) == 0:
                continue
            free_fixed_adjv2 = compute_genotype(graph.vp["fingerprint"][graph.vertex(adjv2)].get_array(), basis_genotype)
            if len(free_fixed_adjv2) == 0 or adjv2 in updated:
                    continue
            # adjv was updated but adjv2 has not been updated, so see if we need to make updates to adjv2
            # find complement genotype to adjv and the genotype of the edge between adjv and adjv2
            free_fixed_adjv = compute_genotype(graph.vp["fingerprint"][graph.vertex(adjv)].get_array(), basis_genotype)
            is_compat = fast_check_intersection(free_fixed_adjv,genotypes[graph.ep["genotype"][graph.edge(adjv,adjv2)]])
            if not is_compat:
                # kill this edge
                edges_to_remove.add(graph.edge(adjv,adjv2))
            else:
                comp_gen = fast_complement_genotype(free_fixed_adjv,genotypes[graph.ep["genotype"][graph.edge(adjv,adjv2)]],M)
                if comp_gen.tobytes() in free_fixed_found:
                    return None
                if np.array_equal(comp_gen,free_fixed_adjv2):
                    # no changes in free/fixed positions, so no need to continue down this branch
                    continue
                else:
                    if len(graph.vp["fingerprint"][graph.vertex(adjv2)].get_array()) == 0:
                        # no fingerprint existed before, instantiate
                        distance_from_root = graph.vp["root_distance"][graph.vertex(adjv)]+1
                        free_fixed_root = compute_genotype(graph.vp["fingerprint"][graph.vertex(root_vertex)].get_array(), basis_genotype)
                        fingerprint = fast_get_fingerprint(comp_gen,free_fixed_root,distance_from_root, M)
                        basis_genotype = update_basis(basis_genotype, fingerprint, comp_gen)
                        print("old label",end=" ")
                        print(comp_gen, end=" ")
                        print("new label",end=" ")
                        print(compute_genotype(fingerprint, basis_genotype))
                        label_vertex_full(graph, adjv2, fingerprint,distance_from_root)
                    else:
                        basis_genotype = update_basis(basis_genotype, graph.vp["fingerprint"][graph.vertex(adjv2)].get_array(), comp_gen)
                        # a fingerprint existed, don't touch it
                        print("old label",end=" ")
                        print(comp_gen, end=" ")
                        print("new label",end=" ")
                        print(compute_genotype(graph.vp["fingerprint"][graph.vertex(adjv2)].get_array(), basis_genotype))
                        label_vertex_full(graph, adjv2, graph.vp["fingerprint"][graph.vertex(adjv2)].get_array(),graph.vp["root_distance"][graph.vertex(adjv2)])

                    updated.add(adjv2)
                    to_update_neighbors.add(adjv2)

    return edges_to_remove


def compute_genotype(fingerprint, basis_genotype):
    # compute the current free_fixed (partially resolved) genotype from the fingerprint and basis_genotype
    genotype = np.zeros((M),dtype=np.uint8)
    for i in range(len(fingerprint)):
        if basis_genotype[i] == 2: # the position is still free
            genotype[i] = 2
        elif fingerprint[i] == 0: # the position is fixed to the basis genotype
            genotype[i] = basis_genotype[i]
        elif fingerprint[i] == 1: # the position is fixed to the opposite of the basis genotype
            genotype[i] = np.mod(basis_genotype[i]+1,2)
    return genotype

@njit(types.uint8[:](types.Array(types.uint8, 1, 'C', readonly=False),types.Array(types.uint8, 1, 'C', readonly=True),types.Array(types.uint8, 1, 'C', readonly=False)))
def fast_compute_genotype(fingerprint, basis_genotype, genotype):
    # compute the current free_fixed (partially resolved) genotype from the fingerprint and basis_genotype
    for i in range(len(fingerprint)):
        if basis_genotype[i] == 2: # the position is still free
            genotype[i] = 2
        elif fingerprint[i] == 0: # the position is fixed to the basis genotype
            genotype[i] = basis_genotype[i]
        elif fingerprint[i] == 1: # the position is fixed to the opposite of the basis genotype
            genotype[i] = np.mod(basis_genotype[i]+1,2)
    return genotype



def label_vertex_full(graph,v_idx, fingerprint, root_distance):
    graph.vp["fingerprint"][graph.vertex(v_idx)]=fingerprint
    graph.vp["root_distance"][graph.vertex(v_idx)]=root_distance


def label_edge(graph,v1_idx,v2_idx,label):
    graph.ep["genotype"][graph.edge(v1_idx,v2_idx)]=label


def add_vertex_and_edge_to_component(graph,v_idx,g_idx):
    added_v = graph.add_vertex()
    graph.vp["root"][added_v]=0
    graph.add_edge(graph.vertex(v_idx), 
                   added_v)
    
    return graph.vertex_index[added_v]


def is_g_edge_adjacent(genotype,vertex,graph,genotypes):
    # check if the genotype is adjacent to the vertex
    for v in graph.get_all_neighbours(vertex):
        if np.array_equal(genotype,genotypes[graph.ep["genotype"][graph.edge(vertex,v)]]):
            return True
    return False


def faster_is_g_edge_adjacent(genotype_idx,adjacent_genotypes):
    # check if the genotype is adjacent to the vertex
    for adj_geno in adjacent_genotypes:
        if genotype_idx==adj_geno:
            return True
    return False


def fastest_is_g_edge_adjacent(genotype_idx,vertex,graph):
    # check if the genotype is adjacent to the vertex
    edges = find_edge(graph,graph.ep["genotype"],genotype_idx)
    for edge in edges:
        if edge.source()==vertex or edge.target()==vertex:
            return True
    return False


def fast_is_g_edge_adjacent(genotype_idx,vertex,graph):
    # check if the genotype is adjacent to the vertex
    for v in graph.get_all_neighbours(vertex):
        if genotype_idx==graph.ep["genotype"][graph.edge(vertex,v)]:
            return True
    return False



def get_node_label(free_fixed, fingerprint):
    return ''.join(map(str, free_fixed)) + '\n' + ''.join(map(str, fingerprint))

# free = 2, fixed = 0 or 1

def get_fingerprint(target_vertex_label,root_vertex_label,distance_from_root):
    fingerprint = np.zeros((M),dtype=np.uint8)
    for i in range(len(target_vertex_label)):
        if target_vertex_label[i]==2:
            # vertex is still free
            fingerprint[i]= np.mod(distance_from_root,2)
        elif target_vertex_label[i] in [0,1]:
            # vertex is fixed
            fingerprint[i]=np.mod(root_vertex_label[i]+target_vertex_label[i],2)
        else: 
            # infinite sites not assumed...
            fingerprint[i]=target_vertex_label[i]            

    return fingerprint

@njit(types.uint8[:](types.Array(types.uint8, 1, 'C', readonly=True),types.Array(types.uint8, 1, 'C', readonly=True),types.int64,types.int64))
def fast_get_fingerprint(target_vertex_label,root_vertex_label,distance_from_root,M):
    fingerprint = np.zeros((M),dtype=np.uint8)
    for i in range(len(target_vertex_label)):
        if target_vertex_label[i]==2:
            # vertex is still free
            fingerprint[i]= np.mod(distance_from_root,2)
        elif target_vertex_label[i] in [0,1]:
            # vertex is fixed
            fingerprint[i]=np.mod(root_vertex_label[i]+target_vertex_label[i],2)
        else: 
            # infinite sites not assumed...
            fingerprint[i]=target_vertex_label[i]            

    return fingerprint


def get_free_fixed_vector(genotype):
    free_fixed_vector = np.zeros((M),dtype=np.uint8)
    for i in range(len(genotype)):
        if genotype[i]==2:
            free_fixed_vector[i]=1
        else:
            free_fixed_vector[i]=0
    return free_fixed_vector


def has_gedge_or_conflict(vertex,genotype_idx,graph,genotypes,basis_genotype):
    has_gedge = False
    for v in graph.get_all_neighbours(vertex):
        if graph.ep["genotype"][graph.edge(v,vertex)]==genotype_idx:
            has_gedge = True

    free_fixed = compute_genotype(graph.vp["fingerprint"][vertex].get_array(), basis_genotype)
    is_compat = fast_check_intersection(genotypes[genotype_idx],free_fixed)

    return has_gedge or not is_compat


def homomorphic(existing_core_component,new_core_component,root_vertex,basis_genotype, old_basis_genotype):
    # check if the graph is homomorphic to the template
    found_a_good_vertex = False
    # first, find a vertex that is compatible with the root
    for v in existing_core_component.get_vertices():
        free_fixed_v = compute_genotype(existing_core_component.vp["fingerprint"][v].get_array(), old_basis_genotype)
        free_fixed_root = compute_genotype(new_core_component.vp["fingerprint"][root_vertex].get_array(), basis_genotype)
        if fast_check_intersection(free_fixed_v,free_fixed_root):
            # found a compatible vertex
            # check if the graph is homomorphic
            # if existing_core_component has a g-edge and new_core_component does not, then this is OK
            # if new_core_component has a g-edge and existing_core_component does not, then this we are not homomorphic
            stack = [(root_vertex.__int__(),v.__int__())]
            visited = set()
            vertices_processed = 0

            while stack:
                new_core_vertex,existing_vertex = stack.pop()
                if new_core_vertex in visited:
                    continue
                vertices_processed+=1
                visited.add(new_core_vertex)

                # Check edges between existing_core_component and new_core_component
                for adj_vertex in new_core_component.get_all_neighbours(new_core_vertex):
                    # get g-edge for new core component
                    new_core_genotype_idx = new_core_component.ep["genotype"][new_core_component.edge(new_core_vertex,adj_vertex)]
                    # then, the existing core component should have a g-edge with the same genotype
                    g_edge_adj_vertex = None
                    for v in existing_core_component.get_all_neighbours(existing_vertex):
                        if existing_core_component.ep["genotype"][existing_core_component.edge(v,existing_vertex)]==new_core_genotype_idx:
                            g_edge_adj_vertex = v

                    if g_edge_adj_vertex is None:
                        break
                    
                    if adj_vertex not in visited:
                        stack.append((adj_vertex,g_edge_adj_vertex))
            if vertices_processed == len(new_core_component.get_vertices()):
                found_a_good_vertex = True
    return found_a_good_vertex


def draw_graph(graph, genotypes, basis_genotype, output="graph.svg", node_labels=True, edges_labels=True, shorten=False):

    if graph is None:
        return
    edge_labels_print = graph.new_ep("string") 
    node_labels_print = graph.new_vp("string") 
    if node_labels:
        # for each vertex in graph
        for v in graph.get_vertices():    
            free_fixed_v = compute_genotype(graph.vp["fingerprint"][v].get_array(), basis_genotype)
            if len(graph.vp["fingerprint"][v].get_array())==0:
                if shorten:
                    node_labels_print[v] = (''.join(map(str, free_fixed_v)))[0:5]+"..."
                else:
                    node_labels_print[v] = ''.join(map(str, free_fixed_v))
            else:
                if shorten:
                    node_labels_print[v] = get_node_label(free_fixed_v[0:5],graph.vp["fingerprint"][v].get_array()[0:5])
                else:
                    node_labels_print[v] = get_node_label(free_fixed_v,graph.vp["fingerprint"][v].get_array())
    if graph.ep["genotype"] is not None and edges_labels:
        # for each edge in the graph
        for e in graph.edges():
            if shorten:
                edge_labels_print[e] = (''.join(map(str, genotypes[graph.ep["genotype"][e]])))[0:5]+"..."
            else:
                edge_labels_print[e] = ''.join(map(str, genotypes[graph.ep["genotype"][e]]))
    graphviz_draw(graph,output=output,vprops={"label":node_labels_print},eprops={"label":edge_labels_print},size=(30,30),ratio = "expand", sep=10, overlap=False)


def clean_component(new_core_component, v):
    # only retain the connected component
    comp, hist = label_components(new_core_component)
    component_id = comp[v]


    # NEW TRYING THIS OUT
    # filter the graph
    new_core_component = GraphView(new_core_component, vfilt = comp.a == component_id)

    root_vertex = np.where(new_core_component.vp["root"].get_array()==1)[0][0]
        
    return root_vertex, new_core_component



def process_core(core_to_process, free_fixed_found, core_components, draw, cores_to_process, basis_genotype):
    
    # make a new array with the same size as basis_genotype
    old_basis = np.zeros_like(basis_genotype)
    np.copyto(old_basis,core_to_process[1])
    core_to_process = core_to_process[0]
    new_cores = []

    templates = []
    saved_basis_genotype = np.zeros_like(basis_genotype)
    for v in core_to_process.get_vertices():

        # try adding every edge to v such that v does not already have an edge with that genotype
        # like in Figure 5: try adding genotype C to root -> it breaks the B edge
        for genotype_idx in range(len(genotypes)):
            if not fast_is_g_edge_adjacent(genotype_idx,v,core_to_process):
                np.copyto(saved_basis_genotype,basis_genotype)
                new_core_component = Graph(core_to_process, directed=False)

                vprop = new_core_component.new_vertex_property("int")
                new_core_component.vp["root"] = vprop

                # update the basis genotype for the new component
                np.copyto(saved_basis_genotype,genotypes[genotype_idx])

                # add genotype_idx edge to v
                genotype = genotypes[genotype_idx]
                added_vertex_idx = add_vertex_and_edge_to_component(new_core_component,v,genotype_idx)
                label_edge(new_core_component, v,added_vertex_idx, genotype_idx)

                new_core_component.vp["root"][added_vertex_idx]=1
                root_vertex = added_vertex_idx

                # remove all edges from v that now conflict based on this new edge
                edges_to_remove = set()
                for adjv in new_core_component.get_all_neighbours(v):
                    if not consistent(genotypes[genotype_idx],genotypes[new_core_component.ep["genotype"][new_core_component.edge(v,adjv)]]):
                        edges_to_remove.add(new_core_component.edge(v,adjv))
                
                if len(edges_to_remove)>0:
                    for redge in edges_to_remove:
                        new_core_component.remove_edge(redge) 

                
                homo = annotate_template(new_core_component,genotypes,free_fixed_found,saved_basis_genotype,root_vertex)  # This should correct the template based on the new edge. 
                if homo:
                    continue

                # now, maximize it!
                root_vertex, new_core_component = clean_component(new_core_component, root_vertex)
                
                (root_vertex, new_core_component), saved_basis_genotype = find_connected_component_core(genotypes, new_core_component, root_vertex, free_fixed_found, saved_basis_genotype)
                
                if root_vertex is None and new_core_component is None:
                    homomorphic_template = True
                    continue

                # use Lemma 4.3 to make sure we are not homomorphic to something in the core
                # template_num+=1
                homomorphic_template = False
                for idx2,existing_core_component in enumerate(core_components):
                    old_basis = existing_core_component[1]
                    existing_core_component = existing_core_component[0]
                    if homomorphic(existing_core_component,new_core_component, root_vertex,basis_genotype,old_basis):
                        homomorphic_template = True
                if not homomorphic_template:
                    
                    for v_iterator in new_core_component.get_vertices():
                        free_fixed_v = compute_genotype(new_core_component.vp["fingerprint"][v_iterator].get_array(), saved_basis_genotype)
                        free_fixed_found.add(free_fixed_v.tobytes())
                    if draw:
                        draw_graph(new_core_component, genotypes,saved_basis_genotype, "core"+str(random.randint(0, 10000))+".svg") # CORE PRINTING
                    new_cores.append((new_core_component, saved_basis_genotype))

    return new_cores
            

def core_phase(genotypes, output_dir, genotype_multiplicities=None, threads=1, draw=False, skip_prop=100, em_init="proportional", num_it=1, greedy_min_expl=None, greedy_genmult_min_expl=None):

    # contains all found free-fixed positions
    free_fixed_found = set()
    genotypes_found = set()

    # core_count = 0
    core_components = []
    cores_to_process = []

    # find additional core components such that each genotype is in at least 1 core component
    # take a core piece, take all genotypes not in core, expand out a core from that piece, iterate until all genotypes are part of some core
    while len(genotypes_found) < len(genotypes):
        # find a genotype not in the core
        g_not_in_core = None
        g_idx_not_in_core = None
        for idx,g in enumerate(genotypes):
            if g.data.tobytes() not in genotypes_found:
                g_idx_not_in_core = idx
                g_not_in_core = g
                break
        if g_not_in_core is not None:
            # find a connected component with this genotype
            core_component, basis_genotype = find_initial_connected_component_core(genotypes,  g_not_in_core, g_idx_not_in_core)
        # add the genotype to the core
        for v in core_component.get_vertices():
            free_fixed_v = compute_genotype(core_component.vp["fingerprint"][v].get_array(), basis_genotype)
            free_fixed_found.add(free_fixed_v.tobytes())

            for edge in core_component.get_all_edges(v):
                edge_genotype = genotypes[core_component.ep["genotype"][edge]]
                genotypes_found.add(edge_genotype.data.tobytes())

        if draw:
            draw_graph(core_component, genotypes,basis_genotype, "core"+str(random.randint(0, 10000))+".svg") # CORE PRINTING
        core_components.append((core_component,basis_genotype))


    processed_core = set()

    homomorph_num = 0
    template_num = 0

    em_init = "proportional"

    found_new_core_component = True
    cores_to_process.extend(core_components)
    queue = multiprocessing.Queue()

    while cores_to_process:
        print("cores left to process: ",len(cores_to_process), file=sys.stderr)
        if threads > 1:
            # Enqueue initial cores to process
            for core in cores_to_process:
                print("found core component with " + str(len(core[0].get_vertices())) + " vertices and " + str(len(core[0].get_edges())) + " edges", file=sys.stderr)
                # cutting off the expansion of large core components
                if len(core[0].get_vertices()) > skip_prop * len(genotypes):
                    print("skipping core component with " + str(len(core[0].get_vertices())) + " vertices and " + str(len(genotypes)) + " genotypes", file=sys.stderr)
                    continue
                queue.put(core)

            result_queue = multiprocessing.Queue()

            # Create worker tasks
            workers = [multiprocessing.Process(target=worker, args=(queue, result_queue, free_fixed_found, core_components, draw, cores_to_process, basis_genotype)) for _ in range(threads)]

            # Start worker processes
            for w in workers:
                w.start()

            # Stop workers
            for _ in range(threads):
                queue.put(None)
            
            # Wait for all workers to finish
            for w in workers:
                w.join()

            result_queue.put(None)

            cores_to_process.clear()
            while result_queue:
                new_core = result_queue.get()

                if new_core is None:  # Check for sentinel value
                    break
                cores_to_process.append(new_core)
                core_components.append(new_core)
        else:
            while cores_to_process:
                core_to_process = cores_to_process.pop(0)
                print("found core component with " + str(len(core_to_process[0].get_vertices())) + " vertices and " + str(len(core_to_process[0].get_edges())) + " edges", file=sys.stderr)
                
                if len(core_to_process[0].get_vertices()) > skip_prop * len(genotypes):
                    print("skipping core component with " + str(len(core_to_process[0].get_vertices())) + " vertices and " + str(len(genotypes)) + " genotypes", file=sys.stderr)
                    continue
                new_cores = process_core(core_to_process, free_fixed_found, core_components, draw, cores_to_process, basis_genotype)
                cores_to_process.extend(new_cores)
                core_components.extend(new_cores)

    print("Size of core: ", len(core_components), file=sys.stderr)
 
    best_explanations = None
    best_probability = float('-inf')
    
    num_iterations = num_it
    if greedy_min_expl is not None or greedy_genmult_min_expl is not None:
        num_iterations = 1  # only do one iteration if using greedy 

    for it in range(num_iterations):
        print(f"Running COREPHASE EM iteration {it + 1}/{num_iterations}", file=sys.stderr)
        explanations, probability = EM_free_fixed_log(core_components, em_init, genotype_multiplicities, 
                                                      args.em_component_size_threshold, args.em_min_genotype_explained,
                                                      greedy_min_expl=greedy_min_expl, greedy_genmult_min_expl=greedy_genmult_min_expl)
    
        if probability > best_probability:
            best_probability = probability
            best_explanations = explanations

    return best_explanations, best_probability


#     For each genotype, selects the haplotype pair that maximizes:
#    - If use_genmult=False: sum of haplotype degrees (number of genotypes each explains)
#    - If use_genmult=True: weighted sum considering genotype_multiplicities
#    If no pair satisfies min_expl threshold, uses the pair with the largest sum anyway.
def greedy_explanations(genotypes_bytes, gen_to_haplotypes, hap_to_freq_map, min_expl, use_genmult=False, genotype_multiplicities=None):
    
    genotype_to_explanation = {}
    total_log_prob = 0.0
    
    # Create a mapping from genotype_bytes to a weight (multiplicity)
    geno_bytes_to_weight = {}
    if use_genmult and genotype_multiplicities is not None:
        # Build genotypes array from global variable to get indices
        for g_idx in range(len(genotypes)):
            g_bytes = genotypes[g_idx].data.tobytes()
            mult = genotype_multiplicities.get(g_idx, 1)
            geno_bytes_to_weight[g_bytes] = mult
    
    for genotype_bytes in genotypes_bytes:
        haplotypes = gen_to_haplotypes.get(genotype_bytes, [])
        
        if len(haplotypes) == 0:
            # No explanations available for this genotype
            genotype_to_explanation[genotype_bytes] = None
            continue
        
        # Find all valid explanation pairs and their scores
        best_pair = None
        best_score = float('-inf')
        pairs_above_threshold = []
        
        for h1 in haplotypes:
            h2 = complement_genotype(h1, genotype_bytes)
            if h2 is None:
                continue
            h2_bytes = h2.tobytes()
            
            # Get degrees (number of genotypes explained by each haplotype)
            degree_h1 = hap_to_freq_map.get(h1, 0)
            degree_h2 = hap_to_freq_map.get(h2_bytes, 0)
            
            # Compute score based on whether we use genotype multiplicities
            if use_genmult:
                # Weight the degree sum by genotype multiplicity
                geno_weight = geno_bytes_to_weight.get(genotype_bytes, 1)
                score = (degree_h1 + degree_h2) * geno_weight
            else:
                score = degree_h1 + degree_h2
            
            # Check if both haplotypes meet the min_expl threshold
            if degree_h1 >= min_expl and degree_h2 >= min_expl:
                pairs_above_threshold.append((h1, h2_bytes, score))
            
            # Track best pair overall
            if score > best_score:
                best_score = score
                best_pair = (h1, h2_bytes)
        
        # If we have pairs above threshold, pick the best one; otherwise use best overall
        if len(pairs_above_threshold) > 0:
            # Pick the one with the highest score among those above threshold
            best_pair = max(pairs_above_threshold, key=lambda x: x[2])[:2]
        # If no pairs above threshold, best_pair already holds the pair with highest overall score
        # If still None, no valid haplotype pair exists for this genotype
        
        if best_pair is not None:
            genotype_to_explanation[genotype_bytes] = best_pair
            # Assign a log probability (since EM isn't used, just use uniform or based on frequency)
            total_log_prob += np.log(1.0)  # uniform log probability
        else:
            genotype_to_explanation[genotype_bytes] = None

    explanations = []

    for g in genotypes_bytes:
        if genotype_to_explanation[g][0] is None:
            print("all explanations equally likely for genotype ",''.join(map(str, np.frombuffer(g, dtype=np.uint8)))) 
            hap1, hap2 = generate_random_haplotype_pair(g)
            explanations.append((''.join(map(str, hap1)), ''.join(map(str, hap2))))
        else:
            explanations.append((''.join(map(str, np.frombuffer(genotype_to_explanation[g][0], dtype=np.uint8))),
                                 ''.join(map(str, np.frombuffer(genotype_to_explanation[g][1], dtype=np.uint8)))))
    
    return explanations, total_log_prob



def EM_free_fixed_log(core_components, em_init, genotype_multiplicities, component_size_threshold=1000, min_genotype_explained=1, greedy_min_expl=None, greedy_genmult_min_expl=None):

    # used to store genotypes to haplotypes later for EM
    gen_to_haplotypes = {}
    for g in genotypes:
        gen_to_haplotypes[g.data.tobytes()]=[]
    
    # now perform EM on the core components        
    gens = []        
    hap_to_freq_map = {}

    # hap_to_freq_map = {}

    # Create a set of tuples with component and the number of vertices
    component_and_sizes = [(component[0], component[0].num_vertices(), component[1])  for component in core_components]
    sorted_component_and_sizes = sorted(component_and_sizes, key=lambda x: x[1], reverse=True)

    apply_heuristic = False

    for idx, component_tuple in enumerate(sorted_component_and_sizes):
        (component, basis_genotype) = component_tuple[0], component_tuple[2]
        
        # Check if we should apply the heuristic
        if not apply_heuristic:
            apply_heuristic = component.num_vertices() > component_size_threshold
        
        # First pass: compute vertex degrees (number of edges) for heuristic filtering
        vertex_degrees = {}
        if apply_heuristic:
            for vertex in component.get_vertices():
                vertex_degrees[vertex] = len(component.get_all_edges(vertex))
        
        total_cnt = 0
        for vertex in component.get_vertices():
            num_adjacent = len(component.get_all_edges(vertex))
            free_fixed_v = np.zeros((M), dtype=np.uint8)
            free_fixed_v = fast_compute_genotype(component.vp["fingerprint"][vertex].get_array(), basis_genotype, free_fixed_v)  # gets free fixed for target node
            tmp = free_fixed_v
            node_genotype = tmp.tobytes()
            gens.append(node_genotype)
            if node_genotype not in hap_to_freq_map:
                hap_to_freq_map[node_genotype] = 0
            hap_to_freq_map[node_genotype] += num_adjacent
            total_cnt += num_adjacent

            for edge in component.get_all_edges(vertex):
                edge_genotype = genotypes[component.ep["genotype"][edge]]
                gen_to_haplotypes[edge_genotype.tobytes()].append(node_genotype)

    # Apply heuristic: prune haplotype pairs with low degrees
    if apply_heuristic and min_genotype_explained > 1:
        print(f"Applying heuristic: component_size_threshold={component_size_threshold}, min_genotype_explained={min_genotype_explained}", file=sys.stderr)
        
        new_hap_to_freq_map = {}
        # For each genotype, filter explanations
        for genotype_bytes in gen_to_haplotypes:
            haplotypes = gen_to_haplotypes[genotype_bytes]
            
            if len(haplotypes) > 0:
                # Find valid explanations: both haplotypes must explain at least min_genotype_explained genotypes
                valid_pairs = []
                for i in range(0, len(haplotypes)):
                    h1 = haplotypes[i]
                    # get the complement haplotype
                    h2 = complement_genotype(h1, genotype_bytes).tobytes() 
                    
                    # Get degrees (number of genotypes explained by each haplotype)
                    degree_h1 = hap_to_freq_map.get(h1, 0)
                    degree_h2 = hap_to_freq_map.get(h2, 0)
                    
                    if degree_h1 >= min_genotype_explained and degree_h2 >= min_genotype_explained:
                        valid_pairs.extend([h1, h2])
                
                # If all pairs were pruned, keep the pair with the largest sum of degrees
                if len(valid_pairs) == 0 and len(haplotypes) > 0:
                    best_pair = None
                    best_sum = 0
                    
                    for i in range(0, len(haplotypes)):
                        h1 = haplotypes[i]
                        h2 = complement_genotype(h1, genotype_bytes).tobytes() 
                        degree_sum = hap_to_freq_map.get(h1, 0) + hap_to_freq_map.get(h2, 0)
                        
                        if degree_sum > best_sum:
                            best_sum = degree_sum
                            best_pair = [h1, h2]
                    
                    if best_pair:
                        valid_pairs = best_pair
                
                gen_to_haplotypes[genotype_bytes] = valid_pairs
                # add valid_pairs to new_hap_to_freq_map
                for h in valid_pairs:
                    if h not in new_hap_to_freq_map:
                        new_hap_to_freq_map[h] = 0
                    new_hap_to_freq_map[h] += 1
        hap_to_freq_map = new_hap_to_freq_map

    # print to stderr the number of haplotypes being used
    print(f"Total unique haplotypes before EM: {len(hap_to_freq_map)}", file=sys.stderr)

    genotypes_bytes = [g.data.tobytes() for g in genotypes]

    # Check if greedy heuristics should be used instead of EM
    if greedy_min_expl is not None:
        print(f"Using --greedy heuristic with min_expl={greedy_min_expl}", file=sys.stderr)
        return greedy_explanations(genotypes_bytes, gen_to_haplotypes, hap_to_freq_map, greedy_min_expl, use_genmult=False, genotype_multiplicities=genotype_multiplicities)
    
    if greedy_genmult_min_expl is not None:
        print(f"Using --greedy_genmult heuristic with min_expl={greedy_genmult_min_expl}", file=sys.stderr)
        return greedy_explanations(genotypes_bytes, gen_to_haplotypes, hap_to_freq_map, greedy_genmult_min_expl, use_genmult=True, genotype_multiplicities=genotype_multiplicities)

    # Apply the selected frequency initialization method
    if em_init == "proportional":
        # Proportional to the number of adjacent edges (original functionality)
        for h in hap_to_freq_map:
            hap_to_freq_map[h] = np.log(hap_to_freq_map[h] / total_cnt)
    elif em_init == "random":
        # Equal frequencies for all haplotypes (uniform distribution)
        for h in hap_to_freq_map:
            hap_to_freq_map[h] = np.log(1 / len(hap_to_freq_map))
    elif em_init == "proprandom":
        # Sample frequencies proportionally to the number of adjacent edges
        # Normalize counts to probabilities for sampling
        freq_counts = hap_to_freq_map.copy()
        frequencies = np.random.dirichlet([freq_counts[h] for h in hap_to_freq_map])
        for h, freq in zip(hap_to_freq_map.keys(), frequencies):
            hap_to_freq_map[h] = np.log(freq)
    else:
        raise ValueError(f"Unknown em_init method: {em_init}")


    return expectation_maximization_log(genotypes_bytes, hap_to_freq_map, gen_to_haplotypes, genotype_multiplicities)


def expectation_maximization_log(genotypes, freqs, genotype_to_haplotypes, genotype_to_multiplicity=None):
    from scipy.special import logsumexp
    
    genotype_to_explanation_to_probability = {}
    # compute explanations for each genotype
    for g in genotypes:
        genotype_to_explanation_to_probability[g]={}
        used = set()
        for h in genotype_to_haplotypes[g]:
            if h not in used:
                for h2 in genotype_to_haplotypes[g]:
                    if h2 not in used:
                        compat_g = compatible(h,h2,g)
                        if compat_g is not None:
                            used.add(h)
                            used.add(h2)
                            genotype_to_explanation_to_probability[g][(h,h2)]=float('-inf')

    prev_prob = float('-inf')

    # Preprocessing step: iterate through all genotypes and explanations and remove an explanation if one of its haplotypes is not used anywhere else AND there is at least one other explanation for that genotype
    # Count haplotype usage across all genotypes
    haplotype_usage_count = {}
    for g in genotypes:
        for explanation in genotype_to_explanation_to_probability[g]:
            hap1, hap2 = explanation
            haplotype_usage_count[hap1] = haplotype_usage_count.get(hap1, 0) + 1
            haplotype_usage_count[hap2] = haplotype_usage_count.get(hap2, 0) + 1
    
    # Remove redundant explanations
    for g in genotypes:
        explanations_to_remove = []
        for explanation in genotype_to_explanation_to_probability[g]:
            hap1, hap2 = explanation
            # Check if at least one haplotype is used only in this explanation
            hap1_used_once = haplotype_usage_count[hap1] == 1
            hap2_used_once = haplotype_usage_count[hap2] == 1
            
            # Check if there are other explanations for this genotype
            num_explanations = len(genotype_to_explanation_to_probability[g])
            
            # Remove if: (one haplotype used only here) AND (at least 2 explanations exist)
            if (hap1_used_once or hap2_used_once) and num_explanations > 1:
                explanations_to_remove.append(explanation)
        
        # Remove the redundant explanations
        for explanation in explanations_to_remove:
            del genotype_to_explanation_to_probability[g][explanation]
            # Decrement usage counts for removed explanation
            hap1, hap2 = explanation
            haplotype_usage_count[hap1] -= 1
            haplotype_usage_count[hap2] -= 1
        
        # If all explanations were removed for a genotype (shouldn't happen), keep one at random
        if len(genotype_to_explanation_to_probability[g]) == 0:
            # Find an explanation with haplotypes used elsewhere
            all_explanations = list(genotype_to_haplotypes[g])
            # Reconstruct valid explanation pairs
            used = set()
            for h in all_explanations:
                if h not in used:
                    for h2 in all_explanations:
                        if h2 not in used:
                            compat_g = compatible(h, h2, g)
                            if compat_g is not None:
                                genotype_to_explanation_to_probability[g][(h, h2)] = float('-inf')
                                haplotype_usage_count[h] += 1
                                haplotype_usage_count[h2] += 1
                                used.add(h)
                                used.add(h2)
                                break
                    if len(genotype_to_explanation_to_probability[g]) > 0:
                        break

    for k in range(1000):
        # compute the explanation log probabilities for each genotype given fixed haplotype frequencies
        # COMPUTE THE DENOMINATOR FOR Algorithm 2
        for gidx in np.random.permutation(range(len(genotypes))):
            g = genotypes[gidx]
            
            # Calculate log probabilities for each explanation (do NOT normalize)
            for explanation in genotype_to_explanation_to_probability[g]:
                # Sum haplotype frequencies in log space
                if explanation[0] == explanation[1]:
                    # Homozygous case
                    # entropy of the explanation distribution
                    if freqs[explanation[0]] == float('-inf'):
                        entropy = 0
                    else:
                        entropy = - (2 * np.exp(freqs[explanation[0]]) * (freqs[explanation[0]])) 
                    genotype_to_explanation_to_probability[g][explanation] = np.log(2) + freqs[explanation[0]] + freqs[explanation[1]] + 0*entropy
                else:
                    if freqs[explanation[0]] == float('-inf') or freqs[explanation[1]] == float('-inf'):
                        entropy = 0
                    else:  
                        entropy = - (np.exp(freqs[explanation[0]]) * (freqs[explanation[0]])) - (np.exp(freqs[explanation[1]]) * (freqs[explanation[1]]))
                    genotype_to_explanation_to_probability[g][explanation] = freqs[explanation[0]] + freqs[explanation[1]] + 0*entropy
                
            # normalization constant
            norm_const = logsumexp(list(genotype_to_explanation_to_probability[g].values()))

            # Normalize log probabilities
            for explanation in genotype_to_explanation_to_probability[g]:
                genotype_to_explanation_to_probability[g][explanation] -= norm_const
                            
        # compute the new haplotype frequencies given those that are used to explain the genotypes
        # Initialize log frequencies
        freq_sums = {h: 0 for h in freqs.keys()}
        
        normalizing_constant = 0.0

        # Compute new haplotype frequencies in log space
        for g_idx, g in enumerate(genotypes):
            n_copies = genotype_multiplicities[g_idx] if genotype_multiplicities is not None else 1
            
            for explanation in genotype_to_explanation_to_probability[g]:
                log_prob = genotype_to_explanation_to_probability[g][explanation]
                
                if explanation[0] == explanation[1]:
                    # Add log(2) for homozygous cases
                    freq_sums[explanation[0]] = freq_sums[explanation[0]] + n_copies * 2 * np.exp(log_prob) 
                    normalizing_constant += n_copies * 2 * np.exp(log_prob)
                else:
                    freq_sums[explanation[0]] = freq_sums[explanation[0]] + n_copies * np.exp(log_prob)
                    freq_sums[explanation[1]] = freq_sums[explanation[1]] + n_copies * np.exp(log_prob)
                    normalizing_constant += n_copies * 2 * np.exp(log_prob)
        

        # Normalize frequencies in log space
        # Account for genotype multiplicities when calculating total samples
        for h in freqs.keys():
            if freq_sums[h] == 0:
                freqs[h] = float('-inf')  # log(0) = -inf
            else:
                freqs[h] = np.log(freq_sums[h]) - np.log(normalizing_constant)

        # Compute complete data log likelihood (proper EM objective)
        # This should be the log marginal probability summed over all genotypes
        total_prob = 0
        genotype_to_expl = {}
        for g_idx, g in enumerate(genotypes):
            n_copies = genotype_multiplicities[g_idx] if genotype_multiplicities is not None else 1
            
            # Calculate log probability for each explanation
            log_probs = []
            max_expl = None
            max_log_prob = float('-inf')
            
            for explanation, log_prob in genotype_to_explanation_to_probability[g].items():
                log_probs.append(log_prob)
                if log_prob > max_log_prob:
                    max_log_prob = log_prob
                    max_expl = explanation
            
            # Compute marginal log likelihood using logsumexp
            if log_probs:
                genotype_log_likelihood = logsumexp(log_probs)
            else:
                genotype_log_likelihood = float('-inf')
            
            # Accumulate log likelihood (multiplied by genotype multiplicity)
            total_prob += n_copies * genotype_log_likelihood
            genotype_to_expl[g] = (max_expl, max_log_prob)
        
        print(np.exp(total_prob), file=sys.stderr)

        if np.isclose(total_prob, prev_prob, atol=1e-10):
            break
        prev_prob = total_prob

    explanations = []
    for g in genotypes:
        if genotype_to_expl[g][0] is None:
            print("all explanations equally likely for genotype ",''.join(map(str, np.frombuffer(g, dtype=np.uint8)))) 
            hap1, hap2 = generate_random_haplotype_pair(g)
            explanations.append((''.join(map(str, hap1)), ''.join(map(str, hap2))))
        else:
            explanations.append((''.join(map(str, np.frombuffer(genotype_to_expl[g][0][0], dtype=np.uint8))),
                                 ''.join(map(str, np.frombuffer(genotype_to_expl[g][0][1], dtype=np.uint8)))))
    return explanations, total_prob


def generate_random_haplotype_pair(g):
    genotype = np.frombuffer(g, dtype=np.uint8)
    hap1 = np.zeros_like(genotype)
    hap2 = np.zeros_like(genotype)
    
    # Copy fixed positions (0s and 1s)
    mask_0 = genotype == 0
    mask_1 = genotype == 1
    mask_2 = genotype == 2
    
    # For positions with 0s and 1s, both haplotypes get same value
    hap1[mask_0] = 0
    hap2[mask_0] = 0
    hap1[mask_1] = 1
    hap2[mask_1] = 1
    
    # For positions with 2s, randomly assign 0/1 to first haplotype
    # and give opposite value to second haplotype
    random_bits = np.random.randint(0, 2, size=len(genotype))
    hap1[mask_2] = random_bits[mask_2]
    hap2[mask_2] = 1 - random_bits[mask_2]
    
    return hap1, hap2


if(__name__ == "__main__"):

    parser = argparse.ArgumentParser(description="CORE Phase.")
    parser.add_argument("--filename", type=str, help="The input file name",default="input.vcf")
    parser.add_argument("--output_prefix", type=str, help="The output file prefix",default="output")
    parser.add_argument("--input_filetype", type=str, choices=["raw", "vcf"], default="vcf", help="The input file type (default: vcf)")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--threads", type=int, help="The number of threads to use",default=1)
    parser.add_argument("--draw", action="store_true", help="Draw the core components", default=False)
    parser.add_argument("--txtout", action="store_true", help="Write VCF instead of BCF", default=True)
    parser.add_argument("--skipprop", type=float, help="The size of a component to skip, given as a proportion of the number of input genotypes. E.g., "
                                                        "if --skipprop=1.5, then a core component will not be processed if its size is greater than 1.5*num_genotypes.",default=1)
    parser.add_argument("--em_component_size_threshold", type=int, help="If a component has more vertices than this threshold, trigger heuristic in EM",default=10000)
    parser.add_argument("--em_min_genotype_explained", type=int, help="Minimum number of genotypes (degree) a haplotype must explain to be considered in valid explanations",default=1 )
    parser.add_argument("--num_it", type=int, help="The number of times to run corephase and keep only the highest probability explanations",default=1)
    parser.add_argument("--em_init", type=str, choices=["proportional", "random", "proprandom"], help="Haplotype frequency initialization method: 'proportional' (default, proportional to adjacent edges), 'random' (equal frequencies), 'proprandom' (sampled proportionally to adjacent edges)",default="proportional")
    parser.add_argument("--greedy", type=int, metavar="MIN_EXPL", help="Use greedy heuristic: select haplotype pairs that maximize the sum of degrees. If no pairs satisfy min_expl, use the pair with largest degree sum.", default=None)
    parser.add_argument("--greedy_genmult", type=int, metavar="MIN_EXPL", help="Use greedy heuristic considering genotype multiplicities: select haplotype pairs that maximize the weighted sum of degrees. If no pairs satisfy min_expl, use the pair with largest weighted degree sum.", default=None)

    args = parser.parse_args()
    args.draw = False
    start_time = time.time()

    if args.test:

        genotype_multiplicities = None
        filename=None
        genotypes = np.zeros((4,5),dtype=np.uint8)
        genotypes[0][0]=1
        genotypes[0][1]=2
        genotypes[0][2]=2
        genotypes[0][3]=2
        genotypes[0][4]=0

        genotypes[1][0]=2
        genotypes[1][1]=2
        genotypes[1][2]=0
        genotypes[1][3]=1
        genotypes[1][4]=2

        genotypes[2][0]=2
        genotypes[2][1]=2
        genotypes[2][2]=1
        genotypes[2][3]=2
        genotypes[2][4]=0

        genotypes[3][0]=2
        genotypes[3][1]=1
        genotypes[3][2]=2
        genotypes[3][3]=2
        genotypes[3][4]=2

        # Set the random seed
        # np.random.seed(42)

        # Generate 10 random genotypes of length 15
        genotypes = np.random.randint(0, 3, size=(30, 10), dtype=np.uint8)
        # retain only unique genotypes
        genotypes = np.unique(genotypes, axis=0)


        print(genotypes, file=sys.stderr)

        genotypes.flags.writeable = False

    else:

        filename=args.filename
        if filename.endswith(".gz"):
            gen_filename = gzip.open(filename, "rb")
        else:
            gen_filename = open(filename, "r")

        if args.input_filetype == "raw":
            genotypes=np.genfromtxt(filename, delimiter=1, dtype=np.uint8)
        elif args.input_filetype == "vcf" or args.input_filetype == "bcf":
            input_haplotype = open_bcf_to_read(filename)
            sample_names = list(input_haplotype.header.samples)
            num_samples = 0
            num_vars = 0
            for input_vcf_rec in input_haplotype.fetch():
                num_vars+=1
                if num_vars == 1:
                    num_samples = len(input_vcf_rec.samples)

            input_haplotype.close()
            genotypes=np.zeros((num_samples,num_vars),dtype=np.uint8)

            input_haplotype = open_bcf_to_read(filename)
            var_no = 0
            for input_vcf_rec in input_haplotype.fetch():
                sample_no = 0
                for sample in input_vcf_rec.samples:
                    output_gt = input_vcf_rec.samples[sample]['GT']
                    genotypes[sample_no,var_no]=output_gt[0]+output_gt[1]
                    sample_no+=1
                var_no+=1
            input_haplotype.close()

        # create matrix of unique genotype rows and keep multiplicity (preserve first-occurrence order)
        genotypes_original = genotypes.copy()
        _uniques, _first_idx, _counts = np.unique(genotypes_original, axis=0, return_index=True, return_counts=True)
        _order = np.argsort(_first_idx)
        genotypes = _uniques[_order]
        multiplicities = _counts[_order].tolist()

        # build map from sample name -> index in the new (unique) genotypes array
        # map each unique row to its new index, then map each sample by its original row
        row_to_new_idx = {tuple(row.tolist()): i for i, row in enumerate(genotypes)}
        sample_to_genotype_idx = {}
        for sample_idx, sample in enumerate(sample_names):
            sample_row = tuple(genotypes_original[sample_idx].tolist())
            sample_to_genotype_idx[sample] = row_to_new_idx[sample_row]

        genotype_multiplicities = {}
        # compute the number of times each genotype appears based on the genotype matrix
        for sample_idx, sample in enumerate(sample_names):  
            g_idx = sample_to_genotype_idx[sample]
            genotype_multiplicities[g_idx] = genotype_multiplicities.get(g_idx, 0) + 1

        genotypes[genotypes == 2] = 3
        genotypes[genotypes == 1] = 2
        genotypes[genotypes == 3] = 1
        gen_filename.close()
        
        # optionally make the unique matrix read-only
        genotypes.flags.writeable = False

    M=genotypes.shape[1]

    print("Running COREPHASE", file=sys.stderr)
    
    best_explanations, best_probability = core_phase(genotypes, args.output_prefix, genotype_multiplicities, args.threads, args.draw, args.skipprop, args.em_init, args.num_it, greedy_min_expl=args.greedy, greedy_genmult_min_expl=args.greedy_genmult)
        
    explanations = best_explanations
    probability = best_probability
    print(f"Best probability found: {probability}", file=sys.stderr)



    # write out VCF file output
    vin = VariantFile(filename, "r")
    # create output VariantFile using the same header

    if args.txtout:
        mode = "w"
        vout = VariantFile(args.output_prefix + ".vcf", mode, header=vin.header)
    else:
        mode = "wb" 
        vout = VariantFile(args.output_prefix + ".bcf", mode, header=vin.header)

    var_idx = 0
    for rec in vin:
        # If ALT missing or empty, set a single placeholder ALT
        if rec.alts is None or len(rec.alts) == 0:
            # pysam requires tuple for rec.alts
            rec.alts = ("N",)

        # change the record to reflect the alleles in explanations 
        for sample_idx, sample_name in enumerate(rec.samples):
            major = int(explanations[sample_to_genotype_idx[sample_name]][0][var_idx])
            minor = int(explanations[sample_to_genotype_idx[sample_name]][1][var_idx])
            if major not in (0,1) or minor not in (0,1): # if neither allele is specified, then any allele is equally likely 
                # set major to 0 or 1 at random
                major = np.random.randint(2)
                # set minor to the opposite of major
                minor = 1 - major
            rec.samples[sample_name]['GT'] = tuple((major, minor))
            rec.samples[sample_name].phased = True

        vout.write(rec)
        var_idx+=1

    vin.close()
    vout.close()
    print("wrote file out to " + args.output_prefix + ".bcf", file=sys.stderr)
    print("Process finished --- %s seconds ---" % (time.time() - start_time), file=sys.stderr)
