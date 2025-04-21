import networkx as nx
from itertools import combinations
import numpy as np
import scipy

from collections import defaultdict
def shen_modularity(G, community_list):
    """
    G: NetworkX graph (undirected, unweighted)
    community_list: list of sets, each set is a community (can overlap)

    Returns: Shen modularity (float)
    """
    m = G.number_of_edges()
    if m == 0:
        return 0

    # Compute degrees
    degrees = dict(G.degree())

    # Compute O: number of communities each node belongs to
    O = {}
    for cid, community in enumerate(community_list):
        for node in community:
            O[node] = O.get(node, 0) + 1

    Q = 0.0
    nodes = list(G.nodes())
    for i in nodes:
        for j in nodes:
            A_ij = 1 if G.has_edge(i, j) else 0
            k_i = degrees[i]
            k_j = degrees[j]
            O_i = O.get(i, 1)
            O_j = O.get(j, 1)

            # Check how many communities i and j share
            shared = 0
            for community in community_list:
                if i in community and j in community:
                    shared += 1

            if shared > 0:
                delta_Q = shared * (A_ij - (k_i * k_j) / (2 * m)) / (O_i * O_j)
                Q += delta_Q

    Q /= (2 * m)
    return Q

import networkx as nx

def lazar_modularity(G, community_list):
    """
    G: neorientovaný, neohodnocený NetworkX graf
    community_list: seznam množin, každá množina reprezentuje komunitu (může být překryv)
    Vrací: Lazarovu modularitu Mov (float)
    """

    # Stupně uzlů
    degrees = dict(G.degree())

    # Počet komunit, do kterých každý uzel patří
    O = {}
    for community in community_list:
        for node in community:
            O[node] = O.get(node, 0) + 1

    # Počet komunit
    K = len(community_list)

    # Výpočet Mov
    total_Mov = 0.0
    for community in community_list:
        n_c = len(community)
        if n_c <= 1:
            continue  # hustota nedefinována, přeskakujeme nebo považujeme za 0

        e_c = 0  # počet vnitřních hran
        for u in community:
            for v in community:
                if u < v and G.has_edge(u, v):
                    e_c += 1

        density = e_c / (n_c * (n_c - 1) / 2)

        sum_contributions = 0.0
        for i in community:
            d_i = degrees[i]
            if d_i == 0 or i not in O:
                continue  # izolovaný uzel nebo bez komunity

            in_edges = sum(1 for j in community if j != i and G.has_edge(i, j))
            out_edges = sum(1 for j in G.neighbors(i) if j not in community)

            contrib = (in_edges - out_edges) / (d_i * O[i])
            sum_contributions += contrib

        Mov_c = (sum_contributions / n_c) * density
        total_Mov += Mov_c

    Mov = total_Mov / K if K > 0 else 0
    return Mov

import numpy as np
from sklearn.metrics import mutual_info_score
from math import log

import math
import scipy as sp

logBase = 2


def partialEntropyAProba(proba: float) -> float:
    if proba == 0:
        return 0
    return -proba * math.log(proba, logBase)


def coverEntropy(cover: list[set], allNodes: set) -> float:
    allEntr = []
    for com in cover:
        fractionIn = len(com) / len(allNodes)
        allEntr.append(sp.stats.entropy([fractionIn, 1 - fractionIn], base=logBase))
    return sum(allEntr)


def comPairConditionalEntropy(cl: set, clKnown: set, allNodes: set) -> float:
    nbNodes = len(allNodes)

    a = len((allNodes - cl) - clKnown) / nbNodes
    b = len(clKnown - cl) / nbNodes
    c = len(cl - clKnown) / nbNodes
    d = len(cl & clKnown) / nbNodes

    entropyKnown = sp.stats.entropy(
        [len(clKnown) / nbNodes, 1 - len(clKnown) / nbNodes], base=logBase
    )
    conditionalEntropy = sp.stats.entropy([a, b, c, d], base=logBase) - entropyKnown

    return conditionalEntropy


def coverConditionalEntropy(
    cover: list[set], coverRef: list[set], allNodes: set
) -> float:
    allMatches = []
    for com in cover:
        matches = [
            comPairConditionalEntropy(com, com2, allNodes) for com2 in coverRef
        ]
        bestMatchEntropy = min(matches)
        allMatches.append(bestMatchEntropy)
    return sum(allMatches)


def mgh_onmi(cover_a: list[set], cover_b: list[set]) -> float:
    """
    McDaid et al. MGH ONMI: Normalized Mutual Information for overlapping covers.
    cover_a: detected communities (list of sets)
    cover_b: ground truth communities (list of sets)
    """
    if (len(cover_a) == 0 and len(cover_b) != 0) or (
        len(cover_a) != 0 and len(cover_b) == 0
    ):
        return 0.0
    if cover_a == cover_b:
        return 1.0

    allNodes = {n for c in cover_a for n in c} | {n for c in cover_b for n in c}

    HX = coverEntropy(cover_a, allNodes)
    HY = coverEntropy(cover_b, allNodes)

    H_X_given_Y = coverConditionalEntropy(cover_a, cover_b, allNodes)
    H_Y_given_X = coverConditionalEntropy(cover_b, cover_a, allNodes)

    IXY = 0.5 * (HX - H_X_given_Y + HY - H_Y_given_X)
    NMI = IXY / max(HX, HY)

    # Safety check
    if math.isnan(NMI) or round(NMI, 2) < 0 or round(NMI, 2) > 1:
        print("NMI: %s  from %s %s %s %s " % (NMI, H_X_given_Y, H_Y_given_X, HX, HY))
        raise Exception("Incorrect NMI value")

    return NMI




import numpy as np
import scipy

class NF1(object):
    def __init__(self, communities, ground_truth):
        self.communities = communities
        self.ground_truth = ground_truth
        self.__compute_precision_recall()

    def get_f1(self):
        """
        :return: a tuple composed by (average_f1, std_f1)
        """
        f1_list = np.array(self.__communities_f1())
        return np.mean(f1_list)  # Only return the average F1 score

    def __compute_precision_recall(self):
        """
        :return: a list of tuples (precision, recall)
        """
        # ground truth
        gt_coms = {cid: nodes for cid, nodes in enumerate(self.ground_truth)}
        node_to_com = defaultdict(list)
        for cid, nodes in gt_coms.items():
            for n in nodes:
                node_to_com[n].append(cid)

        # community
        ext_coms = {cid: nodes for cid, nodes in enumerate(self.communities)}
        prl = []

        for cid, nodes in ext_coms.items():
            ids = {}
            for n in nodes:
                try:
                    # community in ground truth
                    idd_list = node_to_com[n]
                    for idd in idd_list:
                        if idd not in ids:
                            ids[idd] = 1
                        else:
                            ids[idd] += 1
                except KeyError:
                    pass

            try:
                # identify the maximal match ground truth communities (label) and their absolute frequency (p)
                maximal_match = {label: p for label, p in ids.items() if p == max(ids.values())}
                for label, p in maximal_match.items():
                    precision = float(p) / len(nodes)
                    recall = float(p) / len(gt_coms[label])
                    prl.append((precision, recall))
            except (ZeroDivisionError, ValueError):
                pass

        self.prl = prl
        return prl

    def __communities_f1(self):
        """
        :return: list of community f1 scores
        """
        f1s = []
        for l in self.prl:
            x, y = l[0], l[1]
            z = 2 * (x * y) / (x + y)
            f1s.append(z)
        return f1s

