import re
import networkx as nx 
from typing import List, Optional
import sqlite3
import os
import pandas as pd
from tqdm import tqdm
import logging
from collections import defaultdict
from pathlib import Path
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import numpy as np
from rapidfuzz import fuzz
from Levenshtein import distance as lev_dist
from Levenshtein import jaro_winkler
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from graph_tool.all import Graph, minimize_nested_blockmodel_dl, mcmc_equilibrate, BlockState
import graph_tool.inference as gt_inf

sys.path.append(str(Path(__file__).parent.parent.parent))
from amici.utils.normalizers import normalize_interest_group_name, shorten_common_terms

class DedupeGraph(nx.Graph):
    """
    A class that represents a deduplication graph for interest groups.
    It extends the NetworkX Graph class to provide additional functionality
    for deduplication tasks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_interest_group(self, name: str, doc: int):
        """
        Add an interest group to the graph.

        Args:
            name (str): The name of the interest group.
            id (int): The ID of the interest group.
        """
        normalized_name = normalize_interest_group_name(name)

        # Add the interest group to the graph
        if normalized_name not in self.nodes:
            self.add_node(normalized_name, docs={doc:name}, type_='amicus')
        elif doc not in self.nodes[normalized_name]['docs']:
            self.nodes[normalized_name]['docs'][doc] = name
        else:
            logging.warning(f"Interest group {normalized_name} ({name}) already exists in document {doc}.")

        return normalized_name

    def add_docket(self, docket: str, doc: int):
        """
        Add a docket to the graph.

        Args:
            docket (str): The docket number, in "YY-NNNN" format.
        """
        assert re.match(r'^\d{2}-\d+$', docket), f"Docket {docket} does not match the required format 'YY-NNNN'"

        if docket in self.nodes:
            assert self.nodes[docket]['type_'] == 'docket' # should only be false in the weird case that an amicus has a name that looks like a docket.
            self.nodes[docket]["docs"].add(doc)
        else:
            self.add_node(docket, type_='docket', docs=set([doc]))

        return docket

    def add_position(self, name: str, docket: str, pos: str):
        """
        Add an amicus position on a particular docket.

        Args:
            name (str): The name of the interest group.
            docket (str): The docket number (in "YY-NNNN" format).
            pos (str): The position, either "P" for supports petitioner or "R" for supports respondent.

        Raises:
            ValueError: If the interest group or docket doesn't exist in the graph.
            ValueError: If the position is not "P" or "R".
        """
        normalized_name = normalize_interest_group_name(name)

        assert re.match(r'^\d{2}-\d+$', docket), f"Docket {docket} does not match the required format 'YY-NNNN'"
        
        # Check that nodes exist
        if normalized_name not in self.nodes:
            raise ValueError(f"Interest group {name} not found in graph.")
        if docket not in self.nodes:
            raise ValueError(f"Docket {docket} not found in graph.")

        # Make sure that the docket's "doc" value is in the amicus's "docs" dictionary keys
        if len(self.nodes[docket]["docs"].intersection(set(self.nodes[normalized_name]["docs"]))) == 0:
            raise ValueError(f"Interest group {name} does not appear in docket {self[docket]}")
        
        # Validate position
        if pos not in ["P", "R"]:
            raise ValueError(f"Position must be 'P' (petitioner) or 'R' (respondent), got {pos}")
        
        # Add the position edge
        self.add_edge(normalized_name, docket, position=pos, type_='position')

    def add_probable_match(self, name1: str, name2: str, p: float=1.0):
        """
        Add a match between two interest groups.

        Args:
            name1 (str): The name of the first interest group.
            name2 (str): The name of the second interest group.
        """
        normalized_name1 = normalize_interest_group_name(name1)
        normalized_name2 = normalize_interest_group_name(name2)

        if normalized_name1 == normalized_name2:
            raise ValueError(f"Cannot add match between identical interest groups: {name1} and {name2}.")
        elif normalized_name1 not in self.nodes:
            raise ValueError(f"Interest group {name1} not found in graph.")
        elif normalized_name2 not in self.nodes:
            raise ValueError(f"Interest group {name2} not found in graph.")

        u1 = self.nodes[normalized_name1]
        u2 = self.nodes[normalized_name2]

        docs1 = u1['docs']
        docs2 = u2['docs']

        # Check if the names are associated with any of the same documents
        # Since each interest group should only appear once in a given document, there should not be matches
        # between interest groups that appear in the same document.
        if any(doc in docs1 for doc in docs2):
            raise ValueError(f"Interest groups {name1} and {name2} are associated with the same document.")
        
        # Add the match to the graph
        self.add_edge(normalized_name1, normalized_name2, weight=p, type_='probable_match')

    def check_match_legality(self, name1: str, name2: str):
        """
        Check if a match between two interest groups is legal.
        A match is legal if the two interest groups are not associated with the same document.
        Args:
            name1 (str): The name of the first interest group.
            name2 (str): The name of the second interest group.
        Returns:
            bool: True if the match is legal, False otherwise.
        """
        normalized_name1 = normalize_interest_group_name(name1)
        normalized_name2 = normalize_interest_group_name(name2)

        if normalized_name1 == normalized_name2:
            raise ValueError(f"Cannot check legality of match between identical interest groups: {name1} and {name2}.")
        elif normalized_name1 not in self.nodes:
            raise ValueError(f"Interest group {name1} not found in graph.")
        elif normalized_name2 not in self.nodes:
            raise ValueError(f"Interest group {name2} not found in graph.")

        u1 = self.nodes[normalized_name1]
        u2 = self.nodes[normalized_name2]

        docs1 = u1['docs']
        docs2 = u2['docs']

        # Check if the names are associated with any of the same documents
        return not any(doc in docs1 for doc in docs2)

    def get_match_component(self, name):
        """
        Get the connected component of all nodes with confirmed matches to a given node.

        Args:
            name (str): The name of the interest group.
        Returns:
            set: A set of interest group names in the connected component.
        """
        normalized_name = normalize_interest_group_name(name)

        if normalized_name not in self.nodes:
            raise ValueError(f"Interest group {name} not found in graph.")

        # If the node does not have any match edges, just return the node itself
        if not any(self.has_edge(normalized_name, v) and 
            self.edges[normalized_name, v]['type_'] == 'match' 
            for v in self.neighbors(normalized_name)):

            return {normalized_name}

        match_edges = [(u, v) for u, v in self.edges if self.edges[u, v]['type_'] == 'match']
        match_subgraph = self.edge_subgraph(match_edges)

        # Get the connected component of the node
        component = set(nx.node_connected_component(match_subgraph, normalized_name))

        return component

    def label_match(self, name1: str, name2: str, is_match: bool):
        """
        Label two interest groups as either a match or a mismatch.
        When a match/mismatch is confirmed we need to update nearby matches accordingly:
            1. Get the confirmed-match components of nodes 1 and 2 (c1 and c2, respectively).
            2. Check that there are no conflicting labels between any pairs from c1 and c2. 
                - If there are, raise an error
            3. Add confirmed matches/mismatches between all pairs in c1 and c2.

        Args:
            name1 (str): The name of the first interest group.
            name2 (str): The name of the second interest group.
            is_match (bool): True if the entities match, False if they don't match.
            
        Returns:
            list: A list of (u, v, bool) tuples representing new edges added, where the boolean
                  indicates whether u and v match.
        """
        normalized_name1 = normalize_interest_group_name(name1)
        normalized_name2 = normalize_interest_group_name(name2)

        if normalized_name1 == normalized_name2:
            raise ValueError(f"Cannot label match between identical interest groups: {name1} and {name2}.")
        elif normalized_name1 not in self.nodes:
            raise ValueError(f"Interest group {name1} not found in graph.")
        elif normalized_name2 not in self.nodes:
            raise ValueError(f"Interest group {name2} not found in graph.")

        # Get the connected components of the two nodes
        c1 = self.get_match_component(normalized_name1)
        c2 = self.get_match_component(normalized_name2)
        
        if is_match:
            # Handle match case
            if c1 == c2:
                print(f"Interest groups {name1} and {name2} are already in the same component.")
                return []  # Names are already in the same component, no updates needed
            
            # Check for confirmed mismatches
            for u in c1:
                for v in c2:
                    if self.has_edge(u, v) and self.edges[u, v]['type_'] == 'mismatch':
                        raise ValueError(f"Confirmed mismatch between {u} and {v}, but match given between {normalized_name1} and {normalized_name2}.")
        else:
            # Handle mismatch case
            if c1 == c2:
                raise ValueError(f"Interest groups {name1} and {name2} are in the same component, so they cannot be mismatched.")
            
            # Check for confirmed matches
            for u in c1:
                for v in c2:
                    if self.has_edge(u, v) and self.edges[u, v]['type_'] == 'match':
                        raise ValueError(f"Confirmed match between {u} and {v}, but mismatch given between {normalized_name1} and {normalized_name2}.")
        
        # Add confirmed matches or mismatches between all pairs in c1 and c2
        updates = []
        edge_type = 'match' if is_match else 'mismatch'
        
        for u in c1:
            for v in c2:
                if self.has_edge(u, v):
                    if self.edges[u, v]['type_'] == edge_type:
                        continue
                    elif self.edges[u, v]['type_'] == 'probable_match':
                        # If the edge is a probable match, we need to remove it
                        self.remove_edge(u, v)
                    else:
                        raise ValueError(f"Edge between {u} and {v} is not a {edge_type}: {self.edges[u, v]}.")
                
                updates.append((u, v, is_match))
                self.add_edge(u, v, type_=edge_type)

        # If this is a match, add confirmed mismatches between all nodes in c1 U c2 and any other 
        # component which has a mismatch to a node in c1 U c2.
        if is_match:
            inside = c1.union(c2)
            outside = set(n for n in self.nodes if n not in inside)
            for u in inside:
                for edge in self.edges(u):
                    v = edge[1]
                    if v in outside and self.edges[u, v]['type_'] == 'mismatch':
                        v_component = self.get_match_component(v)
                        for w in v_component:
                            if w in outside:  # Check to avoid duplicates
                                updates.append((u, w, False))
                                self.add_edge(u, w, type_='mismatch')
                                outside.remove(w)

        return updates

    def get_mapping(self) -> dict:
        """
        Get a mapping of interest group names to their merged names.

        Returns:
            dict: A dictionary mapping interest group names to their merged names.
        """
        # Add edges between all interest groups that have same normalized name
        match_edges = [(u, v) for u, v in self.edges if self.edges[u, v]['type_'] == 'match']
        match_subgraph = self.edge_subgraph(match_edges)
        components = list(nx.connected_components(match_subgraph))
        mapping = {}
        for component in components:
            merged_name = self.merge_interest_groups(component)
            for name in component:
                mapping[name] = merged_name
        return mapping

    def merge_interest_groups(self, component: set) -> str:
        """
        Merge interest groups in a connected component. Select the most
        common name as the merged name, and update the aliases.

        Args:
            component (set): A set of interest group names in the component.

        Returns:
            str: The merged name of the interest groups.
        """
        names = []
        docs = {}
        for u in component:
            # None of the merged nodes should appear in the same document
            if any(doc in docs for doc in self.nodes[u]['docs']):
                raise ValueError(f"Interest groups {u} are associated with the same document.")

            names.extend(self.nodes[u]['docs'].values())
            docs.update(self.nodes[u]['docs'])

        # Find the name with the smallest average edit distance to all other names
        merged_name = min(names, key=lambda x: sum(editdistance.eval(x, y) for y in names) / len(names))

        return merged_name
