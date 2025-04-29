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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

    def add_match(self, name1: str, name2: str, p: float=1.0):
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
        self.add_edge(normalized_name1, normalized_name2, weight=p, type_='match')

    def get_mapping(self) -> dict:
        """
        Get a mapping of interest group names to their merged names.

        Returns:
            dict: A dictionary mapping interest group names to their merged names.
        """
        # Add edges between all interest groups that have same normalized name
        components = list(nx.connected_components(self))
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

    def join_amici(self, node1: str, node2: str):
        """
        Merge two amicus nodes, joining their docs metadata, positions on dockets, and matches to other nodes.
        
        Args:
            node1 (str): The name of the first amicus node to merge.
            node2 (str): The name of the second amicus node to merge.
            
        Returns:
            str: The name of the surviving node (node1).
            
        Raises:
            ValueError: If either node is not in the graph or not an amicus node.
            ValueError: If both nodes have positions on the same docket.
        """
        # Check if nodes exist and are amicus nodes
        if node1 not in self.nodes:
            raise ValueError(f"Node {node1} not found in graph.")
        if node2 not in self.nodes:
            raise ValueError(f"Node {node2} not found in graph.")
        
        if self.nodes[node1].get('type_') != 'amicus':
            raise ValueError(f"Node {node1} is not an amicus node.")
        if self.nodes[node2].get('type_') != 'amicus':
            raise ValueError(f"Node {node2} is not an amicus node.")
            
        # Merge docs metadata
        for doc, name in self.nodes[node2]['docs'].items():
            if doc in self.nodes[node1]['docs']:
                raise ValueError(f"Both nodes appear in document {doc}.")
            self.nodes[node1]['docs'][doc] = name
            
        # Handle edges (positions and matches)
        for neighbor in list(self.neighbors(node2)):
            edge_data = self.get_edge_data(node2, neighbor)
            edge_type = edge_data.get('type_')
            
            # If it's a docket position
            if self.nodes[neighbor].get('type_') == 'docket':
                if self.has_edge(node1, neighbor):
                    # Both nodes have positions on the same docket
                    raise ValueError(f"Both nodes have positions on docket {neighbor}.")
                # Copy the position edge
                self.add_edge(node1, neighbor, **edge_data)
            
            # If it's a match to another amicus
            elif edge_type == 'match':
                if self.has_edge(node1, neighbor):
                    # Keep the edge with the higher weight
                    old_weight = self.get_edge_data(node1, neighbor).get('weight', 0)
                    new_weight = edge_data.get('weight', 0)
                    if new_weight > old_weight:
                        self.edges[node1, neighbor].update(edge_data)
                else:
                    # Add the match edge
                    self.add_edge(node1, neighbor, **edge_data)
        
        # Remove the second node
        self.remove_node(node2)
        
        return node1



class DbDeduplicator():

    def __init__(self, db_path: str):
        """
        Initialize the DbDeduplicator with the path to the database.

        Args:
            db_path (str): The path to the database file.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        self.dedupe_graph = DedupeGraph()
        self.load_interest_groups()

        self.conn.close()
        self.blocks = defaultdict(list)
        self.matches = defaultdict(list)

    def load_interest_groups(self):
        self.cursor.execute("SELECT * FROM amici;")  # Replace with your actual table name
        rows = self.cursor.fetchall()
        docs_to_amici = defaultdict(list)
        for row in rows:
            id_ = row[0]
            doc = row[1]
            name = row[2]
            type_ = row[3]

            if type_ == 'organization':
                amicus = self.dedupe_graph.add_interest_group(name, doc)
                docs_to_amici[doc].append(amicus)

        self.cursor.execute("SELECT * FROM dockets;")
        rows = self.cursor.fetchall()
        for row in rows:
            id_ = row[0]
            doc = row[1]
            year = row[2]
            number = row[3]
            position = row[4]
            docket = f"{year}-{number}"

            self.dedupe_graph.add_docket(docket, doc)

            for amicus in docs_to_amici[doc]:
                self.dedupe_graph.add_position(amicus, docket, position)

        return True

    
    def blocking(self):
        """
        Perform blocking on the interest groups to reduce the number of comparisons.
        Uses TF-IDF vectorization at both character and word levels to find candidate matches.
        """
        
        # Get all normalized interest group names from the graph
        interest_groups = [node for node, data in self.dedupe_graph.nodes(data=True) 
                        if data.get('type_') == 'amicus']
        
        if len(interest_groups) == 0:
            logger.warning("No interest groups found in the graph.")
            return
        
        logger.info(f"Performing blocking on {len(interest_groups)} interest groups")
        
        # Define function to get sparse matrix matches
        def get_matches_df(sparse_matrix, row_vector, col_vector, top=None):
            non_zeros = sparse_matrix.nonzero()
            
            sparserows = non_zeros[0]
            sparsecols = non_zeros[1]
            
            nr_matches = min(top, sparserows.size) if top else sparserows.size
            
            left_side = np.empty([nr_matches], dtype=object)
            right_side = np.empty([nr_matches], dtype=object)
            similarity = np.zeros(nr_matches)
            
            for index in range(0, nr_matches):
                left_side[index] = row_vector[sparserows[index]]
                right_side[index] = col_vector[sparsecols[index]]
                similarity[index] = sparse_matrix.data[index]
            
            return pd.DataFrame({
                'left_side': left_side,
                'right_side': right_side,
                'similarity': similarity
            })
        
        # Define function to find matches using TF-IDF
        def match_names(names, vectorizer, lowerbound=0.6, n_matches=10):
            """
            Takes a list of names and returns pairs that are similar based on TF-IDF similarity.
            """
            names_array = np.array(names)
            
            # Transform the names using the vectorizer
            tfidf_matrix = vectorizer.fit_transform(names_array)
            
            # Use sparse matrix multiplication to compute similarities efficiently
            from sparse_dot_topn import sp_matmul_topn
            
            # Get the top n_matches for each name with similarity >= lowerbound
            matches = sp_matmul_topn(tfidf_matrix, tfidf_matrix.transpose(), top_n=n_matches, threshold=lowerbound, sort=True)
            
            # Convert to DataFrame
            matches_df = get_matches_df(matches, names_array, names_array)
            
            # Remove self-matches
            matches_df = matches_df[matches_df.left_side != matches_df.right_side]
            
            return matches_df
        
        # Create TF-IDF vectorizers for both blocking and prediction
        logger.info("Creating TF-IDF vectorizers")
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 3),
            strip_accents='unicode',
            lowercase=False,
            max_features=20000,
            preprocessor=normalize_interest_group_name
        )
        
        self.word_vectorizer = TfidfVectorizer(
            analyzer='word',
            strip_accents='unicode',
            lowercase=False,
            max_features=20000,
            preprocessor=normalize_interest_group_name
        )
        
        # Get matches from both methods
        logger.info("Computing character-level TF-IDF matches")
        char_matches = match_names(interest_groups, self.char_vectorizer)
        char_matches.rename(columns={'similarity': 'char_similarity'}, inplace=True)
        
        logger.info("Computing word-level TF-IDF matches")
        word_matches = match_names(interest_groups, self.word_vectorizer)
        word_matches.rename(columns={'similarity': 'word_similarity'}, inplace=True)
        
        # Merge the matches
        all_matches = pd.merge(
            char_matches, 
            word_matches, 
            on=['left_side', 'right_side'], 
            how='outer'
        ).fillna(0)
        
        # Store the candidate pairs for further processing
        for _, row in all_matches.iterrows():
            left = row['left_side']
            right = row['right_side']
            
            # Ensure we don't match interest groups that appear in the same document
            left_docs = set(self.dedupe_graph.nodes[left]['docs'].keys())
            right_docs = set(self.dedupe_graph.nodes[right]['docs'].keys())
            
            if not left_docs.intersection(right_docs):
                # Add to blocks for further processing
                self.blocks[left].append(right)
                # Also add the reverse to ensure symmetry
                self.blocks[right].append(left)
        
        logger.info(f"Identified {len(self.blocks)} blocks with potential matches")
        
        return self.blocks

    def predict(self):
        """
        Predict matches between interest groups using the blocked sets.
        """
        logger.info("Initializing prediction phase with featurization")
        
        # Initialize SentenceTransformer model - only load once and store as instance variable
        try:
            self.st_mini = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Successfully loaded SentenceTransformer model")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer model: {e}")
            logger.warning("Proceeding without sentence encoding features")
            self.st_mini = None
        
        # Define similarity functions
        def first_letter_jaccard(a, b):
            words_a = a.split(" ")
            words_b = b.split(" ")

            fl_a = {w[0] for w in words_a if len(w) > 0}
            fl_b = {w[0] for w in words_b if len(w) > 0}
            
            if not fl_a or not fl_b:
                return 0
            
            return len(fl_a & fl_b) / len(fl_a | fl_b)
        
        # Precompute embeddings for all names
        self.embeddings_cache = {}
        
        # Use the existing vectorizers if they were created in blocking
        if not hasattr(self, 'char_vectorizer') or not hasattr(self, 'word_vectorizer'):
            # Get all interest group names for fitting vectorizers
            interest_groups = [node for node, data in self.dedupe_graph.nodes(data=True) 
                            if data.get('type_') == 'amicus']
            all_names = []
            for group in interest_groups:
                all_names.extend(self.dedupe_graph.nodes[group]['docs'].values())
            
            if not hasattr(self, 'char_vectorizer'):
                logger.info(f"Creating character-level TF-IDF vectorizer with {len(all_names)} names")
                self.char_vectorizer = TfidfVectorizer(
                    analyzer='char',
                    ngram_range=(3, 3),
                    min_df=2,
                    max_df=0.9
                )
                self.char_vectorizer.fit(all_names)
            
            if not hasattr(self, 'word_vectorizer'):
                logger.info(f"Creating word-level TF-IDF vectorizer with {len(all_names)} names")
                self.word_vectorizer = TfidfVectorizer(
                    analyzer='word',
                    min_df=2,
                    max_df=0.9
                )
                self.word_vectorizer.fit(all_names)
        
        # Precompute embeddings for all raw names if SentenceTransformer is available
        if self.st_mini is not None:
            logger.info("Precomputing embeddings for all names")
            all_raw_names = set()
            for node in self.blocks.keys():
                all_raw_names.update(self.dedupe_graph.nodes[node]['docs'].values())
                
            for candidates in self.blocks.values():
                for node in candidates:
                    all_raw_names.update(self.dedupe_graph.nodes[node]['docs'].values())
            
            # Compute embeddings in batches to improve efficiency
            batch_size = 32
            all_raw_names_list = list(all_raw_names)
            
            for i in tqdm(range(0, len(all_raw_names_list), batch_size), desc="Computing embeddings"):
                batch = all_raw_names_list[i:i+batch_size]
                batch_embeddings = self.st_mini.encode(batch)
                
                for j, name in enumerate(batch):
                    self.embeddings_cache[name] = batch_embeddings[j]
            
            logger.info(f"Cached embeddings for {len(self.embeddings_cache)} unique names")
            
            # Define sentence similarity using cached embeddings
            def sentence_cross_encoding(a, b):
                # Get embeddings from cache
                e_a = self.embeddings_cache.get(a)
                e_b = self.embeddings_cache.get(b)
                
                if e_a is None or e_b is None:
                    # Handle case where embedding wasn't precomputed (shouldn't happen)
                    logger.warning(f"Missing embedding in cache for: {a if e_a is None else b}")
                    if e_a is None:
                        e_a = self.st_mini.encode(a)
                        self.embeddings_cache[a] = e_a
                    if e_b is None:
                        e_b = self.st_mini.encode(b)
                        self.embeddings_cache[b] = e_b
                
                # Calculate cosine similarity
                return np.dot(e_a, e_b) / (np.linalg.norm(e_a) * np.linalg.norm(e_b))
        
        def tfidf_sim(a, b, vectorizer):
            v_a = vectorizer.transform([a])
            v_b = vectorizer.transform([b])
            return (v_a * v_b.T).toarray()[0][0]
        
        # Define methods for featurization
        methods = {
            'charsim': lambda a, b: tfidf_sim(a, b, self.char_vectorizer),
            'wordsim': lambda a, b: tfidf_sim(a, b, self.word_vectorizer),
            'ratio': lambda a, b: fuzz.ratio(a, b)/100,
            'partialratio': lambda a, b: fuzz.partial_ratio(a, b)/100,
            'tokensort': lambda a, b: fuzz.token_sort_ratio(a, b)/100,
            'tokenset': lambda a, b: fuzz.token_set_ratio(a, b)/100,
            'levenstein': lambda a, b: 1 - (lev_dist(a, b) / max(len(a), len(b)) if max(len(a), len(b)) > 0 else 0),
            'jaro_winkler': lambda a, b: jaro_winkler(a, b),
            'first_letter_jaccard': lambda a, b: first_letter_jaccard(a, b),
            'combined_len': lambda a, b: np.log(len(a) + len(b)),
            'len_ratio': lambda a, b: min(len(a), len(b)) / max(len(a), len(b)) if max(len(a), len(b)) > 0 else 0
        }
        
        if self.st_mini is not None:
            methods['sentence_cross_encoding'] = lambda a, b: sentence_cross_encoding(a, b)
        
        # Generate pairs from blocks
        logger.info("Generating pairs from blocking results")
        pairs = []
        for left, right_candidates in self.blocks.items():
            for right in right_candidates:
                # Get the raw names (not normalized) for better feature extraction
                for doc_left, name_left in self.dedupe_graph.nodes[left]['docs'].items():
                    for doc_right, name_right in self.dedupe_graph.nodes[right]['docs'].items():
                        # Add the pair with both raw and normalized names
                        pairs.append({
                            'left_norm': left,
                            'right_norm': right,
                            'left_raw': name_left,
                            'right_raw': name_right,
                            'left_doc': doc_left,
                            'right_doc': doc_right
                        })
        
        logger.info(f"Generated {len(pairs)} pairs for featurization")
        
        # Featurize pairs
        def featurize_pairs(pairs_data):
            logger.info("Featurizing pairs")
            results = []
            
            for i, pair in enumerate(tqdm(pairs_data, desc="Featurizing")):
                left_raw = pair['left_raw']
                right_raw = pair['right_raw']
                    
                # Extract features
                features = {
                    'left_norm': pair['left_norm'],
                    'right_norm': pair['right_norm'],
                    'left_raw': left_raw,
                    'right_raw': right_raw,
                    'left_doc': pair['left_doc'],
                    'right_doc': pair['right_doc']
                }
                
                # Calculate similarity features
                for method_name, method_func in methods.items():
                    try:
                        features[method_name] = method_func(left_raw, right_raw)
                    except Exception as e:
                        logger.warning(f"Error calculating {method_name} for pair {i}: {e}")
                        features[method_name] = 0
                
                results.append(features)
                
                # Log progress periodically
                if (i + 1) % 10000 == 0:
                    logger.info(f"Processed {i + 1} pairs")
            
            return pd.DataFrame(results)
        
        # Featurize all pairs
        features_df = featurize_pairs(pairs)
        
        # Apply a model (placeholder for your trained model)
        logger.info("Applying model to predict matches")

        # Create a subgraph containing only the amici in the blocks and their connected dockets
        logger.info("Creating subgraph for HBSBM based on blocked amici")
        
        # Extract all amici involved in potential matches
        blocked_amici = set(self.blocks.keys())
        for candidates in self.blocks.values():
            blocked_amici.update(candidates)
        
        logger.info(f"Found {len(blocked_amici)} amici involved in potential matches")
        
        # Get all dockets connected to these amici
        connected_dockets = set()
        for amicus in blocked_amici:
            for neighbor in self.dedupe_graph.neighbors(amicus):
                if self.dedupe_graph.nodes[neighbor].get('type_') == 'docket':
                    connected_dockets.add(neighbor)
        
        logger.info(f"Found {len(connected_dockets)} dockets connected to blocked amici")
        
        # Create subgraph
        subgraph_nodes = blocked_amici.union(connected_dockets)
        subgraph = self.dedupe_graph.subgraph(subgraph_nodes)
        
        logger.info(f"Created subgraph with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges")
        
        # Create a graph-tool graph from the NetworkX subgraph
        logger.info("Creating graph-tool representation for HBSBM")
        
        # Create a new graph-tool graph
        gt_graph = Graph(directed=False)
        
        # Create vertex property for node type (amicus or docket)
        v_type = gt_graph.new_vertex_property("string")
        v_name = gt_graph.new_vertex_property("string")
        gt_graph.vp.type = v_type
        gt_graph.vp.name = v_name
        
        # Create a mapping from subgraph nodes to gt_graph vertices
        node_map = {}
        
        # First add all nodes
        logger.info("Adding nodes to graph-tool graph")
        for node, data in subgraph.nodes(data=True):
            v = gt_graph.add_vertex()
            node_map[node] = v
            v_type[v] = data.get('type_', 'unknown')
            v_name[v] = node
        
        # Create edge property for position (P or R)
        e_position = gt_graph.new_edge_property("string")
        gt_graph.ep.position = e_position
        
        # Add all position edges
        logger.info("Adding position edges to graph-tool graph")
        for u, v, data in subgraph.edges(data=True):
            if data.get('type_') == 'position':
                e = gt_graph.add_edge(node_map[u], node_map[v])
                e_position[e] = data.get('position', 'unknown')
        
        # Create edge property map for categorical position values (P=0, R=1)
        e_pos_cat = gt_graph.new_edge_property("int")
        for e in gt_graph.edges():
            e_pos_cat[e] = 0 if e_position[e] == "P" else 1
        
        # Create a LayeredBlockState using the categorical edge property
        logger.info("Creating LayeredBlockState for HBSBM with categorical edge covariate")
        state = gt_inf.minimize_nested_blockmodel_dl(
            gt_graph,
            state_args=dict(
                base_type=gt_inf.LayeredBlockState,
                state_args=dict(ec=e_pos_cat, layers=True)
            )
        )
        
        # Equilibrate the HBSBM to improve fit
        logger.info("Equilibrating the HBSBM model")
        mcmc_equilibrate(state, wait=1000, mcmc_args=dict(niter=10))
        
        # Sample partitions from the posterior
        logger.info("Sampling partitions from the posterior distribution")
        
        # Get all amicus nodes
        amicus_nodes = [v for v in gt_graph.vertices() if v_type[v] == 'amicus']
        n_amici = len(amicus_nodes)
        
        # Create a mapping from vertex index to position in the matrix
        amicus_map = {int(v): i for i, v in enumerate(amicus_nodes)}
        
        # Matrix to store co-occurrence probabilities
        cooccurrence = np.zeros((n_amici, n_amici))
        
        n_samples = 1000
        collected_partitions = 0
        
        def collect_partition(s):
            nonlocal collected_partitions
            # Get the partition at the lowest level
            b = s.get_bs()[0].a  # Get the partition as an array
            
            # Create a matrix for this partition
            A = np.zeros((n_amici, n_amici))
            
            # Fill in the matrix
            for i, v1 in enumerate(amicus_nodes):
                v1_idx = int(v1)
                for j, v2 in enumerate(amicus_nodes):
                    if i <= j:  # Only need to fill upper triangle
                        v2_idx = int(v2)
                        # If they're in the same block, mark as 1
                        if b[v1_idx] == b[v2_idx]:
                            A[i, j] = 1
                            A[j, i] = 1  # Ensure symmetry
            
            # Add to the co-occurrence matrix
            nonlocal cooccurrence
            cooccurrence += A
            collected_partitions += 1
            
            # Log progress
            if collected_partitions % 100 == 0:
                logger.info(f"Collected {collected_partitions} partitions")
        
        # Sample from the posterior
        mcmc_equilibrate(state, force_niter=n_samples, mcmc_args=dict(niter=1),
                        callback=collect_partition)
        
        # Normalize the co-occurrence matrix
        cooccurrence /= n_samples
        
        # Create a DataFrame with amicus names
        amicus_names = [v_name[v] for v in amicus_nodes]
        cooccurrence_df = pd.DataFrame(cooccurrence, index=amicus_names, columns=amicus_names)
        
        # Add HBSBM co-occurrence probabilities to features
        logger.info("Adding HBSBM co-occurrence probabilities to features")
        
        # Function to look up probability from the cooccurrence_df
        def get_cooccurrence_prob(left, right):
            if left in cooccurrence_df.index and right in cooccurrence_df.columns:
                return cooccurrence_df.loc[left, right]
            return 0.0
        
        # Add the cooccurrence probability as a feature
        features_df['hbsbm_prob'] = features_df.apply(
            lambda row: get_cooccurrence_prob(row['left_norm'], row['right_norm']), 
            axis=1
        )

        return features_df

    


if __name__ == "__main__":
    # Get relative path to the database file
    # Get relative path to the database file
    current_file = Path(__file__)
    project_root = current_file.parent.parent  # Go up two directories
    db_path = project_root / "database" / "supreme_court_docs.db"
    db_path = str(db_path)  # Convert to string for sqlite3.connect

    deduplicator = DbDeduplicator(db_path)

    blocks = deduplicator.blocking()

    features = deduplicator.predict()

    features.to_csv(str(project_root / "data" / "features.csv"))
    

        


    