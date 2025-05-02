import re
import networkx as nx 
from typing import List, Optional, Dict, Any, Tuple, Set
import sqlite3
import os
import pandas as pd
from tqdm import tqdm
import logging
from collections import defaultdict
from pathlib import Path
import sys
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import numpy as np
from rapidfuzz import fuzz
from Levenshtein import distance as lev_dist
from Levenshtein import jaro_winkler
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.ensemble import RandomForestClassifier
from graph_tool.all import Graph, minimize_nested_blockmodel_dl, mcmc_equilibrate, BlockState
import graph_tool.inference as gt_inf

sys.path.append(str(Path(__file__).parent.parent.parent))
from amici.utils.normalizers import normalize_interest_group_name, shorten_common_terms
from amici.deduplication.dedupe_graph import DedupeGraph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DbDeduplicator():
    
    def __init__(self, db_path: str, 
                 # Blocking parameters
                 char_ngram_range=(3, 3),
                 char_max_features=20000,
                 word_max_features=20000,
                 char_similarity_threshold=0.6,
                 word_similarity_threshold=0.6,
                 top_n_matches=10,
                 # Prediction parameters
                 sentence_crossencoder_model='cross-encoder/stsb-roberta-base',
                 hbsbm_samples=1000,
                 hbsbm_wait=1000,
                 hbsbm_niter=10,
                 # Persistence parameters
                 output_dir=None):
        """
        Initialize the DbDeduplicator with database path and parameters.

        Args:
            db_path (str): Path to the database file
            
            # Blocking parameters
            char_ngram_range (tuple): n-gram range for character vectorizer
            char_max_features (int): Maximum features for character vectorizer
            word_max_features (int): Maximum features for word vectorizer
            char_similarity_threshold (float): Minimum similarity for character TF-IDF matches
            word_similarity_threshold (float): Minimum similarity for word TF-IDF matches
            top_n_matches (int): Number of top matches to consider per entity
            
            # Prediction parameters
            sentence_crossencoder_model (str): Name of the sentence cross encoder model
            hbsbm_samples (int): Number of samples for HBSBM
            hbsbm_wait (int): Wait iterations for HBSBM equilibration
            hbsbm_niter (int): Number of iterations for HBSBM
            
            # Persistence parameters
            output_dir (str, optional): Directory to store persistent data. If None, 
                                        defaults to a directory next to the database.
        """
        # Database parameters
        self.db_path = db_path
        
        # Blocking parameters
        self.char_ngram_range = char_ngram_range
        self.char_max_features = char_max_features
        self.word_max_features = word_max_features
        self.char_similarity_threshold = char_similarity_threshold
        self.word_similarity_threshold = word_similarity_threshold
        self.top_n_matches = top_n_matches
        
        # Prediction parameters
        self.sentence_crossencoder_model = sentence_crossencoder_model
        self.hbsbm_samples = hbsbm_samples
        self.hbsbm_wait = hbsbm_wait
        self.hbsbm_niter = hbsbm_niter
        
        # Setup output directory for persistent storage
        self.output_dir = output_dir or os.path.join(os.path.dirname(db_path), "dedupe_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Paths for persistent storage
        self.features_path = os.path.join(self.output_dir, "features.csv")
        self.graph_path = os.path.join(self.output_dir, "dedupe_graph.pkl")
        self.hitl_state_path = os.path.join(self.output_dir, "hitl_state.pkl")
        
        # Initialize objects
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.dedupe_graph: DedupeGraph = DedupeGraph()
        self.blocks = defaultdict(list)
        self.matches = defaultdict(list)
        self.features_df = None
        self.char_vectorizer = None
        self.word_vectorizer = None
        self.embeddings_cache = {}
        
        # Load data
        self.load_interest_groups()
        self.conn.close()

    def load_interest_groups(self):
        # Implementation remains the same
        self.cursor.execute("SELECT * FROM amici;")
        rows = self.cursor.fetchall()
        docs_to_amici = defaultdict(list)
        amicus_to_raw = {}
        for row in rows:
            id_ = row[0]
            doc = row[1]
            name = row[2]
            type_ = row[3]

            if type_ == 'organization':
                amicus = self.dedupe_graph.add_interest_group(name, doc)
                docs_to_amici[doc].append(amicus)
                amicus_to_raw[amicus] = name

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
                try:
                    self.dedupe_graph.add_position(amicus, docket, position)
                except ValueError as e:
                    logger.warning(f"Failed to add position for {amicus} ({amicus_to_raw[amicus]}) and {docket}: {e}")
                    continue

        return True
    
    def blocking(self):
        """
        Perform blocking on the interest groups using instance parameters.
        
        Returns:
            dict: Dictionary mapping interest groups to their potential matches.
        """
        interest_groups = [node for node, data in self.dedupe_graph.nodes(data=True) 
                          if data.get('type_') == 'amicus']
        
        if len(interest_groups) == 0:
            logger.warning("No interest groups found in the graph.")
            return self.blocks
        
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
        def match_names(names, vectorizer, lowerbound, n_matches):
            """
            Takes a list of names and returns pairs that are similar based on TF-IDF similarity.
            """
            names_array = np.array(names)
            
            # Transform the names using the vectorizer
            tfidf_matrix = vectorizer.fit_transform(names_array)
            
            # Use sparse matrix multiplication to compute similarities efficiently
            from sparse_dot_topn import sp_matmul_topn
            
            # Get the top n_matches for each name with similarity >= lowerbound
            matches = sp_matmul_topn(tfidf_matrix, tfidf_matrix.transpose(), 
                                     top_n=n_matches, threshold=lowerbound, sort=True)
            
            # Convert to DataFrame
            matches_df = get_matches_df(matches, names_array, names_array)
            
            # Remove self-matches
            matches_df = matches_df[matches_df.left_side != matches_df.right_side]
            
            return matches_df
        
        # Create TF-IDF vectorizers for both blocking and prediction
        logger.info("Creating TF-IDF vectorizers with configured parameters")
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=self.char_ngram_range,
            strip_accents='unicode',
            lowercase=False,
            max_features=self.char_max_features,
            preprocessor=normalize_interest_group_name
        )
        
        self.word_vectorizer = TfidfVectorizer(
            analyzer='word',
            strip_accents='unicode',
            lowercase=False,
            max_features=self.word_max_features,
            preprocessor=normalize_interest_group_name
        )
        
        # Get matches from both methods
        logger.info("Computing character-level TF-IDF matches")
        char_matches = match_names(
            interest_groups, 
            self.char_vectorizer, 
            self.char_similarity_threshold,
            self.top_n_matches
        )
        char_matches.rename(columns={'similarity': 'char_similarity'}, inplace=True)
        
        logger.info("Computing word-level TF-IDF matches")
        word_matches = match_names(
            interest_groups, 
            self.word_vectorizer, 
            self.word_similarity_threshold,
            self.top_n_matches
        )
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

    def compute_similarity_scores(self):
        """
        Predict matches between interest groups using the blocked sets and instance parameters.
        
        Returns:
            pd.DataFrame: DataFrame containing pairs and their similarity features
        """
        logger.info("Initializing prediction phase with featurization")
        
        # Initialize SentenceTransformer model using the configured model name
        try:
            self.cross_encoder = CrossEncoder(self.sentence_crossencoder_model)
            logger.info(f"Successfully loaded SentenceTransformer model: {self.sentence_crossencoder_model}")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer model: {e}")
            logger.warning("Proceeding without sentence encoding features")

        # Generate pairs from blocks
        logger.info("Generating pairs from blocking results")
        pairs = set()
        for left, right_candidates in self.blocks.items():
            for right in right_candidates:
                # Skip if already processed
                left, right = sorted([left, right])
                if ((left, right) in pairs) or left==right:
                    continue

                # Check that it's a valid link
                if self.dedupe_graph.check_match_legality(left, right):
                    # Add to pairs
                    pairs.add((left, right))

        logger.info(f"Generated {len(pairs)} pairs for featurization")

        all_names = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))
        
        # Use the existing vectorizers if they were created in blocking
        if not hasattr(self, 'char_vectorizer') or not hasattr(self, 'word_vectorizer'):
            # Get all interest group names for fitting vectorizers
            interest_groups = [node for node, data in self.dedupe_graph.nodes(data=True) 
                            if data.get('type_') == 'amicus']
            
            if not hasattr(self, 'char_vectorizer'):
                logger.info(f"Creating character-level TF-IDF vectorizer with {len(all_names)} names")
                self.char_vectorizer = TfidfVectorizer(
                    analyzer='char',
                    ngram_range=self.char_ngram_range,
                    min_df=2,
                    max_df=0.9,
                    max_features=self.char_max_features
                )
                self.char_vectorizer.fit(all_names)
            
            if not hasattr(self, 'word_vectorizer'):
                logger.info(f"Creating word-level TF-IDF vectorizer with {len(all_names)} names")
                self.word_vectorizer = TfidfVectorizer(
                    analyzer='word',
                    min_df=2,
                    max_df=0.9,
                    max_features=self.word_max_features
                )
                self.word_vectorizer.fit(all_names)
        
        # Define similarity functions
        def first_letter_jaccard(a, b):
            words_a = a.split(" ")
            words_b = b.split(" ")

            fl_a = {w[0] for w in words_a if len(w) > 0}
            fl_b = {w[0] for w in words_b if len(w) > 0}
            
            if not fl_a or not fl_b:
                return 0
            
            return len(fl_a & fl_b) / len(fl_a | fl_b)

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
        
        # Featurize pairs
        def featurize_pairs(pairs_data):
            logger.info("Featurizing pairs")
            results = []
            
            for i, (left, right) in tqdm(enumerate(pairs_data), desc="Featurizing", total=len(pairs_data)):
                    
                # Extract features
                features = {
                    'left_norm': left,
                    'right_norm': right,
                }
                
                # Calculate similarity features
                for method_name, method_func in methods.items():
                    try:
                        features[method_name] = method_func(left, right)
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

        # # Add cross-encoded scores if available
        # if self.sentence_crossencoder_model:
        #     logger.info("Cross-encoding pairs using SentenceTransformer model")
        #     features_df = self.cross_encode(features_df)

        # Add regular equivalence scores
        features_df = self.deg_corr_reg_equiv(features_df)
        
        # Store the features DataFrame
        self.features_df = features_df
        
        # Save the features after computation
        self.save_features(self.features_path)
        
        return features_df

    def cross_encode(self, features_df):
        """
        Cross-encode the features using a specified model.

        Args:
            features_df (pd.DataFrame): DataFrame containing pairs and their features
            model (str): Name of the cross-encoder model to use
        Returns:
            pd.DataFrame: Updated features DataFrame with cross-encoded scores
        """
        # Ensure the model is loaded
        if not hasattr(self, 'cross_encoder'):
            try:
                self.cross_encoder = CrossEncoder(self.sentence_crossencoder_model)
                logger.info(f"Loaded cross-encoder model: {self.sentence_crossencoder_model}")
            except Exception as e:
                logger.warning(f"Could not load cross-encoder model: {e}")
                return features_df
        
        # Prepare input for cross-encoder
        pairs = list(zip(features_df['left_norm'], features_df['right_norm']))
        
        # Get cross-encoded scores
        scores = self.cross_encoder.predict(pairs)
        
        # Add scores to DataFrame
        features_df['sentence_cross_encoder'] = scores
        
        return features_df

    def deg_corr_reg_equiv(self, features_df, alpha=0.85):
        """
        Compute the degre-corrected regular equivalence for the pairs of nodes in the graph.

        Args:
            features_df (pd.DataFrame): DataFrame containing pairs and their features
        Returns:
            pd.DataFrame: Updated features DataFrame with degree-corrected regular equivalence
        """
        # Create a subgraph containing only the amici in the blocks and their connected dockets
        logger.info("Creating subgraph for DC-reg-eq based on blocked amici")
        
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
        position_edges = [(e[0], e[1]) for e in self.dedupe_graph.edges(data=True) if e[2].get('type_') == 'position']
        subgraph = self.dedupe_graph.edge_subgraph(position_edges).copy()
        subgraph_nodes = list(set(subgraph.nodes()))

        # Weight the edges: R = 1, P = -1
        for u, v, data in subgraph.edges(data=True):
            if data.get('type_') == 'position':
                position = data.get('position', 'unknown')
                if position == 'R':
                    subgraph.edges[u, v]['weight'] = 1
                elif position == 'P':
                    subgraph.edges[u, v]['weight'] = -1
                else:
                    logger.warning(f"Unknown position type: {position}. Defaulting to 0.")
                    subgraph.edges[u, v]['weight'] = 0

        A = nx.to_numpy_array(subgraph, nodelist=subgraph_nodes, weight='weight')
        D = np.diag(abs(A**2).sum(axis=1))
        sigma = np.linalg.inv(D - alpha * A**2)
        sigma = np.nan_to_num(sigma, nan=0.0)
        logger.info("Computed degree-corrected regular equivalence matrix")

        # Create a DataFrame with amicus names
        sigma_df = pd.DataFrame(sigma, index=subgraph_nodes, columns=subgraph_nodes)
        sigma_df = sigma_df.reindex(index=blocked_amici, columns=blocked_amici, fill_value=0)
        logger.info("Created DataFrame for degree-corrected regular equivalence")

        # Create a dictionary with (left, right) as keys and sigma values
        sigma_dict = sigma_df.stack().to_dict()

        # Add DC-reg-eq to features
        features_df['dc_reg_eq'] = features_df.apply(
            lambda row: sigma_dict.get((row['left_norm'], row['right_norm']), 0.0) if row['left_norm'] != row['right_norm'] else 1.0,
            axis=1
        )
        logger.info("Added DC-reg-eq feature to DataFrame")

        return features_df

    def hbsbm_cooccurrence(self, features_df):
        """
        Compute co-occurrence probabilities using HBSBM on the deduplication graph.

        Args:
            features_df (pd.DataFrame): DataFrame containing pairs and their features
        Returns:
            pd.DataFrame: Updated features DataFrame with HBSBM co-occurrence probabilities
        """
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

        flatgraph = nx.Graph()

        # Split the dockets into R and P types, instead of using edge properties
        for edge in subgraph.edges(data=True):
            if edge[2].get('type_') == 'position':
                # Set the position type based on the edge data
                position = edge[2].get('position', 'unknown')
                u, v = edge[0], edge[1]
                if subgraph.nodes[u].get('type_') == 'amicus' and subgraph.nodes[v].get('type_') == 'docket':
                    # Add edge to flatgraph with position type
                    flatgraph.add_edge(u, v + position)
                    flatgraph.nodes[u]['type_'] = 'amicus'
                    flatgraph.nodes[v + position]['type_'] = 'docket'
                elif subgraph.nodes[v].get('type_') == 'amicus' and subgraph.nodes[u].get('type_') == 'docket':
                    # Add edge to flatgraph with position type
                    flatgraph.add_edge(v, u + position)
                    flatgraph.nodes[v]['type_'] = 'amicus'
                    flatgraph.nodes[u + position]['type_'] = 'docket'
                else:
                    continue
        
        logger.info(f"Created subgraph with {len(flatgraph.nodes)} nodes and {len(flatgraph.edges)} edges")
        
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
        for node, data in flatgraph.nodes(data=True):
            v = gt_graph.add_vertex()
            node_map[node] = v
            v_type[v] = data.get('type_', 'unknown')
            v_name[v] = node

        # Make an int v_type_int property for the graph-tool graph
        v_type_int = gt_graph.new_vertex_property("int")
        gt_graph.vp.type_int = v_type_int
        for v in gt_graph.vertices():
            if v_type[v] == 'amicus':
                v_type_int[v] = 0
            elif v_type[v] == 'docket':
                v_type_int[v] = 1
            else:
                raise ValueError(f"Unknown node type: {v_type[v]}")
        
        # Add all position edges
        logger.info("Adding position edges to graph-tool graph")
        for u, v, data in flatgraph.edges(data=True):
            if data.get('type_') == 'position':
                e = gt_graph.add_edge(node_map[u], node_map[v])
        
        # Create a BlockState
        logger.info("Creating BlockState for HBSBM")
        state = gt_inf.minimize_nested_blockmodel_dl(gt_graph, state_args=dict(clabel=v_type_int, pclabel=v_type_int))
        
        # Equilibrate the HBSBM to improve fit
        # logger.info(f"Equilibrating the HBSBM model with wait={self.hbsbm_wait}, niter={self.hbsbm_niter}")
        # mcmc_equilibrate(state, wait=self.hbsbm_wait, mcmc_args=dict(niter=self.hbsbm_niter))
        logger.info("Enhancing the HBSBM fit with multiple sweeps")
        for i in tqdm(range(1000)): # this should be sufficiently large
            state.multiflip_mcmc_sweep(beta=np.inf, niter=10)
        
        # Sample partitions from the posterior
        logger.info(f"Sampling {self.hbsbm_samples} partitions from the posterior distribution")
        
        # Get all amicus nodes
        amicus_nodes = [v for v in gt_graph.vertices() if v_type[v] == 'amicus']
        n_amici = len(amicus_nodes)
        
        # Create a mapping from vertex index to position in the matrix
        amicus_map = {int(v): i for i, v in enumerate(amicus_nodes)}
        
        # Matrix to store co-occurrence probabilities
        cooccurrence = np.zeros((n_amici, n_amici))
        
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
        mcmc_equilibrate(state, force_niter=self.hbsbm_samples, mcmc_args=dict(niter=1),
                         callback=collect_partition)
        
        # Normalize the co-occurrence matrix
        cooccurrence /= self.hbsbm_samples
        
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

    def save_features(self, output_path, format_='csv'):
        """
        Save the features along with metadata about the parameters used.
        
        Args:
            output_path: Path where to save the features
            format_: Storage format_ ('zarr' or 'csv')
        """
        if self.features_df is None:
            raise ValueError("No features to save. Run predict() first.")
        
        # Create metadata dictionary with all parameters
        metadata = {
            'db_path': self.db_path,
            'blocking_params': {
                'char_ngram_range': self.char_ngram_range,
                'char_max_features': self.char_max_features,
                'word_max_features': self.word_max_features,
                'char_similarity_threshold': self.char_similarity_threshold,
                'word_similarity_threshold': self.word_similarity_threshold,
                'top_n_matches': self.top_n_matches
            },
            'prediction_params': {
                'sentence_crossencoder_model': self.sentence_crossencoder_model,
                'hbsbm_samples': self.hbsbm_samples,
                'hbsbm_wait': self.hbsbm_wait,
                'hbsbm_niter': self.hbsbm_niter
            },
            'features_shape': self.features_df.shape,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Remove ending from output_path
        output_path = output_path.rstrip('.zarr').rstrip('.csv')
            
        if format_.lower() == 'csv':
            # Save features as CSV
            self.features_df.to_csv(f"{output_path}.csv", index=False)
            
            # Save metadata separately as JSON
            import json
            with open(f"{output_path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Saved features to {output_path}.csv and metadata to {output_path}_metadata.json")
            
        else:
            raise ValueError(f"Unsupported format: {format_}. Only 'csv' is supported right now.")
    
    @classmethod
    def load_features(cls, input_path, format_='csv'):
        """
        Load features and metadata from disk and instantiate a DbDeduplicator.
        
        Args:
            input_path: Path to the stored features
            format_: Storage format_ ('zarr' or 'csv')
            
        Returns:
            (DbDeduplicator, DataFrame): The instantiated deduplicator and features
        """
        import json

        # remove ending from input_path
        input_path = input_path.rstrip('.zarr').rstrip('.csv')
        
        if format_.lower() == 'zarr':
            import zarr
            
            # Open Zarr store directly
            root = zarr.open(input_path, mode='r')
            
            # Load metadata
            metadata = json.loads(root.attrs['metadata'])
            
            # Create deduplicator with loaded parameters
            deduplicator = cls(
                db_path=metadata['db_path'],
                char_ngram_range=tuple(metadata['blocking_params']['char_ngram_range']),
                char_max_features=metadata['blocking_params']['char_max_features'],
                word_max_features=metadata['blocking_params']['word_max_features'],
                char_similarity_threshold=metadata['blocking_params']['char_similarity_threshold'],
                word_similarity_threshold=metadata['blocking_params']['word_similarity_threshold'],
                top_n_matches=metadata['blocking_params']['top_n_matches'],
                sentence_transformer_model=metadata['prediction_params']['sentence_transformer_model'],
                hbsbm_samples=metadata['prediction_params']['hbsbm_samples'],
                hbsbm_wait=metadata['prediction_params']['hbsbm_wait'],
                hbsbm_niter=metadata['prediction_params']['hbsbm_niter'],
                output_dir=os.path.dirname(input_path)
            )
            
            # Load features
            features_group = root['features']
            features_dict = {}
            for col in features_group:
                features_dict[col] = features_group[col][:]
            
            features_df = pd.DataFrame(features_dict)
            sorted_left_right = features_df[['left_norm', 'right_norm']].apply(sorted, axis=1).copy()
            features_df = features_df[features_df.left_norm != features_df.right_norm]
            features_df = features_df[~sorted_left_right.duplicated(keep='first')]
            features_df = features_df.reset_index(drop=True)

            deduplicator.features_df = features_df
            
            logger.info(f"Loaded features and metadata from Zarr store at {input_path}")
            
        elif format_.lower() == 'csv':
            # Load metadata from JSON
            with open(f"{input_path}_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Create deduplicator with loaded parameters
            deduplicator = cls(
                db_path=metadata['db_path'],
                char_ngram_range=tuple(metadata['blocking_params']['char_ngram_range']),
                char_max_features=metadata['blocking_params']['char_max_features'],
                word_max_features=metadata['blocking_params']['word_max_features'],
                char_similarity_threshold=metadata['blocking_params']['char_similarity_threshold'],
                word_similarity_threshold=metadata['blocking_params']['word_similarity_threshold'],
                top_n_matches=metadata['blocking_params']['top_n_matches'],
                sentence_transformer_model=metadata['prediction_params']['sentence_transformer_model'],
                hbsbm_samples=metadata['prediction_params']['hbsbm_samples'],
                hbsbm_wait=metadata['prediction_params']['hbsbm_wait'],
                hbsbm_niter=metadata['prediction_params']['hbsbm_niter'],
                output_dir=os.path.dirname(input_path)
            )
            
            # Load features
            features_df = pd.read_csv(f"{input_path}.csv")
            sorted_left_right = features_df[['left_norm', 'right_norm']].apply(sorted, axis=1).copy()
            features_df = features_df[features_df.left_norm != features_df.right_norm]
            features_df = features_df[~sorted_left_right.duplicated(keep='first')]
            features_df = features_df.reset_index(drop=True)
            deduplicator.features_df = features_df
            
            logger.info(f"Loaded features from {input_path}.csv and metadata from {input_path}_metadata.json")
        
        else:
            raise ValueError(f"Unsupported format: {format_}. Use 'zarr' or 'csv'.")
        
        return deduplicator, features_df
        
    def save_state(self):
        """
        Save the complete state of the deduplicator.
        
        This method saves:
        1. The features DataFrame (if available)
        2. The deduplication graph
        
        Returns:
            dict: Dictionary with paths to saved state components
        """
        # Save features if available
        if self.features_df is not None:
            self.save_features(self.features_path)
            
        # Save graph if available
        if hasattr(self.dedupe_graph, 'save_to_file'):
            self.dedupe_graph.save_to_file(self.graph_path)
        else:
            # If DedupeGraph doesn't have save_to_file, add implementation here
            import pickle
            import os
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.graph_path)), exist_ok=True)
            
            # Use atomic write pattern
            temp_path = f"{self.graph_path}.tmp"
            with open(temp_path, 'wb') as f:
                pickle.dump(self.dedupe_graph, f)
                
            if os.path.exists(self.graph_path):
                os.replace(temp_path, self.graph_path)
            else:
                os.rename(temp_path, self.graph_path)
        
        logger.info(f"Saved complete deduplicator state to {self.output_dir}")
        
        return {
            'features_path': self.features_path,
            'graph_path': self.graph_path
        }
    
    def load_state(self):
        """
        Load the complete state of the deduplicator from disk.
        
        This loads:
        1. The features DataFrame (if available)
        2. The deduplication graph (if available)
        
        Returns:
            bool: True if any state was successfully loaded, False otherwise
        """
        state_loaded = False
        
        # Load graph if available
        if os.path.exists(self.graph_path):
            try:
                if hasattr(DedupeGraph, 'load_from_file'):
                    self.dedupe_graph = DedupeGraph.load_from_file(self.graph_path)
                else:
                    # If DedupeGraph doesn't have load_from_file, implement here
                    import pickle
                    with open(self.graph_path, 'rb') as f:
                        self.dedupe_graph = pickle.load(f)
                
                logger.info(f"Loaded graph from {self.graph_path}")
                state_loaded = True
            except Exception as e:
                logger.error(f"Failed to load graph: {e}")
                
                # Try to load a backup if it exists
                backup_path = f"{self.graph_path}.bak"
                if os.path.exists(backup_path):
                    try:
                        with open(backup_path, 'rb') as f:
                            self.dedupe_graph = pickle.load(f)
                        logger.info(f"Loaded graph from backup {backup_path}")
                        state_loaded = True
                    except Exception as e2:
                        logger.error(f"Failed to load graph backup: {e2}")
                
        # Load features if available
        if os.path.exists(self.features_path):
            try:
                _, self.features_df = self.load_features(self.features_path)
                logger.info(f"Loaded features from {self.features_path}")
                state_loaded = True
            except Exception as e:
                logger.error(f"Failed to load features: {e}")
                
                # Try CSV backup if zarr failed
                csv_backup = f"{os.path.splitext(self.features_path)[0]}.csv"
                if os.path.exists(csv_backup):
                    try:
                        _, self.features_df = self.load_features(csv_backup, format='csv')
                        logger.info(f"Loaded features from backup {csv_backup}")
                        state_loaded = True
                    except Exception as e2:
                        logger.error(f"Failed to load features backup: {e2}")
                
        return state_loaded

    def run_hitl_process(self, batch_size=10, host='127.0.0.1', port=5000, resume=True, autosave_interval=5):
        """
        Run a human-in-the-loop deduplication process with persistence.
        
        Args:
            batch_size (int): Number of candidate pairs to show in each batch
            host (str): Host address for the web interface
            port (int): Port number for the web interface
            resume (bool): Whether to try resuming from saved state
            autosave_interval (int): How often to save state (in number of decisions)
            
        Returns:
            HITLManager: The HITL manager instance
        """
        # Load existing state if requested
        if resume:
            state_loaded = self.load_state()
            logger.info(f"State load {'succeeded' if state_loaded else 'failed'}")
            
            if os.path.exists(self.hitl_state_path):
                logger.info(f"Resuming HITL process from {self.hitl_state_path}")
                hitl_manager = HITLManager.load_state(self.hitl_state_path, self)
                resume_successful = True
            else:
                resume_successful = False
        else:
            resume_successful = False
        
        # If no saved state or resume failed, start fresh
        if not resume or not resume_successful:
            # If no features, compute them
            if self.features_df is None:
                logger.info("Computing features for candidate pairs")
                self.blocking()
                self.compute_similarity_scores()
                self.save_state()  # Save initial state
                
            logger.info("Initializing new HITL process")
            hitl_manager = HITLManager(
                self, 
                batch_size=batch_size, 
                save_path=self.hitl_state_path,
                autosave_interval=autosave_interval
            )
            hitl_manager.initialize_candidates()
            hitl_manager.save_state()  # Initial save
        
        # Setup automatic save on exit
        import atexit
        def save_on_exit():
            logger.info("Process ending, saving final state")
            hitl_manager.save_state()
            self.save_state()
        atexit.register(save_on_exit)
        
        # Create and run the GUI
        gui = HITLGui(hitl_manager, host=host, port=port)
        gui.run()
        
        return hitl_manager

    def apply_confirmed_matches(self, output_path=None):
        """
        Apply all confirmed matches and generate a mapping file.
        
        This function generates a mapping from all original entity names to their
        canonical representations based on the deduplication results.
        
        Args:
            output_path (str, optional): Path where to save the mapping CSV.
                                         If None, no file is written.
                
        Returns:
            pd.DataFrame: DataFrame containing the mapping from original to mapped names
        """
        # Get the mapping from the DedupeGraph
        mapping = self.dedupe_graph.get_mapping()
        
        # Convert to DataFrame for easier viewing/saving
        mapping_df = pd.DataFrame(
            [(original, mapped) for original, mapped in mapping.items()],
            columns=['original_name', 'mapped_name']
        )
        
        # Save if requested
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Use atomic write for CSV
            temp_path = f"{output_path}.tmp"
            mapping_df.to_csv(temp_path, index=False)
            
            if os.path.exists(output_path):
                os.replace(temp_path, output_path)
            else:
                os.rename(temp_path, output_path)
                
            logger.info(f"Saved mapping to {output_path}")
        
        return mapping_df

import pandas as pd
import logging
import pickle
import os
import time
import glob
import numpy as np
from typing import List, Dict, Set, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

class ClassifierModel:
    """Encapsulates the ML model for predicting matches"""
    
    def __init__(self, model_type='random_forest', feature_columns=None):
        """
        Initialize a classifier model for predicting entity matches.
        
        Args:
            model_type (str): Type of model to use ('random_forest' or 'xgboost')
            feature_columns (list, optional): List of feature column names to use for prediction
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
        # Default feature columns to use (can be overridden)
        self.feature_columns = feature_columns or [
            'ratio', 'partialratio', 'tokensort', 'tokenset', 
            'levenstein', 'jaro_winkler', 'charsim', 'wordsim',
            'hbsbm_prob', 'first_letter_jaccard'
        ]
    
    def _initialize_model(self):
        """Initialize the ML model based on model_type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, 
                class_weight='balanced',
                random_state=42
            )
        elif self.model_type == 'xgboost':
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                scale_pos_weight=2,  # For imbalanced data
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X, y):
        """
        Train the model with labeled data.
        
        Args:
            X: Features array
            y: Labels array
        
        Returns:
            None
        """
        if self.model is None:
            self._initialize_model()
            
        self.model.fit(X, y)
        self.is_trained = True
        
        # Log feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            importances = list(zip(self.feature_columns, self.model.feature_importances_))
            importances.sort(key=lambda x: x[1], reverse=True)
            logger.info("Feature importances:")
            for feature, importance in importances:
                logger.info(f"  {feature}: {importance:.4f}")
    
    def predict_proba(self, features_df):
        """
        Predict match probabilities for candidate pairs.
        
        Args:
            features_df: DataFrame with feature columns
            
        Returns:
            np.array: Array of predicted match probabilities
        """
        if not self.is_trained:
            # Return a default score if model is not trained
            if 'sentence_cross_encoding' in features_df.columns:
                return features_df['sentence_cross_encoding'].values
            else:
                return features_df['jaro_winkler'].values
            
        # Extract features from the dataframe
        X = features_df[self.feature_columns].values
        
        if self.model is None:
            raise ValueError("Model is not initialized. Call _initialize_model() first.")

        # Predict probabilities
        if hasattr(self.model, 'predict_proba'):
            p = self.model.predict_proba(X)
            try:
                # Check if the model has predict_proba method
                pr = p[:, 1]
            except IndexError:
                if p.shape[1] == 1:
                    # If the model returns a single column, use that
                    pr = p[:, 0]
                else:
                    # For debugging -- show X
                    p = self.model.predict_proba(X)
                    logger.error(f"X shape: {X.shape}")
                    logger.error(f"X: {X}")
                    logger.error(f"Model: {self.model}")
                    logger.error(f"Model type: {type(self.model)}")
                    logger.error(f"Model predict_proba {p.shape}: {p}")

                    raise IndexError("Model prediction failed. Check the input shape.")
        else:
            # Use predict for regressors
            pr = self.model.predict(X)
        # Ensure p is a 1D array
        if pr.ndim > 1:
            pr = pr.flatten()

        return pr
    
    def save_to_dict(self):
        """
        Save model to a dictionary.
        
        Returns:
            dict: Dictionary containing model state
        """
        import pickle
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_columns': self.feature_columns,
            'model_bytes': pickle.dumps(self.model) if self.model else None
        }
    
    @classmethod
    def load_from_dict(cls, data):
        """
        Load model from a dictionary.
        
        Args:
            data (dict): Dictionary containing model state
            
        Returns:
            ClassifierModel: Loaded model instance
        """
        import pickle
        model = cls(
            model_type=data['model_type'],
            feature_columns=data['feature_columns']
        )
        model.is_trained = data['is_trained']
        if data['model_bytes']:
            model.model = pickle.loads(data['model_bytes'])
        return model


class HITLManager:
    """
    Human-in-the-Loop (HITL) Manager for entity deduplication.
    
    This class manages the interactive deduplication process where human reviewers
    make decisions about candidate duplicate pairs. It handles candidate selection,
    records human decisions, updates a deduplication graph, and trains a machine
    learning model to improve candidate selection over time.
    
    Attributes:
        deduplicator (DbDeduplicator): Deduplicator instance with candidate pairs
        model (ClassifierModel): ML model for predicting matches
        batch_size (int): Number of candidates to return in each batch
        labeled_pairs (list): History of human decisions [(left, right, is_match, timestamp)]
        pending_review (list): Current queue of candidate pairs waiting for review
        reviewed_pairs (set): Set of pairs that have already been reviewed
        uncertainty_sampling (bool): Whether to use uncertainty sampling for candidate selection
        save_path (str): Path where state is saved
        autosave_interval (int): Number of decisions after which to autosave
        last_save_count (int): Count of labeled pairs at last save
    """
    def __init__(self, deduplicator, model=None, batch_size=10, 
                 save_path='hitl_state.pkl', autosave_interval=5):
        """
        Initialize the HITL Manager.
        
        Args:
            deduplicator (DbDeduplicator): Deduplicator instance with candidate pairs
            model (ClassifierModel, optional): ML model for match prediction
            batch_size (int, optional): Number of candidates to return in each batch
            save_path (str, optional): Path where to save state
            autosave_interval (int, optional): Number of decisions after which to autosave
        """
        self.deduplicator = deduplicator
        self.model = model or ClassifierModel(feature_columns=deduplicator.features_df.drop(
            columns=['left_norm', 'right_norm', 'match_probability'], errors='ignore').columns.tolist())
        self.batch_size = batch_size
        self.labeled_pairs = []  # (left, right, label, timestamp)
        self.pending_review = []
        self.reviewed_pairs = set()  # Track already reviewed pairs
        self.uncertainty_sampling = False
        self.save_path = save_path
        self.autosave_interval = autosave_interval
        self.last_save_count = 0
        
    def initialize_candidates(self):
        """
        Load candidates from deduplicator if needed and prepare initial candidate queue.
        
        This method ensures blocking and similarity score computation have been performed,
        then initializes the queue of candidates for review.
        
        Returns:
            list: The initial set of candidates for review
        """
        if self.deduplicator.features_df is None:
            self.deduplicator.blocking()
            self.deduplicator.compute_similarity_scores()
        
        # Initialize pending review queue
        return self._refresh_candidates()
    
    def _refresh_candidates(self):
        """
        Update candidate pairs with latest model predictions and prioritize them.
        
        This method:
        1. Uses the current model to predict match probabilities (if model is trained)
        2. Filters out already reviewed pairs
        3. Sorts candidates based on uncertainty or match probability
        
        Returns:
            list: Updated list of candidate pairs for review
        """
        df = self.deduplicator.features_df.copy()
        # Drop duplicate pairs
        sorted_left_right = df[['left_norm', 'right_norm']].apply(sorted, axis=1).copy()
        df = df[~sorted_left_right.duplicated(keep='first')]
        df = df.reset_index(drop=True)

        if self.model.is_trained:
            # Use model to predict match probabilities
            probs = self.model.predict_proba(df)
            df['match_probability'] = probs
        else:
            # Without a trained model, use heuristics
            if 'sentence_cross_encoding' in df.columns:
                df['match_probability'] = df['sentence_cross_encoding']  # Fallback to a heuristic
            else:
                df['match_probability'] = df['jaro_winkler']  # Alternative fallback
            
        # Filter out already reviewed pairs
        mask = ~df.apply(lambda x: (x['left_norm'], x['right_norm']) in self.reviewed_pairs, axis=1)
        candidates = df[mask].copy()
        
        # Calculate uncertainty (closer to 0.5 means more uncertain)
        if self.uncertainty_sampling:
            candidates['uncertainty'] = 0.5 - abs(candidates['match_probability'] - 0.5)
            
            # Combine uncertainty and match probability for ranking
            # This creates a balance between reviewing uncertain pairs and high-confidence matches
            candidates['priority_score'] = (
                candidates['uncertainty'] * 0.7 +  # Emphasize uncertainty
                candidates['match_probability'] * 0.3  # Still consider likely matches
            )
            candidates = candidates.sort_values('priority_score', ascending=False)
        else:
            # Traditional sorting by match probability only
            candidates = candidates.sort_values('match_probability', ascending=False)
        
        self.pending_review = candidates.to_dict('records')
        return self.pending_review
    
    def toggle_uncertainty_sampling(self, enable=None):
        """
        Toggle between uncertainty-based sampling and match probability-based sampling.
        
        When uncertainty sampling is enabled, candidates where the model is more uncertain
        (probabilities closer to 0.5) are prioritized for review.
        
        Args:
            enable (bool, optional): If provided, explicitly set uncertainty sampling mode. 
                                     If None, toggle the current setting.
            
        Returns:
            bool: The current status of uncertainty sampling (True if enabled)
        """
        if enable is not None:
            self.uncertainty_sampling = enable
        else:
            self.uncertainty_sampling = not self.uncertainty_sampling
            
        # Refresh candidates with the new sampling strategy
        self._refresh_candidates()
        return self.uncertainty_sampling
        
    def get_sampling_strategy_stats(self):
        """
        Get statistics about current candidates based on active sampling strategy.
        
        Returns:
            dict: Statistics about the current candidates and sampling strategy:
                 - strategy: Current sampling strategy name
                 - avg_probability: Average match probability for top candidates
                 - avg_uncertainty: Average uncertainty for top candidates (if using uncertainty sampling)
        """
        if not self.pending_review:
            return {}
            
        avg_probability = sum(p.get('match_probability', 0) for p in self.pending_review[:10]) / 10
        
        if self.uncertainty_sampling:
            avg_uncertainty = sum(0.5 - abs(p.get('match_probability', 0) - 0.5) 
                                for p in self.pending_review[:10]) / 10
            return {
                'strategy': 'uncertainty sampling',
                'avg_uncertainty': avg_uncertainty,
                'avg_probability': avg_probability
            }
        else:
            return {
                'strategy': 'probability sampling',
                'avg_probability': avg_probability
            }
    
    def get_next_batch(self):
        """
        Get the next batch of candidate pairs for review.
        
        If the pending queue is empty, it will be refreshed first.
        
        Returns:
            list: A batch of candidate pairs up to the configured batch size
        """
        if not self.pending_review:
            self._refresh_candidates()
            
        batch = self.pending_review[:self.batch_size]
        return batch
    
    def record_decision(self, left_norm, right_norm, is_match):
        """
        Record a human decision and update the deduplication graph.
        
        This method:
        1. Records the human decision
        2. Updates the graph based on the decision
        3. Propagates the decision through the graph (e.g., due to transitivity)
        4. Updates the pending review queue
        5. Autosaves state if needed based on autosave_interval
        
        Args:
            left_norm (str): Normalized name of the first entity
            right_norm (str): Normalized name of the second entity
            is_match (bool): True if the pair represents the same entity, False otherwise
            
        Returns:
            bool: True if the decision was successfully recorded, False if there was a conflict
        """
        retrain = False
        # Record the labeled pair
        self.labeled_pairs.append((left_norm, right_norm, is_match, pd.Timestamp.now()))
        self.reviewed_pairs.add((left_norm, right_norm))
        
        # Update the graph based on the decision
        try:
            updates = self.deduplicator.dedupe_graph.label_match(left_norm, right_norm, is_match)
            for u, v, b in updates:
                self.labeled_pairs.append((u, v, b, pd.Timestamp.now()))
                self.reviewed_pairs.add((u, v))
                if len(self.labeled_pairs) % 10 == 0:
                    retrain = True
        except ValueError as e:
            # Handle conflicts in the graph (e.g., trying to add a mismatch when a match exists)
            logger.warning(f"Conflict while updating graph: {e}")
            return False
            
        # Remove the reviewed pair from pending review
        self.pending_review = [p for p in self.pending_review 
                               if not (p['left_norm'], p['right_norm']) in self.reviewed_pairs]
        
        # Check if we should autosave based on decision count
        if len(self.labeled_pairs) - self.last_save_count >= self.autosave_interval:
            self.save_state(self.save_path)
            self.last_save_count = len(self.labeled_pairs)
            logger.info(f"Autosaved after {self.autosave_interval} decisions to {self.save_path}")
        
        if retrain:
            # Retrain the model if needed
            retrain_success = self.retrain_model()
            if retrain_success:
                logger.info("Model retrained successfully")
            else:
                logger.warning("Model retraining skipped due to insufficient data")
                return False

        return True
    
    def retrain_model(self, force=False):
        """
        Retrain the model with the current labeled data.
        
        Args:
            force (bool, optional): If True, retrain even with few samples. Defaults to False.
            
        Returns:
            bool: True if model was retrained, False otherwise
        """
        # Check if we have enough new labeled data to justify retraining
        if len(self.labeled_pairs) < 10 and not force:
            logger.info("Not enough new labeled pairs to justify retraining")
            return False
            
        # Prepare training data
        X_train = []
        y_train = []
        
        for left, right, label, _ in self.labeled_pairs:
            # Find the corresponding features
            mask = ((self.deduplicator.features_df['left_norm'] == left) & 
                    (self.deduplicator.features_df['right_norm'] == right))
            if mask.any():
                row = self.deduplicator.features_df[mask].iloc[0]
                feature_values = [row[col] for col in self.model.feature_columns if col in row]
                if len(feature_values) == len(self.model.feature_columns):
                    X_train.append(feature_values)
                    y_train.append(1 if label else 0)
                else:
                    logger.warning(f"Missing features for pair ({left}, {right}): {row}")
        
        if not X_train:
            logger.warning("No matching feature rows found for labeled pairs")
            return False
            
        # Train the model
        self.model.train(X_train, y_train)
        
        # Refresh candidates with new model predictions
        self._refresh_candidates()
        
        return True
    
    def save_state(self, path=None):
        """
        Save the current state of the HITL process.
        
        Saves labeled pairs, reviewed pairs, and the trained model to a pickle file.
        
        Args:
            path (str, optional): Path to save the state file. If None, uses self.save_path.
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        path = path or self.save_path
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Also save a timestamped backup periodically
        if len(self.labeled_pairs) % 50 == 0:
            backup_path = f"{os.path.splitext(path)[0]}_{int(time.time())}.pkl"
            self._save_to_path(backup_path)
            logger.info(f"Created backup at {backup_path}")
        
        # Save current state
        success = self._save_to_path(path)
        if success:
            logger.info(f"Saved current state to {path}")
        
        return success
    
    def _save_to_path(self, path):
        """
        Internal method to save state to a specific path.
        
        Args:
            path (str): Path to save the state file
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            state = {
                'labeled_pairs': self.labeled_pairs,
                'reviewed_pairs': list(self.reviewed_pairs),
                'model': self.model.save_to_dict() if self.model else None,
                'last_save_count': len(self.labeled_pairs),
                'pending_review': self.pending_review[:100],  # Save some pending items
                'uncertainty_sampling': self.uncertainty_sampling,
                'batch_size': self.batch_size,
                'autosave_interval': self.autosave_interval,
                'timestamp': time.time(),
                'metadata': {
                    'total_candidates': len(self.deduplicator.features_df) if self.deduplicator.features_df is not None else 0,
                    'decisions_made': len(self.labeled_pairs)
                }
            }
            
            # Use atomic write pattern to prevent corruption
            temp_path = f"{path}.tmp"
            with open(temp_path, 'wb') as f:
                pickle.dump(state, f)
            
            if os.path.exists(path):
                os.replace(temp_path, path)  # Atomic replacement
            else:
                os.rename(temp_path, path)
                
            return True
        except Exception as e:
            logger.error(f"Error saving state to {path}: {e}")
            return False

    @classmethod
    def load_state(cls, path, deduplicator):
        """
        Load a saved HITL state.
        
        Args:
            path (str): Path to the saved state file
            deduplicator (DbDeduplicator): Deduplicator instance to use with the loaded state
            
        Returns:
            HITLManager: A new HITLManager instance with the loaded state
        """
        if not os.path.exists(path):
            logger.warning(f"State file {path} not found, creating new manager")
            return cls(deduplicator, save_path=path)
            
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
                
            manager = cls(
                deduplicator, 
                save_path=path,
                batch_size=state.get('batch_size', 10),
                autosave_interval=state.get('autosave_interval', 5)
            )
            manager.labeled_pairs = state['labeled_pairs']
            manager.reviewed_pairs = set(state['reviewed_pairs'])
            manager.last_save_count = state.get('last_save_count', len(state['labeled_pairs']))
            manager.pending_review = state.get('pending_review', [])
            manager.uncertainty_sampling = state.get('uncertainty_sampling', False)
            
            if state.get('model'):
                manager.model = ClassifierModel.load_from_dict(state['model'])
                
            logger.info(f"Successfully loaded state from {path} with {len(manager.labeled_pairs)} labeled pairs")
            return manager
        except Exception as e:
            logger.error(f"Failed to load state from {path}: {e}")
            # Try to load backup if it exists
            backups = glob.glob(f"{os.path.splitext(path)[0]}_*.pkl")
            if backups:
                latest_backup = max(backups, key=os.path.getmtime)
                logger.info(f"Attempting to load from backup: {latest_backup}")
                return cls.load_state(latest_backup, deduplicator)
            else:
                logger.warning("No backups found, creating new manager")
                return cls(deduplicator, save_path=path)

    def get_probability_distribution(self):
        """
        Get the distribution of match probabilities across all candidate pairs.
        
        Returns:
            list: Histogram of probability distribution with 10 bins
        """
        if self.deduplicator.features_df is None:
            return [0] * 10  # Empty distribution
        
        # If model is trained, use its predictions
        if self.model.is_trained:
            probs = self.model.predict_proba(self.deduplicator.features_df)
        else:
            # Otherwise use a similarity score as proxy
            if 'sentence_cross_encoding' in self.deduplicator.features_df.columns:
                probs = self.deduplicator.features_df['sentence_cross_encoding'].values
            else:
                probs = self.deduplicator.features_df['jaro_winkler'].values
        
        # Create histogram with 10 bins (0-0.1, 0.1-0.2, ..., 0.9-1.0)
        hist, _ = np.histogram(probs, bins=10, range=(0, 1))
        
        return hist.tolist()
    
    def validate_state(self):
        """
        Validate the manager's state for integrity.
        
        Checks for inconsistencies in the HITL state and attempts to repair them.
        
        Returns:
            bool: True if state is valid or repaired successfully, False otherwise
        """
        issues = []
        
        # Check for invalid labeled pairs
        invalid_pairs = []
        for i, (left, right, is_match, timestamp) in enumerate(self.labeled_pairs):
            if left not in self.deduplicator.dedupe_graph.nodes:
                invalid_pairs.append((i, f"Left entity '{left}' not found in graph"))
            if right not in self.deduplicator.dedupe_graph.nodes:
                invalid_pairs.append((i, f"Right entity '{right}' not found in graph"))
                
        if invalid_pairs:
            issues.append(f"Found {len(invalid_pairs)} invalid labeled pairs")
            # Repair by removing invalid pairs
            valid_pairs = [p for i, p in enumerate(self.labeled_pairs) 
                           if i not in [idx for idx, _ in invalid_pairs]]
            self.labeled_pairs = valid_pairs
        
        # Check for graph consistency
        graph_issues = 0
        for u, v, data in self.deduplicator.dedupe_graph.edges(data=True):
            if data.get('type_') == 'match':
                # Check for any mismatches between the same entities
                try:
                    for u2, v2, data2 in self.deduplicator.dedupe_graph.edges(data=True):
                        if ((u == u2 and v == v2) or (u == v2 and v == u2)) and data2.get('type_') == 'mismatch':
                            issues.append(f"Inconsistency: Edge between {u} and {v} is both match and mismatch")
                            graph_issues += 1
                except Exception as e:
                    issues.append(f"Error checking graph consistency: {e}")
                    
        if issues:
            logger.warning(f"State validation found {len(issues)} issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
                
            # If we made repairs, save the fixed state
            if invalid_pairs:
                self.save_state()
                logger.info("Saved repaired state after validation")
        
        return len(issues) == 0

import os
import time
import tempfile
import logging
from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path

logger = logging.getLogger(__name__)

class HITLGui:
    """
    Web-based GUI for human-in-the-loop deduplication.
    
    This class provides a web interface for reviewing and making decisions
    on candidate duplicate pairs, with built-in persistence controls.
    """
    
    def __init__(self, hitl_manager, host='127.0.0.1', port=5000):
        """
        Initialize the HITL GUI.
        
        Args:
            hitl_manager (HITLManager): The HITL manager instance
            host (str): Host address for the web interface
            port (int): Port number for the web interface
        """
        self.hitl_manager = hitl_manager
        self.host = host
        self.port = port
        self.app = None
    
    def _create_app(self):
        """
        Create the Flask application with all required routes.
        
        Returns:
            Flask: The configured Flask application
        """
        from flask import Flask, render_template, request, jsonify, send_file
        app = Flask(__name__, 
                   template_folder=str(Path(__file__).parent / 'templates'),
                   static_folder=str(Path(__file__).parent / 'static'))
        
        @app.route('/')
        def index():
            """Render the main HITL interface."""
            return render_template('hitl_index.html')

        @app.route('/api/toggle_sampling', methods=['POST'])
        def toggle_sampling():
            """Toggle between uncertainty and probability-based sampling."""
            enable = request.json.get('enable')
            active = self.hitl_manager.toggle_uncertainty_sampling(enable)
            stats = self.hitl_manager.get_sampling_strategy_stats()
            return jsonify({
                'uncertainty_sampling': active,
                'stats': stats
            })

        @app.route('/api/sampling_stats', methods=['GET'])
        def sampling_stats():
            """Get statistics about the current sampling strategy."""
            stats = self.hitl_manager.get_sampling_strategy_stats()
            return jsonify({
                'uncertainty_sampling': self.hitl_manager.uncertainty_sampling,
                'stats': stats
            })
        
        @app.route('/api/get_batch', methods=['GET'])
        def get_batch():
            """Get the next batch of candidate pairs for review."""
            batch = self.hitl_manager.get_next_batch()
            formatted_batch = []
            
            for item in batch:
                left_norm = item['left_norm']
                right_norm = item['right_norm']
                
                # Get original names from the graph
                left_names = list(set(self.hitl_manager.deduplicator.dedupe_graph.nodes[left_norm].get('docs', {}).values()))
                right_names = list(set(self.hitl_manager.deduplicator.dedupe_graph.nodes[right_norm].get('docs', {}).values()))
                
                formatted_batch.append({
                    'left_norm': left_norm,
                    'right_norm': right_norm,
                    'left_names': left_names,
                    'right_names': right_names,
                    'match_probability': item.get('match_probability', 0),
                    # Include similarity metrics for transparency
                    'metrics': {
                        'jaro': item.get('jaro_winkler', 0),
                        'token_sort': item.get('tokensort', 0),
                        'levenshtein': item.get('levenstein', 0),
                        'hbsbm': item.get('hbsbm_prob', 0)
                    }
                })
            
            return jsonify(formatted_batch)
        
        @app.route('/api/record_decision', methods=['POST'])
        def record_decision():
            """Record a human decision about a candidate pair."""
            data = request.json
            left_norm = data.get('left_norm')
            right_norm = data.get('right_norm')
            is_match = data.get('is_match')
            
            if not all([left_norm, right_norm, is_match is not None]):
                return jsonify({'success': False, 'error': 'Missing required fields'})
            
            success = self.hitl_manager.record_decision(left_norm, right_norm, is_match)
            
            return jsonify({'success': success})
        
        @app.route('/api/statistics', methods=['GET'])
        def statistics():
            """Get statistics about the deduplication process."""
            total_pairs = len(self.hitl_manager.deduplicator.features_df) if self.hitl_manager.deduplicator.features_df is not None else 0
            labeled_count = len(self.hitl_manager.labeled_pairs)
            match_count = sum(1 for _, _, is_match, _ in self.hitl_manager.labeled_pairs if is_match)
            
            # Get probability distribution for the histogram
            prob_distribution = self.hitl_manager.get_probability_distribution()
            
            return jsonify({
                'total_candidate_pairs': total_pairs,
                'reviewed_pairs': labeled_count,
                'match_count': match_count,
                'non_match_count': labeled_count - match_count,
                'model_trained': self.hitl_manager.model.is_trained,
                'probability_distribution': prob_distribution
            })
        
        @app.route('/api/save_state', methods=['POST'])
        def save_state():
            """Save the current state of the HITL process."""
            path = request.json.get('path', self.hitl_manager.save_path)
            success = self.hitl_manager.save_state(path)
            
            # Also save deduplicator state
            deduplicator_saved = self.hitl_manager.deduplicator.save_state()
            
            return jsonify({
                'success': success and bool(deduplicator_saved), 
                'timestamp': time.time(),
                'decisions_recorded': len(self.hitl_manager.labeled_pairs),
                'saved_paths': {
                    'hitl_state': path,
                    'deduplicator': deduplicator_saved 
                }
            })
        
        @app.route('/api/download_mapping', methods=['GET'])
        def download_mapping():
            """Generate and provide the current mapping for download."""
            # Get current mapping
            mapping = self.hitl_manager.deduplicator.dedupe_graph.get_mapping()
            
            # Convert to DataFrame
            import pandas as pd
            mapping_df = pd.DataFrame(
                [(original, mapped) for original, mapped in mapping.items()],
                columns=['original_name', 'mapped_name']
            )
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                mapping_df.to_csv(tmp.name, index=False)
                tmp_path = tmp.name
                
            return send_file(
                tmp_path,
                as_attachment=True,
                download_name='current_mapping.csv',
                mimetype='text/csv'
            )
            
        @app.route('/api/progress_status', methods=['GET'])
        def progress_status():
            """Provide detailed progress information."""
            total_pairs = len(self.hitl_manager.deduplicator.features_df) if self.hitl_manager.deduplicator.features_df is not None else 0
            labeled_count = len(self.hitl_manager.labeled_pairs)
            
            # Get time-based statistics
            if labeled_count > 0:
                first_decision = self.hitl_manager.labeled_pairs[0][3]
                last_decision = self.hitl_manager.labeled_pairs[-1][3]
                elapsed = (last_decision - first_decision).total_seconds()
                decisions_per_hour = (labeled_count / elapsed) * 3600 if elapsed > 0 else 0
                
                # Estimate remaining time
                remaining_pairs = total_pairs - labeled_count
                estimated_hours = remaining_pairs / decisions_per_hour if decisions_per_hour > 0 else 0
            else:
                decisions_per_hour = 0
                estimated_hours = 0
            
            return jsonify({
                'total_pairs': total_pairs,
                'labeled_count': labeled_count,
                'completion_percentage': (labeled_count / total_pairs * 100) if total_pairs > 0 else 0,
                'decisions_per_hour': decisions_per_hour,
                'estimated_remaining_hours': estimated_hours,
                'last_save': os.path.getmtime(self.hitl_manager.save_path) if os.path.exists(self.hitl_manager.save_path) else None,
                'autosave_interval': self.hitl_manager.autosave_interval,
                'decisions_since_last_save': len(self.hitl_manager.labeled_pairs) - self.hitl_manager.last_save_count
            })
            
        @app.route('/api/validate_state', methods=['POST'])
        def validate_state():
            """Validate the consistency of the HITL state."""
            repair = request.json.get('repair', False)
            is_valid = self.hitl_manager.validate_state()
            
            if not is_valid and repair:
                # Save repaired state
                self.hitl_manager.save_state()
                self.hitl_manager.deduplicator.save_state()
                
            return jsonify({
                'valid': is_valid,
                'repaired': not is_valid and repair
            })
            
        self.app = app
        return app
    
    def run(self):
        """Run the web GUI."""
        if self.app is None:
            self._create_app()
            
        # Ensure we have candidates to review
        self.hitl_manager.initialize_candidates()
        
        # Start the Flask app
        self.app.run(host=self.host, port=self.port, debug=False)


# if __name__ == "__main__":
#     # Get relative path to the database file
#     current_file = Path(__file__)
#     project_root = current_file.parent.parent  # Go up two directories
#     db_path = project_root / "database" / "supreme_court_docs.db"
#     db_path = str(db_path)  # Convert to string for sqlite3.connect
    
#     # Create output directory if it doesn't exist
#     output_dir = project_root / "data"
#     output_dir.mkdir(exist_ok=True)
    
#     # Initialize deduplicator with parameters
#     deduplicator = DbDeduplicator(
#         db_path=db_path,
#         # You can customize parameters here
#         char_ngram_range=(3, 3),
#         char_max_features=20000,
#         word_max_features=20000,
#         char_similarity_threshold=0.6,
#         word_similarity_threshold=0.6,
#         top_n_matches=10,
#         sentence_transformer_model='all-MiniLM-L6-v2',
#         hbsbm_samples=1000,
#         hbsbm_wait=1000,
#         hbsbm_niter=10
#     )
    
#     # Run blocking
#     blocks = deduplicator.blocking()
    
#     # Run prediction
#     features = deduplicator.compute_similarity_scores()
    
#     # Save features with metadata
#     zarr_path = output_dir / "dedupe_features"
#     csv_path = output_dir / "features"
    
#     # Save in both formats for demonstration
#     deduplicator.save_features(str(csv_path), format='csv')
#     deduplicator.save_features(str(zarr_path), format='zarr')

#     # Example of loading back
#     loaded_deduplicator, loaded_features = DbDeduplicator.load_features(str(zarr_path), format='zarr')
#     print(f"Loaded features shape: {loaded_features.shape}")





