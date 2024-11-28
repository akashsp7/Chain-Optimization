import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from node2vec import Node2Vec
import pandas as pd
from datetime import datetime
import pickle
import json
from tqdm import tqdm
from collections import defaultdict
import argparse

def create_graph_from_csv(csv_path, chunk_size=100000):
    G = nx.DiGraph()
    chunks = pd.read_csv(csv_path, chunksize=chunk_size)

    scope_counts = defaultdict(int)
    type_counts = defaultdict(int)
    relationship_counts = defaultdict(int)
    
    print("Building graph from CSV...")
    for chunk in tqdm(chunks):
        start_nodes = chunk[['StartNodeID', 'StartNodeLabels', 'Start_found',
                           'Start_id', 'Start_version', 'Start_timestamp']].drop_duplicates()
        for _, row in start_nodes.iterrows():
            G.add_node(row['StartNodeID'],
                      label=eval(row['StartNodeLabels'])[0],
                      found=row['Start_found'],
                      id=row['Start_id'],
                      version=row['Start_version'],
                      timestamp=row['Start_timestamp'])
        
        end_nodes = chunk[['EndNodeID', 'EndNodeLabels', 'End_found',
                          'End_id', 'End_version', 'End_timestamp',
                          'End_type', 'End_value']].drop_duplicates()
        for _, row in end_nodes.iterrows():
            if pd.notna(row['End_type']):
                type_counts[row['End_type']] += 1
            
            G.add_node(row['EndNodeID'],
                      label=eval(row['EndNodeLabels'])[0],
                      found=row['End_found'],
                      id=row['End_id'],
                      version=row['End_version'],
                      timestamp=row['End_timestamp'],
                      type=row['End_type'],
                      value=row['End_value'])
        
        for _, row in chunk.iterrows():
            relationship_counts[row['RelationshipType']] += 1
            if pd.notna(row['Relationship_scope']):
                scope_counts[row['Relationship_scope']] += 1
            
            G.add_edge(row['StartNodeID'],
                      row['EndNodeID'],
                      type=row['RelationshipType'],
                      scope=row['Relationship_scope'],
                      targetVersion=row['Relationship_targetVersion'])

    print("\nGraph created successfully!")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    print("\nRelationship Types Distribution:")
    for rel_type, count in relationship_counts.items():
        print(f"{rel_type}: {count}")
    
    print("\nScope Distribution:")
    for scope, count in scope_counts.items():
        print(f"{scope}: {count}")
    
    print("\nNode Type Distribution:")
    for node_type, count in type_counts.items():
        print(f"{node_type}: {count}")

    print("\nSample Node and Edge Verification:")

    print("\nRandom Nodes:")
    for node in list(G.nodes())[:5]:
        print(f"\nNode {node}:")
        print(f"Attributes: {G.nodes[node]}")
        
        incoming_edges = list(G.in_edges(node, data=True))
        if incoming_edges:
            print(f"Sample incoming edge: {incoming_edges[0]}")
        
        compile_deps = [
            (u, v, d) for u, v, d in incoming_edges 
            if d.get('type') == 'dependency' and d.get('scope') == 'compile'
        ]
        if compile_deps:
            print(f"Found compile scope dependencies: {len(compile_deps)}")
    
    return G

class DependencyNetworkClassifier:
    def __init__(self, G):
        self.G = G
        self.train_nodes = None
        self.test_nodes = None
        self.train_subgraph = None
        self.node_features = {}
        self.node_embeddings = {}
        self.critical_nodes = {}
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.node2vec_model = None
        self.train_threshold = 0

    def split_data(self, test_size=0.2, random_state=42):
        nodes = list(self.G.nodes())
        self.train_nodes, self.test_nodes = train_test_split(
            nodes, test_size=test_size, random_state=random_state
        )
        self.train_subgraph = self.G.subgraph(self.train_nodes).copy()
        print(f"Training nodes: {len(self.train_nodes)}, Testing nodes: {len(self.test_nodes)}")

    def extract_topological_features(self, nodes):

        print("Calculating topological features...")

        degree_cent = nx.degree_centrality(self.G)
        print("Degree calculation finished")
        betweenness_cent = nx.betweenness_centrality(self.G, k=1000)
        print("Betweenness calculation finished")
        pagerank = nx.pagerank(self.G)
        print("Pagerank calculation finished")

        clustering_coef = nx.clustering(self.G)

        local_risk = {}
        for node in self.G.nodes():
            predecessors = list(self.G.predecessors(node))
            successors = list(self.G.successors(node))
            local_risk[node] = len(successors) / (len(predecessors) + 1)

        for node in nodes:
            self.node_features[node] = {
                'degree_centrality': degree_cent[node],
                'betweenness_centrality': betweenness_cent[node],
                'pagerank': pagerank[node],
                'clustering_coefficient': clustering_coef[node],
                'local_risk': local_risk[node]
            }

    def extract_semantic_features(self, nodes):
        print("Calculating semantic features...")
        
        for node in nodes:
            node_data = self.G.nodes[node]
            
            
            scope_features = {f'scope_{scope}': 0 for scope in [
                'compile', 'runtime', 'test', 'provided', 'implementation',
                'runtimeOnly', 'system', 'optional', 'import', 'api',
                'integration-test', 'runtme', 'external'
            ]}
            
            
            type_features = {f'type_{type_}': 0 for type_ in [
                'POPULARITY_1_YEAR', 'CVE', 'FRESHNESS', 'SPEED'
            ]}
            
            
            node_label = node_data.get('label', '')
            is_artifact = 1 if node_label == 'Artifact' else 0
            is_release = 1 if node_label == 'Release' else 0
            is_added_value = 1 if node_label == 'AddedValue' else 0
            
            
            incoming_edges = self.G.in_edges(node, data=True)
            for _, _, edge_data in incoming_edges:
                if edge_data.get('type') == 'dependency':  
                    scope = edge_data.get('scope')
                    if pd.notna(scope):  
                        scope_key = f'scope_{scope}'
                        if scope_key in scope_features:  
                            scope_features[scope_key] += 1
            
            
            node_type = node_data.get('type')
            if pd.notna(node_type):  
                type_key = f'type_{node_type}'  
                if type_key in type_features:
                    type_features[type_key] = 1
            
            
            if node not in self.node_features:
                self.node_features[node] = {}
                
            self.node_features[node].update({
                'is_artifact': is_artifact,
                'is_release': is_release,
                'is_added_value': is_added_value,
                **scope_features,
                **type_features
            })


    def fit_node2vec(self, dimensions=128, walk_length=30, num_walks=200):

        print("Fitting Node2Vec on training graph...")

        node2vec = Node2Vec(self.train_subgraph, dimensions=dimensions,
                            walk_length=walk_length, num_walks=num_walks, workers=4)

        self.node2vec_model = node2vec.fit(window=10, min_count=1)

        for node in self.train_nodes:
            if str(node) in self.node2vec_model.wv:
                self.node_embeddings[node] = self.node2vec_model.wv[str(node)]

    def transform_node2vec(self, nodes):

        print("Generating embeddings for test nodes...")

        for node in nodes:
            if str(node) in self.node2vec_model.wv:
                self.node_embeddings[node] = self.node2vec_model.wv[str(node)]
            else:
                neighbors = list(self.G.neighbors(node))
                valid_embeddings = [
                    self.node_embeddings[neighbor]
                    for neighbor in neighbors
                    if neighbor in self.node_embeddings
                ]
                if valid_embeddings:
                    self.node_embeddings[node] = np.mean(valid_embeddings, axis=0)
                else:
                    self.node_embeddings[node] = np.zeros(self.node2vec_model.vector_size)
                
                if np.isnan(self.node_embeddings[node]).any():
                    print(f"Node {node} embedding contains NaN. Assigning zero vector.")
                    self.node_embeddings[node] = np.zeros(self.node2vec_model.vector_size)

    def identify_critical_nodes(self, nodes, threshold_percentile=95, train=True):

        print("Identifying critical nodes...")

        scores = {}
        for node in nodes:
            features = self.node_features[node]

            score = (
            features['degree_centrality'] * 0.2 +
                features['betweenness_centrality'] * 0.2 +
                features['pagerank'] * 0.2 +
                features['local_risk'] * 0.1 +
                (features['is_artifact'] * 0.1) +
                (features['scope_compile'] > 0) * 0.05 -
                (features['type_CVE'] > 0) * 0.15
            )

            scores[node] = score
        if train:
            threshold = np.percentile(list(scores.values()), threshold_percentile)
            self.train_threshold = threshold
        else:
            threshold = self.train_threshold

        for node in nodes:
            self.critical_nodes[node] = 1 if scores[node] >= threshold else 0

    def prepare_training_data(self, nodes):


        X = []
        y = []

        for node in nodes:
            if node in self.node_embeddings:
                features = list(self.node_features[node].values())
                embedding = list(self.node_embeddings[node])
                X.append(features + embedding)
                y.append(self.critical_nodes[node])

        print("Data Prepared")
        return np.array(X), np.array(y)

    def train_classifier(self):

        print("Training classifier...")

        self.split_data()

        print("\nProcessing training data...")
        self.extract_topological_features(self.train_nodes)
        self.extract_semantic_features(self.train_nodes)
        self.fit_node2vec()
        self.identify_critical_nodes(self.train_nodes)
        X_train, y_train = self.prepare_training_data(self.train_nodes)

        print("\nProcessing testing data...")
        self.extract_topological_features(self.test_nodes)
        self.extract_semantic_features(self.test_nodes)
        self.transform_node2vec(self.test_nodes)
        self.identify_critical_nodes(self.test_nodes)
        X_test, y_test = self.prepare_training_data(self.test_nodes)

        print("\nFitting classifier...")        
        self.classifier.fit(X_train, y_train)

        print("\nMaking Predictions...")        
        y_pred = self.classifier.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return X_test, y_test, y_pred


def load_classifier(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def save_classifier(path, classifier):
    with open(path, 'wb') as f:
        pickle.dump(classifier, f)
    
if __name__ == "__main__":
    csv_path = 'data/500k_processed.csv'
    G = create_graph_from_csv(csv_path)
    classifier = DependencyNetworkClassifier(G)
    X_test, y_test, y_pred = classifier.train_classifier()
    save_classifier('models/classifier-500.pkl', classifier)