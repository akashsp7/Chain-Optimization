
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import wandb  
import os
from tqdm import tqdm
from critical import DependencyNetworkClassifier, load_classifier

class DependencyChainEnv(gym.Env):
    
    def __init__(self, classifier):
        super(DependencyChainEnv, self).__init__()
        
        self.graph = classifier.G
        self.node_features = classifier.node_features
        self.node_embeddings = classifier.node_embeddings
        self.critical_nodes = classifier.critical_nodes
        self.valid_nodes = self._filter_valid_nodes()
        self.n_nodes = len(self.valid_nodes)
        self.embedding_dim = len(next(iter(classifier.node_embeddings.values())))
        self.action_space = spaces.Discrete(self.n_nodes)
        self.base_features = 5  
        self.node_type_features = 3  
        self.relationship_types = 3  
        self.scope_features = 13  
        self.value_type_features = 4  
        
        total_features = (self.base_features + self.node_type_features + 
                         self.relationship_types + self.scope_features + 
                         self.value_type_features + self.embedding_dim + 6)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32)
        
        self.scope_distribution = {
            'compile': 203469,
            'runtime': 26815,
            'test': 75891,
            'provided': 25985,
            'implementation': 39,
            'runtimeOnly': 1,
            'system': 16,
            'optional': 15,
            'import': 4,
            'api': 19,
            'integration-test': 1,
            'runtme': 2,
            'external': 2}
        
        self.type_distribution = {
            'POPULARITY_1_YEAR': 40152,
            'CVE': 39859,
            'FRESHNESS': 39855,
            'SPEED': 1751}
        
        self.total_edges = 499760
        self.relationship_distribution = {
            'dependency': 332259,
            'addedValues': 121617,
            'relationship_AR': 46124}
        
        self.max_chain_length = 10  
        self.required_metrics = {    
            'security_score': 0.7,
            'performance_score': 0.6,
            'freshness_score': 0.6,
            'avg_local_risk': 0.3,
            'critical_nodes_ratio': 0.2}
    
    def _filter_valid_nodes(self):
        valid_nodes = []
        for node in self.graph.nodes():
            
            if (self.graph.in_degree(node) > 0 or self.node_features[node]['is_artifact']) and \
            self.graph.out_degree(node) > 0:
                valid_nodes.append(node)
        
        self.node_to_idx = {node: idx for idx, node in enumerate(valid_nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        return valid_nodes
    
    def get_node_features_vector(self, node):
        
        features = self.node_features[node]
        base_features = [
            features['degree_centrality'],
            features['betweenness_centrality'],
            features['pagerank'],
            features['clustering_coefficient'],
            features['local_risk']]
        
        node_type_features = [
            features['is_artifact'],
            features['is_release'],
            features['is_added_value']]
        
        relationship_counts = [0, 0, 0]  
        for _, _, edge_data in self.graph.edges(node, data=True):
            if edge_data.get('type') == 'dependency':
                relationship_counts[0] += 1
            elif edge_data.get('type') == 'addedValues':
                relationship_counts[1] += 1
            elif edge_data.get('type') == 'relationship_AR':
                relationship_counts[2] += 1
        relationship_features = [count/self.total_edges for count in relationship_counts]
        
        scope_features = []
        for scope in self.scope_distribution.keys():
            scope_key = f'scope_{scope}'
            if scope_key in features:
                
                scope_features.append(features[scope_key] / self.scope_distribution[scope])
            else:
                scope_features.append(0)
        
        value_type_features = []
        for type_key in self.type_distribution.keys():
            feature_key = f'type_{type_key}'
            if feature_key in features:
                value_type_features.append(features[feature_key] / self.type_distribution[type_key])
            else:
                value_type_features.append(0)
        
        return np.array(base_features + 
                       node_type_features + 
                       relationship_features + 
                       scope_features + 
                       value_type_features)
    
    def calculate_chain_metrics(self, chain):

        if not chain:
            return {
                'chain_length': 0,
                'avg_local_risk': 0,
                'critical_nodes_ratio': 0,
                'security_score': 1.0,
                'performance_score': 0,
                'freshness_score': 0
            }
        
        local_risks = [self.node_features[node]['local_risk'] for node in chain]
        avg_local_risk = np.mean(local_risks)
        critical_count = sum(self.critical_nodes[node] for node in chain)
        critical_ratio = critical_count / len(chain)
        cve_count = sum(self.node_features[node]['type_CVE'] for node in chain)
        security_score = np.exp(-cve_count)
        
        perf_scores = [
            1.0 if self.node_features[node]['type_SPEED'] else 0.5
            for node in chain
        ]
        performance_score = np.mean(perf_scores)
        
        freshness_scores = [
            1.0 if self.node_features[node]['type_FRESHNESS'] else 0.5
            for node in chain
        ]
        freshness_score = np.mean(freshness_scores)
        
        return {
            'chain_length': len(chain),
            'avg_local_risk': avg_local_risk,
            'critical_nodes_ratio': critical_ratio,
            'security_score': security_score,
            'performance_score': performance_score,
            'freshness_score': freshness_score
        }
    
    def get_state(self):
        current_node = self.current_chain[-1]        
        node_features = self.get_node_features_vector(current_node)
        node_embedding = self.node_embeddings[current_node]
        
        metrics = self.calculate_chain_metrics(self.current_chain)
        metrics_vector = np.array([
            metrics['chain_length'] / self.n_nodes,  
            metrics['avg_local_risk'],
            metrics['critical_nodes_ratio'],
            metrics['security_score'],
            metrics['performance_score'],
            metrics['freshness_score']
        ])
        
        state = np.concatenate([
            node_features,
            node_embedding,
            metrics_vector
        ])
        
        return state
    
    def step(self, action):

        if not isinstance(action, (int, np.integer)) or action < 0 or action >= len(self.idx_to_node):
            raise ValueError(f"Invalid action: {action}. Must be integer between 0 and {len(self.idx_to_node)-1}")
        
        chosen_node = self.idx_to_node[action]
            
        if chosen_node in self.current_chain:
            return self.get_state(), -20.0, True, self.calculate_chain_metrics(self.current_chain)
            
        if len(self.current_chain) > 0:
            last_node = self.current_chain[-1]
            if not self.graph.has_edge(last_node, chosen_node):
                return self.get_state(), -20.0, True, self.calculate_chain_metrics(self.current_chain)
            
        self.current_chain.append(chosen_node)
            
        new_state = self.get_state()
        reward = self._calculate_reward()
            
        done = self._is_done()
            
        info = self.calculate_chain_metrics(self.current_chain)
        
        return new_state, reward, done, info
    
    def _calculate_reward(self):

        metrics = self.calculate_chain_metrics(self.current_chain)
        
        if not self._is_valid_chain():
            return -20.0    

        chain_length = len(self.current_chain)
        if chain_length <= 3:
            base_reward = 0.5
        elif chain_length <= 5:
            base_reward = 5.0
        else:
            base_reward = 10.0

        scope_rewards = 0
        for node in self.current_chain:
            for scope, freq in self.scope_distribution.items():
                scope_key = f'scope_{scope}'
                if self.node_features[node].get(scope_key, 0) > 0:
                    scope_rewards += np.log(1 + freq/1000)
        
        security_score = 2.0 * metrics['security_score']
        popularity_score = 4.0 * sum(self.node_features[node]['type_POPULARITY_1_YEAR'] 
                                for node in self.current_chain) / len(self.current_chain)
        freshness_score = 4.0 * sum(self.node_features[node]['type_FRESHNESS'] 
                                for node in self.current_chain) / len(self.current_chain)
        speed_bonus = 3.0 * sum(self.node_features[node]['type_SPEED'] 
                            for node in self.current_chain) / len(self.current_chain)
        
        total_reward = (base_reward + 
                    scope_rewards + 
                    security_score + 
                    popularity_score + 
                    freshness_score + 
                    speed_bonus)
        
        if self._all_requirements_met():
            total_reward += 50.0
        
        return total_reward

    
    def _is_valid_chain(self):
        
        if len(set(self.current_chain)) != len(self.current_chain):
            return False        
        
        for i in range(len(self.current_chain) - 1):
            if not self.graph.has_edge(self.current_chain[i], self.current_chain[i + 1]):
                return False
        
        return True
    
    def _is_done(self):
        metrics = self.calculate_chain_metrics(self.current_chain)
        
        return (
            len(self.current_chain) >= self.max_chain_length or  
            not self._is_valid_chain() or     
            metrics['critical_nodes_ratio'] > 0.5 or  
            self._all_requirements_met())
    
    def _all_requirements_met(self):
        metrics = self.calculate_chain_metrics(self.current_chain)
        return (
            metrics['security_score'] > self.required_metrics['security_score'] and
            metrics['performance_score'] > self.required_metrics['performance_score'] and
            metrics['freshness_score'] > self.required_metrics['freshness_score'] and
            metrics['avg_local_risk'] < self.required_metrics['avg_local_risk'] and
            metrics['critical_nodes_ratio'] < self.required_metrics['critical_nodes_ratio'] and
            len(self.current_chain) >= 7
        )
        
    def reset(self):
        
        artifact_nodes = [
            node for node in self.valid_nodes
            if self.node_features[node]['is_artifact'] == 1
        ]
        
        if not artifact_nodes:
            
            print("Warning: No artifact nodes found in valid nodes. Using random valid node.")
            self.current_chain = [np.random.choice(self.valid_nodes)]
        else:
            self.current_chain = [np.random.choice(artifact_nodes)]
        
        return self.get_state()
    
    def render(self, mode='human'):

        if mode != 'human':
            return
        
        print("\nCurrent Chain State:")
        print("Chain:", " -> ".join(map(str, self.current_chain)))
        metrics = self.calculate_chain_metrics(self.current_chain)
        print("Metrics:", metrics)
        
class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logprobs = []
        self.dones = []
        self.infos = []  
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logprobs = []
        self.dones = []
        self.infos = []  
        
    def compute_returns(self, last_value, gamma=0.99, gae_lambda=0.95):
        rewards = np.array(self.rewards + [last_value])
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [0])
        
        advantages = np.zeros_like(rewards[:-1])
        lastgaelam = 0
        
        for t in reversed(range(len(rewards)-1)):
            if dones[t]:
                nextvals = 0
            else:
                nextvals = values[t + 1]
            delta = rewards[t] + gamma * nextvals - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * lastgaelam
            
        returns = advantages + values[:-1]
        return returns, advantages
    
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        self.scope_network = nn.Sequential(
            nn.Linear(13, 32),           
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64)
        )
        
        self.relationship_network = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        
        self.quality_network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        
        main_input_size = state_dim - 20         
        self.main_network = nn.Sequential(
            nn.Linear(main_input_size + 64 + 32 + 32, 512),          
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        
        self.actor_attention = nn.MultiheadAttention(256, 4)
        self.actor = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.LogSoftmax(dim=-1)
        )
        
        self.critic_value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)        
        )
        
        self.critic_quality = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)        
        )
        
    def forward(self, state):
        scope_features = state[..., :13]
        relationship_features = state[..., 13:16]
        quality_features = state[..., 16:20]
        main_features = state[..., 20:]
        
        scope_encoded = self.scope_network(scope_features)
        relationship_encoded = self.relationship_network(relationship_features)
        quality_encoded = self.quality_network(quality_features)
        
        combined = torch.cat([
            main_features,
            scope_encoded,
            relationship_encoded,
            quality_encoded
        ], dim=-1)
        
        features = self.main_network(combined)
        
        attended_features, _ = self.actor_attention(
            features.unsqueeze(0),
            features.unsqueeze(0),
            features.unsqueeze(0)
        )
        attended_features = attended_features.squeeze(0)
        
        action_probs = self.actor(attended_features)
        value = self.critic_value(features)
        quality_pred = self.critic_quality(features)
        
        return action_probs, value, quality_pred
    
class DependencyChainOptimizer:
    
    def __init__(self, env, learning_rate=5e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, c1=1.0, c2=0.01):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1        
        self.c2 = c2        
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.memory = PPOMemory()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
    
    def select_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, value, quality_pred = self.policy(state)
            
            valid_actions = self._get_valid_actions(self.env.current_chain[-1])
            
            if not valid_actions:

                last_action = self.env.current_chain[-1] if self.env.current_chain else 0
                last_action_idx = self.env.node_to_idx.get(last_action, 0)
                return last_action_idx, 0.0, 0.0
            
            try:
                mask = torch.zeros_like(action_probs)
                mask[0, valid_actions] = 1
                masked_probs = torch.where(mask > 0, 
                                        torch.exp(action_probs), 
                                        torch.tensor(1e-8).to(self.device))
                
                prob_sum = masked_probs.sum(dim=1, keepdim=True)
                masked_probs = masked_probs / prob_sum
                
                if torch.isnan(masked_probs).any():
                    masked_probs = torch.zeros_like(action_probs)
                    masked_probs[0, valid_actions] = 1.0 / len(valid_actions)
                
                dist = Categorical(masked_probs)
                action = dist.sample()
                action_logprob = dist.log_prob(action)
                
                self.last_quality_pred = quality_pred.detach().cpu().numpy()
                
                return action.item(), action_logprob.item(), value.item()
                
            except Exception as e:
                print(f"Error in action selection: {e}")
                chosen_idx = np.random.choice(valid_actions)
                return chosen_idx, 0.0, 0.0
    
    def _get_valid_actions(self, current_node):

        valid_actions = []
            
        if current_node not in self.env.node_to_idx:
            return valid_actions
            
        for idx, node in self.env.idx_to_node.items():        
            if node in self.env.current_chain:
                continue
                        
            if not self.env.graph.has_edge(current_node, node):
                continue
                        
            has_valid_next = False
            for next_node in self.env.graph.neighbors(node):
                if next_node not in self.env.current_chain:
                    has_valid_next = True
                    break
                            
            if has_valid_next:
                valid_actions.append(idx)
                
        return valid_actions
    
    def update(self, batch_size=32):
        if len(self.memory.states) < batch_size:
            return
            
        returns, advantages = self.memory.compute_returns(
            last_value=0, gamma=self.gamma, gae_lambda=self.gae_lambda
        )
        
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.memory.actions)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(self.memory.logprobs)).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(4):        

            indices = np.random.permutation(len(states))
            
            for start_idx in range(0, len(states), batch_size):
                idx = indices[start_idx:start_idx + batch_size]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_logprobs = old_logprobs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                
                action_probs, values, quality_pred = self.policy(batch_states)
                dist = Categorical(torch.exp(action_probs))
                curr_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratios = torch.exp(curr_logprobs - batch_old_logprobs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns.squeeze())
                
                quality_target = torch.FloatTensor([
                    [info.get('security_score', 0),
                    info.get('performance_score', 0),
                    info.get('freshness_score', 0),
                    info.get('critical_nodes_ratio', 0)]
                    for info in self.memory.infos
                ]).to(self.device)
                quality_loss = nn.MSELoss()(quality_pred, quality_target[idx])
                
    
                loss = (policy_loss + 
                    self.c1 * value_loss + 
                    0.5 * quality_loss -        
                    self.c2 * entropy)
                
    
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        self.memory.clear()
    
    def train(self, num_episodes=1000, max_steps=100, save_path="dependency_rl_model.pth"):

        wandb.init(project="dependency-chain-optimization", config={
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_epsilon": self.clip_epsilon,
            "num_episodes": num_episodes,
            "max_steps": max_steps
        })
        
        best_reward = float('-inf')
        episode_rewards = deque(maxlen=100)
        
        for episode in tqdm(range(num_episodes)):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
    
                action, logprob, value = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                self.memory.states.append(state)
                self.memory.actions.append(action)
                self.memory.rewards.append(reward)
                self.memory.values.append(value)
                self.memory.logprobs.append(logprob)
                self.memory.dones.append(done)
                self.memory.infos.append(info)        
                
                state = next_state
                episode_reward += reward
                
                if len(self.memory.states) >= 2048:        
                    self.update()
                
                if done:
                    break
            

            if len(self.memory.states) > 0:
                self.update()
            

            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards)
            

            wandb.log({
                "episode_reward": episode_reward,
                "average_reward": avg_reward,
                "episode_length": step + 1,
                "chain_metrics": info
            })
            

            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(self.policy.state_dict(), save_path)
                print(f"New best model saved with average reward: {avg_reward:.2f}")
        
        wandb.finish()
        return self.policy

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, weights_only=True))
        
    def evaluate(optimizer, env, num_episodes=10):
        results = {}
        best_stage = None        
        best_score = float('-inf')
        
        for stage in range(3):
            model_path = f"reinforcement_models/dependency_rl_model_stage_{stage}.pth"
            
            try:    
                optimizer.load(model_path)
                
                total_rewards = []
                chain_lengths = []
                security_scores = []
                performance_scores = []
                freshness_scores = []
                valid_chains = 0
                
                for episode in range(num_episodes):
                    state = env.reset()
                    episode_reward = 0
                    done = False
                    
                    while not done:
                        action, _, _ = optimizer.select_action(state)
                        state, reward, done, info = env.step(action)
                        episode_reward += reward
                    
                    total_rewards.append(episode_reward)
                    chain_lengths.append(info['chain_length'])
                    security_scores.append(info['security_score'])
                    performance_scores.append(info['performance_score'])
                    freshness_scores.append(info['freshness_score'])
                    
                    if env._is_valid_chain():
                        valid_chains += 1
                    

                results[stage] = {
                    'avg_reward': np.mean(total_rewards),
                    'avg_chain_length': np.mean(chain_lengths),
                    'avg_security': np.mean(security_scores),
                    'avg_performance': np.mean(performance_scores),
                    'avg_freshness': np.mean(freshness_scores),
                    'valid_chain_ratio': valid_chains / num_episodes,
                    'std_reward': np.std(total_rewards)
                }
                
                weighted_score = (
                    results[stage]['avg_reward'] * 0.3 +
                    results[stage]['valid_chain_ratio'] * 0.3 +
                    results[stage]['avg_security'] * 0.15 +
                    results[stage]['avg_performance'] * 0.15 +
                    results[stage]['avg_freshness'] * 0.1
                )
                
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_stage = stage
                
                print(f"\nStage {stage + 1} Summary:")
                print(f"Average Reward: {results[stage]['avg_reward']:.2f} ± {results[stage]['std_reward']:.2f}")
                print(f"Average Chain Length: {results[stage]['avg_chain_length']:.2f}")
                print(f"Average Security Score: {results[stage]['avg_security']:.2f}")
                print(f"Average Performance Score: {results[stage]['avg_performance']:.2f}")
                print(f"Average Freshness Score: {results[stage]['avg_freshness']:.2f}")
                print(f"Valid Chain Ratio: {results[stage]['valid_chain_ratio']:.2%}")
                
            except Exception as e:
                print(f"Error evaluating stage {stage} model: {str(e)}")
        
        if results and best_stage is not None:
            print("\nBest Model Analysis:")
            print(f"Stage {best_stage + 1} model performs best with:")
            print(f"Weighted Score: {best_score:.2f}")
            print("Individual Metrics:")
            for metric, value in results[best_stage].items():
                print(f"  {metric}: {value:.2f}")
        else:
            print("\nNo models were successfully evaluated.")
        
        return results, best_stage if best_stage is not None else -1
    
def train_dependency_optimizer(classifier, episodes):
    print("Initializing environment and optimizer...")
    os.makedirs("reinforcement_models", exist_ok=True)
    env = DependencyChainEnv(classifier)

    curricula = [
        {
            'max_chain_length': 5,
            'required_metrics': {
                'security_score': 0.6,
                'performance_score': 0.5,
                'freshness_score': 0.5,
                'avg_local_risk': 0.3,
                'critical_nodes_ratio':0.3
            },
            'episodes': episodes
        },
        {
            'max_chain_length': 10,
            'required_metrics': {
                'security_score': 0.6,
                'performance_score': 0.5,
                'freshness_score': 0.5,
                'avg_local_risk': 0.3,
                'critical_nodes_ratio':0.3
            },
            'episodes': episodes
        },
        {
            'max_chain_length': 15,
            'required_metrics': {
                'security_score': 0.6,
                'performance_score': 0.5,
                'freshness_score': 0.5,
                'avg_local_risk': 0.3,
                'critical_nodes_ratio':0.3
            },
            'episodes': episodes
        }
    ]
    
    optimizer = DependencyChainOptimizer(
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2
    )
    
    for stage_idx, curriculum in enumerate(curricula):
        print(f"\nStarting curriculum stage {stage_idx + 1}")
        env.max_chain_length = curriculum['max_chain_length']
        env.required_metrics = curriculum['required_metrics']
        
        trained_policy = optimizer.train(
            num_episodes=curriculum['episodes'],
            max_steps=curriculum['max_chain_length'],
            save_path=f"reinforcement_models/dependency_rl_model_stage_{stage_idx}.pth"
        )
    
    print("\nTraining completed. Starting evaluation...")
    eval_reward = optimizer.evaluate(optimizer.env, num_episodes=20)
    
    return {
        'optimizer': optimizer,
        'trained_policy': trained_policy,
        'eval_reward': eval_reward,
        'env': env
    }
    
if __name__ == "__main__":

    classifier = load_classifier('models/classifier-500.pkl')
    results = train_dependency_optimizer(classifier, 100)
    optimizer = results['optimizer']
    env = results['env']
    eval_reward = results['eval_reward']

    print(f"\nFinal evaluation reward: {eval_reward}")

    print("\nGenerating sample optimized chain:")
    state = env.reset()
    done = False

    while not done:
        action, _, _ = optimizer.select_action(state)
        state, reward, done, info = env.step(action)
        
    print("Generated chain:", " -> ".join(map(str, env.current_chain)))
    print("Chain metrics:", info)
