"""
Utility functions for saving and loading agents.

These functions help create standalone Python files for agent submission.
"""

import os
import numpy as np # type: ignore

def save_qlearning_agent(agent, output_path, agent_class_name="QLearningTrainedAgent"):
    """
    Save a trained Q-learning agent as a standalone Python file.
    
    Args:
        agent: The trained Q-learning agent instance
        output_path: Path where to save the agent file
        agent_class_name: Name for the agent class in the saved file
    
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract agent parameters
    position_bins = getattr(agent, 'position_bins', 8)
    velocity_bins = getattr(agent, 'velocity_bins', 4)
    wind_bins = getattr(agent, 'wind_bins', 8)
    
    # Start building the file content
    file_content = f'''"""
Q-Learning Agent for the Sailing Challenge - Trained Model

This file contains a Q-learning agent trained on the sailing environment.
The agent uses a discretized state space and a Q-table for decision making.
"""

import numpy as np
from agents.base_agent import BaseAgent

class {agent_class_name}(BaseAgent):
    """
    A Q-learning agent trained on the sailing environment.
    Uses a discretized state space and a lookup table for actions.
    """
    
    def __init__(self):
        """Initialize the agent with the trained Q-table."""
        super().__init__()
        self.np_random = np.random.default_rng()
        
        # State discretization parameters
        self.position_bins = {position_bins}
        self.velocity_bins = {velocity_bins}
        self.wind_bins = {wind_bins}
        
        # Q-table with learned values
        self.q_table = {{}}
        self._init_q_table()
    
    def _init_q_table(self):
        """Initialize the Q-table with learned values."""
'''
    
    # Add all Q-values
    for state, values in agent.q_table.items():
        q_values_str = np.array2string(values, precision=4, separator=', ')
        file_content += f"        self.q_table[{state}] = np.array({q_values_str})\n"
    
    # Add remaining methods
    file_content += '''
    def discretize_state(self, observation):
        """Convert continuous observation to discrete state for Q-table lookup."""
        # Extract position, velocity and wind from observation
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        
        # Discretize position (assume 128x128 grid)
        grid_size = 128
        x_bin = min(int(x / grid_size * self.position_bins), self.position_bins - 1)
        y_bin = min(int(y / grid_size * self.position_bins), self.position_bins - 1)
        
        # Discretize velocity direction
        v_magnitude = np.sqrt(vx**2 + vy**2)
        if v_magnitude < 0.1:  # If velocity is very small, consider it as a separate bin
            v_bin = 0
        else:
            v_direction = np.arctan2(vy, vx)  # Range: [-pi, pi]
            v_bin = int(((v_direction + np.pi) / (2 * np.pi) * (self.velocity_bins-1)) + 1) % self.velocity_bins
        
        # Discretize wind direction
        wind_direction = np.arctan2(wy, wx)  # Range: [-pi, pi]
        wind_bin = int(((wind_direction + np.pi) / (2 * np.pi) * self.wind_bins)) % self.wind_bins
        
        # Return discrete state tuple
        return (x_bin, y_bin, v_bin, wind_bin)
        
    def act(self, observation):
        """Choose the best action according to the learned Q-table."""
        # Discretize the state
        state = self.discretize_state(observation)
        
        # Use default actions if state not in Q-table
        if state not in self.q_table:
            return 0  # Default to North if state not seen during training
        
        # Return action with highest Q-value
        return np.argmax(self.q_table[state])
    
    def reset(self):
        """Reset the agent for a new episode."""
        pass  # Nothing to reset
        
    def seed(self, seed=None):
        """Set the random seed."""
        self.np_random = np.random.default_rng(seed)
'''
    
    # Write the file
    with open(output_path, 'w') as f:
        f.write(file_content)
    
    print(f"Agent saved to {output_path}")
    print(f"The file contains {len(agent.q_table)} state-action pairs.")
    print(f"You can now use this file with validate_agent.ipynb and evaluate_agent.ipynb")



def save_dqn_agent(agent, output_path, agent_class_name="DQNTrainedAgent"):
    """
    Save a trained DQN agent as a standalone Python file.
    
    Args:
        agent: The trained DQN agent instance
        output_path: Path where to save the agent file
        agent_class_name: Name for the agent class in the saved file
    
    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Récupération des poids
    weights = agent.policy_net.copy_weights()
    
    # Préparation du contenu du fichier
    file_content = '''"""
A DQN agent trained on the sailing environment.
"""

import numpy as np
from evaluator.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
    '''
    for key, val in weights.items():
        file_content += f"        self.{key} = np.array({val.tolist()})\n"

    file_content += '''
    def forward(self, X):
        l1 = np.dot(X, self.W1) + self.b1
        relu1 = np.maximum(0, l1) 
        l2 = np.dot(relu1, self.W2) + self.b2
        relu2 = np.maximum(0, l2) 
        l3 = np.dot(relu2, self.W3) + self.b3
        return l3

    def get_features(self, observation, training=True):
        """create (22) features from observations"""
        x, y = observation[0], observation[1] #position
        vx, vy = observation[2], observation[3] #speed
        wx, wy = observation[4], observation[5] #wind
        flattened_world = observation[6+32768:].reshape(128,128) #world map

        if training :
            #randomize wind in training (for generalization)
            theta = self.np_random.uniform(-np.pi/2, np.pi/2) 
            scale = self.np_random.uniform(0.7, 1.3)
            wx, wy = wx * scale, wy * scale
            wx_rot = wx * np.cos(theta) - wy * np.sin(theta)
            wy_rot = wx * np.sin(theta) + wy * np.cos(theta)
            wx, wy = wx_rot, wy_rot

        #clip information (avoid extreme values)
        vx = np.clip(vx, -5, 5) / 5
        vy = np.clip(vy, -5, 5) / 5
        wx = np.clip(wx, -5, 5) / 5
        wy = np.clip(wy, -5, 5) / 5

        #store features
        features = []

        #feature with local speed and position
        v = np.sqrt(vx**2+vy**2) #norm 2 of the speed vector 
        features.extend([x/128, y/128, vx, vy, v])

        #feature with goal
        goal = [64, 127]
        dx, dy = goal[0] - x, goal[1] - y
        dist_to_goal = np.sqrt(dx**2 + dy**2)
        norm = np.sqrt(128**2 + 128**2)
        angle_to_goal = np.arctan2(dy, dx)
        features.extend([dist_to_goal/norm, np.cos(angle_to_goal), np.sin(angle_to_goal)])

        #vmg (velocity made good)
        if v>0.05:
            boat_angle = np.arctan2(vy, vx)
            vmg = v * np.cos(boat_angle - angle_to_goal)
        else :
            vmg = v
            boat_angle = angle_to_goal
        features.append(vmg)

        #feature angle with wind
        wind_angle = np.arctan2(wy, wx)
        relative_wind = abs((wind_angle - boat_angle + np.pi) % (2*np.pi) - np.pi)
        wind_goal_angle = abs((wind_angle - angle_to_goal + np.pi) % (2*np.pi) - np.pi)
        features.extend([np.cos(relative_wind), np.sin(relative_wind),np.cos(wind_goal_angle), np.sin(wind_goal_angle), np.sqrt(wx**2 + wy**2)])

        #feature of danger
        direction = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
        #we choose a direction and create a feature of danger to the island
        for direction_x, direction_y in direction: 
            danger = 0
            for distance in range(1,25):
                new_x = int(x + direction_x*distance)
                new_y = int(y + direction_y*distance)
                if 0 <= new_x < 128 and 0<= new_y < 128:
                    if flattened_world[new_x,new_y] == 1:
                        danger = (25 - distance)/25 
                        break
            features.append(danger)
        
        return np.array(features)
        
    def act(self, observation):
        """Choose an action"""
        state = self.get_features(observation, training=False).reshape(1, -1)
        q_values = self.forward(state)
        return int(np.argmax(q_values))
        
    def reset(self):
        """Reset the agent for a new episode."""
        # Nothing to reset
        pass
        
    def seed(self, seed=None):
        """Set the random seed."""
        self.np_random = np.random.default_rng(seed)
    
'''

    with open(output_path, 'w') as f:
        f.write(file_content)
    
    print(f"Agent saved to {output_path}")