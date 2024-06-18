import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces
from gymnasium.spaces import Dict, Discrete, Tuple


# Functions

# Populate a list "completed_list" with the elements in "reference", when they are present, and 0 otherwise, compared with "original" 
def complete_list_in_order(original, reference):
    completed_list = []
    for elem in reference:
        if elem in original:
            completed_list.append(elem)
        else:
            completed_list.append(0)
    return completed_list

# Create 9 subgrids of the 12x12 grid
def create_subgrids():
    subgrids = [[] for _ in range(9)]
    grid_size = 12  

    subgrid_coords = [(0, 0),(0, 4),(0, 8),(4, 0),(4, 4),(4, 8),(8, 0),(8, 4),(8, 8)]   #Coordinates of the subgrids in the 12x12 grid
    subgrid_size = (4, 4)   #Dimension of the subgrids
    #Populate the subgrids
    for idx, (start_row, start_col) in enumerate(subgrid_coords):
        for i in range(subgrid_size[0]):
            for j in range(subgrid_size[1]):
                number = (start_row + i) * grid_size + (start_col + j)
                subgrids[idx].append(number)

    return subgrids

def reorder_list(lst, order):
    return [lst[i] for i in order]

def find_subgrid(subgrids, number):
    for i, subgrid in enumerate(subgrids):
        if number in subgrid:
            return i
        


class DriveGrid(gym.Env):

    def __init__(self, size: int =144) -> None:
        self.state_names = Discrete(144)
        self.size = size
        self.row = int(np.sqrt(size))
        self.s = lambda x: self.state_names.index(x)
        self._agent_location = 0
        self.finish_positions= range(48,52)
        self.action_space = Tuple((Discrete(3,start=-1), Discrete(3,start=-1)))
        self.horiz_velocity = self.action_space[0]
        self.vertic_velocity = self.action_space[1]
        self.observation_space = Tuple((Discrete(size), Tuple((self.horiz_velocity, self.vertic_velocity)))) # position, horizontal and vertical velocity
        subgrids = create_subgrids() # Creation of subgrids
        order = [0, 1, 2, 5, 8, 7, 6, 3, 4]  #Order of the subgrids like the track 
        self.subgrids=reorder_list(subgrids,order)
        self.restart = False 
        
        # Create features x(s) with N,S,O,E
        self.coordinates = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1),(0,0)]  # N,S,E,O,(N,E),(N,O),(S,E),(S,O),no moves
        state_vectors = {i: [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1),(0,0)] for i in range(size)}
        
        # Populate the available actions for each state, removing actions not allowed
        for i in range(0,size): 
            if i in range(0,12): #first row
                for move in [(1, 0), (1, 1), (1, -1)]:
                    if move in state_vectors[i]:
                        state_vectors[i].remove(move)
            if i in range(132,144): #ultima riga
                for move in [(-1, 0), (-1, 1), (-1, -1)]:
                    if move in state_vectors[i]:
                        state_vectors[i].remove(move)
            if i%12 == 11:  #riga destra
                for move in [(0, 1), (1, 1), (-1, 1)]:
                    if move in state_vectors[i]:
                        state_vectors[i].remove(move)
            if i%12 == 0:  #riga sinistra
                for move in [(0, -1), (1, -1), (-1, -1)]:
                    if move in state_vectors[i]:
                        state_vectors[i].remove(move)
            if i in range(40,43):
                for move in [(-1, 0), (-1, 1), (-1, -1)]:
                    if move in state_vectors[i]:
                        state_vectors[i].remove(move)
            if i == 44:
                if (-1, -1) in state_vectors[i]:
                    state_vectors[i].remove((-1, -1))
            if i in range(40, 44):
                if (-1, 0) in state_vectors[i]:
                    state_vectors[i].remove((-1, 0))
            if i in range(39, 43):
                if (-1, 1) in state_vectors[i]:
                    state_vectors[i].remove((-1, 1))
            if i in range(41, 45):
                if (-1, -1) in state_vectors[i]:
                    state_vectors[i].remove((-1, -1))
            if i in range(100, 104):
                if (1, 0) in state_vectors[i]:
                    state_vectors[i].remove((1, 0))
            if i in range(99, 103):
                if (1, 1) in state_vectors[i]:
                    state_vectors[i].remove((1, 1))
            if i in range(101, 105):
                if (1, -1) in state_vectors[i]:
                    state_vectors[i].remove((1, -1))
            if i in [56, 68, 80, 92]:
                if (0, -1) in state_vectors[i]:
                    state_vectors[i].remove((0, -1))
            if i in [68, 80, 92, 104]:
                if (1, -1) in state_vectors[i]:
                    state_vectors[i].remove((1, -1))
            if i in [44, 56, 68, 80]:
                if (-1, -1) in state_vectors[i]:
                    state_vectors[i].remove((-1, -1))
            if i in [51, 63, 75, 87]:
                if (0, 1) in state_vectors[i]:
                    state_vectors[i].remove((0, 1))
            if i in [63, 75, 87, 99]:
                if (1, 1) in state_vectors[i]:
                    state_vectors[i].remove((1, 1))
            if i in [39, 51, 63, 75]:
                if (-1, 1) in state_vectors[i]:
                    state_vectors[i].remove((-1, 1))


        # From the entire vector of coordinates, substitute with 0 the not available actions (actions removed from the list)   
        self.available_actions = [complete_list_in_order(lst, self.coordinates) for lst in list(state_vectors.values())]
        # Vector with 1 or 0 for allowed actions
        self.available_actions_value = [[1 if element != 0 else 0 for element in sublist] for sublist in self.available_actions]
        

                                          
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # Choose the agent's to start always on location A
        self._agent_location = 36
        self.horiz_velocity=0
        self.vertic_velocity=0
        info = self._get_info()
        obs = self._get_obs()
        return obs, info

    
    def _move(self, action):
        # Velocity is from 0 to 5, so sum the chosen action to the actual velocity
        new_horiz_velocity= self.horiz_velocity + action[0]
        new_vertic_velocity= self.vertic_velocity +action[1]
        new_position = self._agent_location + (new_horiz_velocity) + (new_vertic_velocity*self.row)
        
        # Divide my grid into 9 subgrids, ordered in clockwise mode. 
        # To avoid that the car goes in the avoided centre of the track, or in counterclockwise mode, 
        # it has to be checked that the next position of the car is nothing else than: 
        # - in the same previous grid
        # - in the next grid
        # - in the first grid due to the reset 
        
        subgrid_actual_pos = find_subgrid(self.subgrids, self._agent_location)
        subgrid_next_pos = find_subgrid(self.subgrids, new_position)  
        
        if (0 <= new_position < self.size):            
            if (abs(subgrid_actual_pos - subgrid_next_pos) not in [0,1] ) or \
                ((subgrid_next_pos - subgrid_actual_pos) not in [0,1,2] ) or \
                    ( subgrid_next_pos == 8 ) or \
                        (self._agent_location % self.row == 0 and new_horiz_velocity < 0) or \
                            (self._agent_location % self.row == self.row - 1 and new_horiz_velocity > 0):
                self.reset()
                self.restart = True 
                return

            #cannot go over 5 or under -5 for the velocity
            if new_horiz_velocity > 5: new_horiz_velocity = 5
            if new_vertic_velocity > 5: new_vertic_velocity = 5
            if new_horiz_velocity < -5: new_horiz_velocity = -5
            if new_vertic_velocity < -5: new_vertic_velocity = -5
              
            self._agent_location = new_position
            self.horiz_velocity= new_horiz_velocity
            self.vertic_velocity = new_vertic_velocity
            self.restart = False
            
        else:
            self.reset()
            self.restart = True
            return


        
    def step(self, action):
        self._move(action)
        terminated = self._agent_location in self.finish_positions
        reward = self.rewards(self._agent_location) # always -1
        truncated = self.restart
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info


    def rewards(self, state):
        if state == 0:
            return 0
        if state in self.finish_positions:
            return 1000
        if self.restart == True:
            return -5
        else:
            return -1    


    def _get_obs(self):
        return (self._agent_location, (self.horiz_velocity ,self.vertic_velocity))

    def _get_row(self):
        return self.row
    
    def _get_info(self):
        return dict()


    #Register the environment
register(
          id="DriveGrid-v0",
          entry_point=lambda size: DriveGrid(size=size),
          max_episode_steps=300,
     )
