import numpy as np
from settings import e
from collections import deque


def preprocess(game_state):
     '''
     Prepares the game frame/state for the network to process. Gives grayscale values
     to the all the different objects and normalizes it to a range of -128 to 127, so
     the frame can be stored a int8 array (1 byte per entry) for minimum memory consumption.
     Also crops the frame from (17, 17) to (15, 15), i.e. getting rid of the stonewalls surrounding the arena.
     '''
     try:
          # get state info
          grid = game_state["arena"].copy()
          own_pos = game_state["self"][:2]
          enemies = game_state["others"][:]
          coins = game_state["coins"]
          bombs = game_state["bombs"][:]
          explosions = game_state["explosions"].copy()
     

          # adjust grid, pixel range 0 - 255 grayscale img
          # free tiles
          grid[grid == 0] = 127
          # stone walls
          grid[grid == -1] = 80
          # crates
          grid[grid == 1] = 180
          # place self
          

          # place bombs
          if bombs:
               for bomb in bombs:
                    grid[bomb[:2]] = 20

          # place coins
          for coin in coins:
               grid[coin] = 255

          if np.any(explosions > 0):
               grid[explosions > 0] = 0

          for enemy in enemies:
               grid[enemy[:2]] = 40

          grid[own_pos] = 225

          ### CROPPING to 15x15
          grid_cropped = grid[1:16][:, 1:16]

          # normalize from -128 to 127 (int8)
          grid_cropped = grid_cropped - 128

          # return as 1 byte int for less memory consumption
          return grid_cropped.astype(np.int8)
     except:
          raise RuntimeError("Preprocessing failed")


def get_reward(self):
     '''
     Determines the reward the agent got for taking an action in a certain state.
     '''
     # "Field Of View"
     BOMB_RADIUS = 5
     ENEMY_RADIUS = 5
     COIN_RADIUS = 7.5

     try:
          
          # get coin info
          coins_on_grid = self.game_state["coins"]
          # get agent info
          self_stats = self.game_state["self"]
          x, y = self_stats[:2]
          # get bombs info
          bombs = self.game_state["bombs"][:]
          # get enemy info
          others = self.game_state["others"][:]

          # reset upon new episode
          if self.game_state["step"] == 2:
               self.dodge = False
               self.nearest_coin = 1e100
               self.bomb_dist = 0
               self.dodge_history = deque([False for i in range(3)], maxlen=3)
               self.enemy_dist = 1e100
          
          # compute distance to enemies. If they are close enough, the agent is allowed to drop as many bombs as he pleases
          if others:
               enemy_dist, enemy = dist_to_nearest(others, (x, y))
               if enemy_dist < ENEMY_RADIUS:
                    self.enemy_dist = enemy_dist
               else:
                    self.enemy_dist = 1e100
          else:
               self.enemy_dist = 1e100

          # compute distance to nearest bomb and give rewards, if the agent dodges the bomb
          if bombs:
               bomb_dist, nearest_bomb = dist_to_nearest(bombs, (x, y))
               xb, yb = bombs[nearest_bomb][:2]
               # if the bomb is in a certain radius to the agent, give him a reward if he manages to
               # move around a corner. dodge history, so that there's wont be feedback loops of the agent 
               # trying to trick the system (can dodge a bomb only once)
               if bomb_dist < BOMB_RADIUS:
                    if not (True in self.dodge_history):
                         self.dodge = smart_dodge((xb, yb), bomb_dist, (x, y), self.last_pos)
                    else:
                         self.dodge = False
                    self.dodge_history.append(self.dodge)
               else:
                    self.dodge = False 
                    self.dodge_history.append(self.dodge) 
                    self.bomb_dist = 0
          else:
               self.bomb_dist = 0
               bomb_dist = 0
               self.dodge = False
               self.dodge_history.append(self.dodge) 
          
          # if there are visible coins, compute the agents distance to the coins
          if coins_on_grid:
               current_nearest_coin, coin = dist_to_nearest(coins_on_grid, (x, y))
          else:
               current_nearest_coin = 1e100

          # init reward value
          reward = 0.
          
          # if coin collected -> reward
          if e.COIN_COLLECTED in self.events:
               self.nearest_coin = 1e100
               reward += 25.0
               #print("COLLECTED_COIN")
          if e.KILLED_OPPONENT in self.events:
               reward += 25.0
               #print("KILLED OPPONENT")
          # if agent's bomb destroys crates -> reward
          if e.CRATE_DESTROYED in self.events:
               reward += 5.0
               #print("CRATE_DESTROYED")
          # if the agent drops a bomb -> reward
          if e.BOMB_DROPPED in self.events:
               reward += 0.5
               #print("DROPPED BOMB")
          # if the agent drops a unnecessary bomb -> -reward
          if e.BOMB_EXPLODED in self.events and not (e.CRATE_DESTROYED in self.events or e.KILLED_OPPONENT in self.events or self.enemy_dist < ENEMY_RADIUS):
               reward -= 10.0
               #print("DIDN'T DESTROY CRATE OR ENEMY WITH BOMB")
          # if the agent performs an invalid action -> -reward
          if e.INVALID_ACTION in self.events:
               reward -= 2.0
               #print("INVALID_ACTION")
          # if he waits -> -reward
          if e.WAITED in self.events:
               reward -= 1.0
               #print("WAITED")
          # if the agents kills himself -> -reward
          if e.GOT_KILLED in self.events or e.KILLED_SELF in self.events:
               return -15.0
               #print("KILLED_SELF OR GOT KILLED")
          # +reward is agent moves to visible coin, -reward if the walks away from it
          if current_nearest_coin < self.nearest_coin and current_nearest_coin < COIN_RADIUS:
               reward += 2.0
               #print(f"Coin distance: {current_nearest_coin}")
               #print("MOVED CLOSER TO NEAREST COIN")
          elif current_nearest_coin > self.nearest_coin and current_nearest_coin < COIN_RADIUS:
               reward -= 1.0
             #  print(f"Coin distance: {current_nearest_coin}")
             #  print("MOVED AWAY FROM NEAREST COIN")
          
          # update current nearest coin
          self.nearest_coin = current_nearest_coin

          # give the agent a reward if he walks away from his dropped bomb
          if bomb_dist > self.bomb_dist and bomb_dist < BOMB_RADIUS:
               #print(f"bomb dist: {bomb_dist}")
               self.bomb_dist = bomb_dist
               reward += 1.0
               #print("RAN AWAY FROM BOMB")
          # if he drops the bomb at the starting position -> -reward
          # (it's a better strategy to go around a corner and place a bomb there, so he can find cover afterwards)
          if self.last_pos in [(1,1), (15,1), (1,15), (15,15)] and e.BOMB_DROPPED in self.events:
               reward -= 5.0
               #print("DROPPED BOMB AT START")
          if self.dodge:
               reward += 2.0
               #print("PERFORMED SMART DODGE")

          #print(f"REWARD = {reward}")
          return reward
     except:
          raise RuntimeError("Rewarding failed")


def dist_to_nearest(objs, self_pos):
     # compute distance to the nearest object given as an argument
     dists = []
     for obj in objs:
          dist = np.sqrt((obj[0] - self_pos[0])**2 + (obj[1] - self_pos[1])**2)
          dists.append(dist)

     return np.min(dists), np.argmin(dists)

def smart_dodge(bomb, dist, self_pos, last_pos):
     # check if the agent performed a smart dodge, i.e. walked around a corner
     xb, yb, x, y = bomb[0], bomb[1], self_pos[0], self_pos[1]
     if (last_pos[0] == xb) or (last_pos[1] == yb):
          if xb != x and yb != y and dist < 3:
               return True
          else:
               return False
     else:
          return False