#!/usr/bin/env python
# encoding: utf-8

__copyright__ = "Copyright 2019, AAIR Lab, ASU"
__authors__ = ["Abhyudaya Srinet"]
__credits__ = ["Siddharth Srivastava"]
__license__ = "MIT"
__version__ = "1.0"
__maintainers__ = ["Pulkit Verma", "Abhyudaya Srinet"]
__contact__ = "aair.lab@asu.edu"
__docformat__ = 'reStructuredText'

import rospy
from std_msgs.msg import String
import problem
import json
import os
import argparse
import numpy as np
from copy import copy
import random as random
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-task', help="Task to execute:\n1. Q learning on sample trajectories\n2. Q learning without pruned actions\n3. Q learning with pruned actions", metavar='1', action='store', dest='task', default="1", type=int)
parser.add_argument("-sample", metavar="1", dest='sample', default='1', help="which trajectory to evaluate (with task 1)", type=int)
parser.add_argument('-episodes', help="Number of episodes to run (with task 2 & 3)", metavar='1', action='store', dest='episodes', default="1", type=int)
parser.add_argument('-headless', help='1 when running in the headless mode, 0 when running with gazebo', metavar='1', action='store', dest='headless', default=1, type=int)


class QLearning:

    def __init__(self, task, headless=1, sample=1, episodes=1):
        rospy.init_node('qlearning', anonymous=True)
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
        
        self.books_json_file = root_path + "/books.json"
        self.books = json.load(open(self.books_json_file))
        self.helper = problem.Helper()
        self.helper.reset_world()
        self.headless = headless
        self.alpha = 0.3
        self.gamma = 0.9

        if(task == 1):
            trajectories_json_file = root_path + "/trajectories{}.json".format(sample)
            q_values = self.task1(trajectories_json_file)
        elif(task == 2):
            q_values = self.task2(episodes)
        elif(task == 3):
            q_values = self.task3(episodes ,self.books)

        with open(root_path + "/q_values.json", "w") as fout:
            json.dump(q_values, fout)



    def task3(self, episodes , books):
    
        def prune_actionlist(state1 , book ):
            all_action2 = helper.get_all_actions()
            location_robot = [state1["robot"]["x"] , state1["robot"]["y"]]
              # Please put the path of books.json
            objdict=book
            book_name= objdict["books"]
            book_1d = book_name["book_1"]
            book_1_load_location= book_1d["load_loc"]
            book_1_load_location_1 = book_1_load_location[0]
            book_1_load_location_2 = book_1_load_location[1]
            book_2d = book_name["book_2"]
            book_2_load_location=book_2d["load_loc"]
            book_2_load_location_1 = book_2_load_location[0]
            book_2_load_location_2 = book_2_load_location[1]
            bin_name= objdict["bins"]
            bin_1d = bin_name["trolly_1"]
            bin_1_load_location= bin_1d["load_loc"]
            bin_2d = bin_name["trolly_2"]
            bin_2_load_location= bin_2d["load_loc"]
            if ((location_robot != book_1_load_location_1) and (location_robot != book_1_load_location_2) and  (location_robot != book_2_load_location_1) and (location_robot != book_2_load_location_2) ) :
                for elem in all_action2 :
                    if (elem[:12] == "careful_pick" or elem[:11] == "normal_pick"):
                        all_action2.remove(elem)
            if ( (location_robot != bin_1_load_location[0]) and (location_robot != bin_1_load_location[1]) and (location_robot != bin_1_load_location[2]) and (location_robot != bin_1_load_location[3]) and (location_robot != bin_1_load_location[4]) and (location_robot != bin_1_load_location[5]) and (location_robot != bin_1_load_location[6]) and (location_robot != bin_1_load_location[7]) and (location_robot != bin_2_load_location[0]) and (location_robot != bin_1_load_location[1]) and (location_robot != bin_1_load_location[2]) and (location_robot != bin_1_load_location[3]) and (location_robot != bin_1_load_location[4]) and (location_robot != bin_1_load_location[5]) and (location_robot != bin_1_load_location[6]) and (location_robot != bin_1_load_location[7]) ) :
                for elem in all_action2 :
                    if (elem[:13] == "careful_place" or elem[:12]=="normal_place"):
                        all_action2.remove(elem)

            pruned_action_list = all_action2 
            return pruned_action_list
        
        helper=problem.Helper()
        init_state1 = helper.get_current_state()
        init_state=json.dumps(init_state1)
        all_actions = helper.get_all_actions()
        action_qpair= {}
        pruned_list = prune_actionlist(init_state1 , books )    # pruning the action list with the function prune_actionlist()
        for action in pruned_list :
            action_qpair[action] = 0
        state_dict ={}
        state_dict[init_state]=action_qpair
        state = init_state   # state is a string
        stated = init_state1 # stated is a dictionary
        reward_list = []
        iteration_list = []
        
        for i in range(episodes):
            state = init_state
            stated= init_state1
            reward_prev = 0
            count=0
            step=1
            while(count==0):
                if(helper.is_terminal_state(stated)):
                    count=1
                epsilon=max(0.05, 0.7 - (0.05*i))
                x = state_dict[state].items()
                state_action=sorted(x,key=lambda t:t[1])[-1][0] #The action that the agent is supposed to perform in a given state
                pruned_list1 = prune_actionlist(stated , books )
                random_action=random.choice(pruned_list1)
                if (random.random() < epsilon ) :
                    action_to_perform = random_action
                else:
                    action_to_perform = state_action          # action_to_perform is the full action with the objects
                parsed_action_1=action_to_perform.split(" ")
                parsed_action=parsed_action_1[1:]
                param_cont = {}
                for l in parsed_action :
                    if l[:4]=="book":
                        param_cont["book_name"] = l
                    else:
                        param_cont["bin_name"] = l
                success , next_state1 = helper.execute_action(parsed_action_1[0] , param_cont) # next_state1 is a dictionary
                next_state = json.dumps(next_state1) # next_state is a string
                state_list = state_dict.keys()

                label=0
                for statename in state_list:
                    if statename == next_state :  # trying to check if the next_state already exists as an entry in statename
                        label=1
                        break
                if label==0:                       # if the next_state doesnot exist then the next state is iniatialized with zero qvalues  for each of the actions.
                    action_qpair={}
                    pruned_list2 = prune_actionlist (next_state1 , books) # pruning the action list with the function prune_actionlist()
                    for action in pruned_list2 :
                        action_qpair[action] = 0
                        state_dict[next_state]=action_qpair
                reward = helper.get_reward(stated,parsed_action_1[0],next_state1) 
                reward_cum = reward_prev + ((self.gamma**step)*reward)        # Calculating cumulative reward
                exploring_state_dict = state_dict[state]
                q_old = exploring_state_dict[action_to_perform]
                q= ((1 - self.alpha) * q_old) + self.alpha*(reward + self.gamma * max(state_dict[next_state].values()))  # Calculating the q_value for being in a state and taking a specific action
                exploring_state_dict[action_to_perform]=q    # Plugging the q-value into the action q value dictionary
                state_dict[state]=exploring_state_dict
                state = next_state     # state , next_state is a string 
                stated = next_state1   # stated is a dictionary
                reward_prev = reward_cum
                step = step+1
            print str(i)+","+str(reward_cum)
            reward_list = reward_list + [reward_cum]
            iteration_list = iteration_list + [i]
            helper.reset_world()
        q_values={}
        q_values=state_dict
        #plt.plot(iteration_list)
        #plt.ylabel(reward_list)
        #plt.show()
        return q_values
        



    def task2(self, episodes):
        helper=problem.Helper()
        init_state1 = helper.get_current_state()
        init_state=json.dumps(init_state1)
        all_actions = helper.get_all_actions()
        action_qpair= {}
        for action in all_actions :
            action_qpair[action] = 0
        state_dict ={}
        state_dict[init_state]=action_qpair
        state = init_state   # state is a string
        stated = init_state1 # stated is a dictionary
        reward_list = []
        iteration_list = []
        
        for i in range(episodes):
            state = init_state 
            stated= init_state1
            reward_prev = 0
            count=0
            step=1
            while(count==0):
                if(helper.is_terminal_state(stated)):
                    count=1
                epsilon=max(0.05, 0.7 - (0.05*(i+1)))
                x = state_dict[state].items()
                state_action=sorted(x,key=lambda t:t[1])[-1][0] #The optimal action that the agent is supposed to perform in a given state
                random_action=random.choice(all_actions)
                if (random.random() < epsilon ) :
                    action_to_perform = random_action
                else:
                    action_to_perform = state_action          # action_to_perform is the full action with the objects
                parsed_action_1=action_to_perform.split(" ")
                parsed_action=parsed_action_1[1:]
                param_cont = {}
                for l in parsed_action :
                    if l[:4]=="book":
                        param_cont["book_name"] = l
                    else:
                        param_cont["bin_name"] = l
                success , next_state1 = helper.execute_action(parsed_action_1[0] , param_cont) # next_state1 is a dictionary
                next_state = json.dumps(next_state1) # next_state is a string
                state_list = state_dict.keys()

                label=0
                for statename in state_list:
                    if statename == next_state :  # trying to check if the next_state already exists as an entry in statename
                        label=1
                        break
                if label==0:                       # if the next_state doesnot exist then the next state is iniatialized with zero qvalues for each of the actions.
                    action_qpair={}
                    for action in all_actions :
                        action_qpair[action] = 0
                        state_dict[next_state]=action_qpair
                reward = helper.get_reward(stated,parsed_action_1[0],next_state1) 
                reward_cum = reward_prev + ((self.gamma**step)*reward)        # Calculating cumulative reward
                exploring_state_dict = state_dict[state]
                q_old = exploring_state_dict[action_to_perform]
                q= ((1 - self.alpha) * q_old) + (self.alpha*(reward + self.gamma * max(state_dict[next_state].values())))  # Calculating the q_value for being in a state and taking a specific action
                exploring_state_dict[action_to_perform]=q    # Plugging the q-value into the action q value dictionary
                state_dict[state]=exploring_state_dict
                state = next_state     # state , next_state is a string 
                stated = next_state1   # stated is a dictionary
                reward_prev = reward_cum
                step = step+1
            print str(i)+","+str(reward_cum)    
            reward_list = reward_list + [reward_cum]
            iteration_list = iteration_list + [i]
            helper.reset_world()

        q_values={}
        q_values=state_dict
        #plt.plot(iteration_list)
        #plt.ylabel(reward_list)
        #plt.show()
        return q_values




    def task1(self, trajectories_json_file):
        helper=problem.Helper()
        q_values = {}
        all_actions=helper.get_all_actions()
        trajectory1=open(trajectories_json_file) #location of the trajectory1 file
        List1=json.load(trajectory1)
        lastelem= copy(List1[-1])
        lastelem["reward"]=0
        for entry in List1 :
            action=entry["action"]
            state= entry["state"]
            reward= entry["reward"]
            temp ={}
            for action1 in all_actions:
                temp[action1] = 0
            q_values[state]=temp
        
        for (entry,next_entry) in zip(List1, List1[1:]+[lastelem]) :
            action=entry["action"]
            state=entry["state"]
            reward=entry["reward"]
            next_state=next_entry["state"]
            temp = {}
            q_valuenext = {}
            q_valuenext = q_values[next_state]
            newstate={}
            newstate=q_values[state]
            for action1 in all_actions :
                if action1 == action :
                    q = self.alpha*(reward + self.gamma* (max(q_valuenext.values())))
                    newstate[action1]= q
            q_values[state]= newstate
        print (json.dumps(q_values, sort_keys=True , indent = 6))
        return q_values  

if __name__ == "__main__":

    args = parser.parse_args()

    if args.task == 1:
        QLearning(args.task, headless=args.headless, sample=args.sample)
    elif args.task == 2 or args.task == 3:
        QLearning(args.task, headless=args.headless, episodes=args.episodes)
