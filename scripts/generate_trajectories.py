#!/usr/bin/env python
# encoding: utf-8

import heapq
import problem as api
import rospy
from std_msgs.msg import String
import numpy as np
import random
import json
import os

class TrajectoriesGenerator:

    def __init__(self):
        self.root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
        self.books_json_file = self.root_path + r"/books.json"
        f = open(self.books_json_file)
        self.books = json.load(f)
        f.close()

        self.current_state = api.get_current_state()
        print("Init State:")
        print(self.current_state)

        self.generate_trajectories()


    def get_all_possible_actions(self):
        actions = ["normal_moveF", "normal_TurnCW", "normal_TurnCCW", "careful_moveF", "careful_TurnCW", "careful_TurnCCW"]
        for book_name in self.books["books"]:
            actions.append("normal_pick {}".format(book_name))
            actions.append("careful_pick {}".format(book_name))
            for bin_name in self.books["bins"]:
                actions.append("normal_place {} {}".format(book_name, bin_name))
                actions.append("careful_place {} {}".format(book_name, bin_name))
        return actions


    def generate_trajectories(self):

        trajectory = []

        trajectories_output_file = self.root_path + '/trajectories3.json'
        f = open(self.root_path + '/action_sequence.txt', 'r')
        action_list = f.readlines()
        f.close()

        for chosen_action in action_list:
            
            # possible_actions = self.get_all_possible_actions()
            # idx = random.randint(0, len(possible_actions) - 1)
            # chosen_action = possible_actions[idx]

            chosen_action = chosen_action.strip()
            action_str = chosen_action
            if "pick" in chosen_action:
                chosen_action, book_name = chosen_action.split(' ')
                action_params = {"book_name": book_name}
            elif "place" in chosen_action:
                chosen_action, book_name, bin_name = chosen_action.split(' ')
                action_params = {"book_name": book_name, "bin_name": bin_name}
            else:
                action_params = {}

            print("Executing action: {} with params: {}".format(chosen_action, action_params))

            success, next_state = api.execute_action(chosen_action, action_params)
            if success == 1:
                print("Successfully executed")
                print("")
            else:
                print("Action failed")

            reward = api.get_reward(self.current_state, chosen_action, next_state)
            print("Reward: ", reward)
            
            trajectory_element = { 
                                    "state": json.dumps(self.current_state),
                                    "action": action_str,
                                    "reward": reward
                                }
            trajectory.append(trajectory_element)

            self.current_state = next_state
            print("updated current state:")
            print(self.current_state)

            # raw_input("\nPress Enter to continue execution...")

        if api.is_terminal_state(self.current_state):
            print("Goal Reached!")
        else:
            print("Goal not reached!")

        with open(trajectories_output_file, 'w') as fout:
            json.dump(trajectory, fout)

if __name__ == "__main__":
    trajectoriesGenerator = TrajectoriesGenerator()