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

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-task', help="Task to execute:\n1. Q learning on sample trajectories\n2. Q learning without pruned actions\n3. Q learning with pruned actions", metavar='1', action='store', dest='task', default="1", type=int)
parser.add_argument("-sample", metavar="1", dest='sample', default='1', help="which trajectory to evaluate (with task 1)", type=int)
parser.add_argument('-episodes', help="Number of episodes to run (with task 2 & 3)", metavar='1', action='store', dest='episodes', default="1", type=int)


class QLearning:

    def __init__(self, task, sample=1, episodes=1):
        rospy.init_node('qlearning', anonymous=True)
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
        
        self.books_json_file = root_path + "/books.json"
        self.books = json.load(open(self.books_json_file))
        self.helper = problem.Helper()
        self.helper.reset_world()

        if(task == 1):
            trajectories_json_file = root_path + "/trajectories{}.json".format(sample)
            q_values = self.task1(trajectories_json_file)
        elif(task == 2):
            q_values = self.task2(episodes)
        elif(task == 3):
            q_values = self.task3(episodes)

        with open(root_path + "/q_values.json", "w") as fout:
            json.dump(q_values, fout)


    def task3(self, episodes):
        
        q_values = {}
        
        # Your code here
        
        return q_values

    def task2(self, episodes):
        
        q_values = {}

        # Your code here   
        
        return q_values

    def task1(self, trajectories_json_file):

        q_values = {}

        # Your code here   
        
        return q_values  

if __name__ == "__main__":

    args = parser.parse_args()

    if args.task == 1:
        QLearning(args.task, sample=args.sample)
    elif args.task == 2 or args.task == 3:
        QLearning(args.task, episodes=args.episodes)
