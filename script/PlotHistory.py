import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description = "Plot")
parser.add_argument('--filename',           type=str, help='pt file path')
params = parser.parse_args()

def main():
    filename = params.filename
    episodes = []
    rewards = []
    losses = []
    i = 1
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            r, l = line.split(' ')
            episodes.append( 16*i)
            i = i + 1
            rewards.append(float(r))
            losses.append(float(l))

    fig = plt.figure()
    ax = fig.add_subplot(211)
    plt.plot(episodes, rewards)
    plt.ylabel('Reward')
    plt.xlabel('steps')
    
    ax = fig.add_subplot(212)
    plt.plot(episodes, losses)
    plt.ylabel('Loss')
    plt.xlabel('steps')
    plt.show()

if __name__ == '__main__':
    main()