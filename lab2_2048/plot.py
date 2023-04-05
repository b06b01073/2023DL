import matplotlib.pyplot as plt
import numpy as np

with open("result.txt") as f:
    scores = np.array([int(score) for score in f.readlines()])
    averages = []
    avg = 0

    for i in range(len(scores)):
        avg = avg * (i / (i + 1)) + scores[i] / (i + 1)
        averages.append(avg)

    plt.ylabel('score(mean)')
    plt.xlabel('episode')
    plt.plot(averages)
    plt.savefig('result.png')
