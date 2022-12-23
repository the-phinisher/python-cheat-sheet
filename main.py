# Libraries
import gym
import numpy as np

# The data file
dtfile = 'data.npy'

# Initializing environment & constants
env = gym.make('CartPole-v1')
leos = len(env.observation_space.sample())

# The evaluation function
def think(obs, wts):
    return 0 if np.dot(obs, wts) < 0 else 1

# All settings
render = False
ntests = 2
genwt = lambda : 2 * np.random.random([leos]) - 1
nwts = 1000
detaillog = False
dsprog = True
progstp = 5
tstwdw = 1000

# computed
dn = nwts * progstp // 100

# Best portfolio
bstprf = [None, -1, np.inf]

# For long loops
try:
    # Checking through n different weights
    for i in range(nwts):

        # Getting different evaluatory scores
        tscore = 0
        tdelta = 0

        #Generate weights for this attempt
        wts = genwt()
        if(detaillog): print(wts)

        # Running n times for inclusion of possible cases
        for aaa in range(ntests):
            #scores for each test
            score = 0
            delta = 0

            # One time run code
            obs = env.reset()
            done = False

            for lpk in range(tstwdw):
                if render: env.render()
                obs, _, done, _ = env.step(think(obs, wts))

                # Scores updation
                score += 1
                delta += np.dot(obs, obs) * score
            if(detaillog): print('\t', score, delta, sep='\t')

            # Updating total scores
            tscore += score
            tdelta += delta
        if(detaillog): print('\t', tscore, '\t', tdelta)

        # Percentage progress
        if(dsprog):
            n = i + 1
            if n % dn == 0:
                print(str(n * progstp // dn) + '% Complete')


        # Portfolio : wts, tscore, tdelta
        # Comparing with best portfolio
        if (tscore > bstprf[1]) or (tscore == bstprf[1] and tdelta < bstprf[2]): bstprf = [wts, tscore, tdelta]

    print('Best portfolio:')
    print(bstprf)

    if (str(input('Demonstrate best portfolio?(y/N)'))) != 'y':
        env.close()
        exit()
    print('Warning: Demonstration is user terminated (KeyboardInterrupt)')
    obs = env.reset()
    wts = bstprf[0]
    while True:
        env.render()
        obs, _, _, _ = env.step(think(obs, wts))

finally:
    print(bstprf[0])
    with open(dtfile, 'wb') as file:
        np.save(file, bstprf[0])
    with open(dtfile, 'rb') as file:
        print(np.load(file))
    env.close()
    exit()