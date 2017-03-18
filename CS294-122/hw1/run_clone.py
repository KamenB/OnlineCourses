#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""
import numpy as np
import gym
from keras.models import load_model
import tensorflow as tf
import tf_util
import keras.backend.tensorflow_backend as K
import time

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('clone_model_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building clone model')
    model = load_model(args.clone_model_file)
    policy_fn = model.predict
    print('loaded and built')

    with K.get_session() as session:
        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                # mean, std = 0.353611432088, 2.70487545573
                # obs = (obs - mean) / std
                action = policy_fn(obs[None, :])
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
                time.sleep(0.01)
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

    # To avoid https://github.com/tensorflow/tensorflow/issues/3388
    # Doesn't work all the time
    del session
    import gc
    gc.collect()

if __name__ == '__main__':
    main()
