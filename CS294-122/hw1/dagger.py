#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from clone_behaviour import split_dataset
from keras.models import load_model
import keras.backend.tensorflow_backend as K

def dagger_iter(expert_policy_fn, dagger_policy_fn, env, args):
    returns = []
    observations = []
    dagger_actions = []
    expert_actions = []
    for i in range(args.num_rollouts):
        print('Rollout iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            expert_action = expert_policy_fn(obs[None,:])
            dagger_action = dagger_policy_fn(obs[None, :])
            observations.append(obs)
            expert_actions.append(expert_action)
            dagger_actions.append(dagger_action)
            obs, r, done, _ = env.step(dagger_action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            # if steps % 100 == 0: print("%i/%i"%(steps, args.max_timesteps))
            if steps >= args.max_timesteps:
                break
        returns.append(totalr)

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(expert_actions)}

    return expert_data, returns

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('clone_model_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    expert_policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    print('loading and building dagger model')
    dagger_model = load_model(args.clone_model_file)
    print('loaded and built')

    with K.get_session() as session:
        import gym
        env = gym.make(args.envname)
        args.max_timesteps = args.max_timesteps or env.spec.timestep_limit

        returns_history = []
        dagger_iters = 10
        for i in range(dagger_iters):
            print("DAgger iter", i)
            dagger_policy_fn = dagger_model.predict
            expert_data, returns = dagger_iter(expert_policy_fn, dagger_policy_fn, env, args)

            returns_history.append(returns)

            X, Y = expert_data['observations'], expert_data['actions']

            N, D = X.shape
            A = Y.shape[2]

            # Reshape Y from (N, 1, A) to (N, A)
            Y = Y.reshape(N, A)

            trainX, trainY, valX, valY, testX, testY = split_dataset(X, Y, 80, 20)

            dagger_model.fit(trainX, trainY, epochs=100, batch_size=1000, verbose=2, validation_data=(valX, valY))

        clone_url = "dagger_clones/" + args.expert_policy_file.split("/")[-1].split(".")[0] + ".h5"
        dagger_model.save(clone_url)

    for returns in returns_history:
        # print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

    # To avoid https://github.com/tensorflow/tensorflow/issues/3388
    # Doesn't work all the time
    del session
    import gc
    gc.collect()


if __name__ == '__main__':
    main()
