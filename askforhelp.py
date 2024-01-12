import os
import random
import glob

import pandas as pd
import numpy as np
import sympy as sp
from logging import getLogger
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from .classical_algorithm import DILOGS_IDS, reduce_polylog_expression, construct_polylog_expression, bfs,\
    import_eqs_set, smarter_bfs, beam_search_network
 

def evaluate_model(model, env, initial_start_set=None, deterministic=True, num_runs=1, verbose=False,
                   random_agent=False, non_solved_set=None, distrib_info=False):
    """
    Detailed evaluation run that outputs all of the relevant statistics.
    This runs by only selecting the best action of the model
    :param model: Model to test
    :param env: PolyLog environment
    :param initial_start_set: Set of equations that we should test on
    :param deterministic: Whether the policy is run in a deterministic fashion
    :param num_runs: If we run stochastically we can do it multiple times
    :param verbose:
    :param random_agent: If we want to swap out for a random agent
    :param non_solved_set: If we want to check on a subset of equations
    :param distrib_info: If we want the breakdown by input complexity (e.g number of scrambles)
    :return:
    """
    if initial_start_set is not None:

        # Load the set and initialize the stats
        scr_expr, simple_expr, eq_info_complex = import_eqs_set(initial_start_set, add_info=distrib_info)
        count_poly_train = []
        count_words_train = []
        count_poly_scr = []
        count_words_scr = []
        poly_extra_train = []
        words_extra_train = []
        reward_tots_train = []
        steps_taken = []
        steps_taken_non_cyclic = []
        steps_taken_distrib = []
        steps_taken_solved = []
        steps_taken_solved_non_cyclic = []
        steps_taken_solved_distrib = []
        num_eqs_consider = len(scr_expr)
        solved_num_poly = 0
        solved_num_words = 0
        num_cyclic_traj = 0
        non_solved_resolved = 0
        simple_len = {'0': 0, '1': 0, '2': 0, '3': 0}
        solved_simple_len = {'0': 0, '1': 0, '2': 0, '3': 0}
        solved_simple_len_min = {'0': 0, '1': 0, '2': 0, '3': 0}

        if distrib_info:
            unique_label = np.unique(np.array(eq_info_complex), axis=0)
            dict_valid_count = {'s' + str(label[0]) + 't' + str(label[1]): 0 for label in unique_label}
            dict_valid_count_min = {'s' + str(label[0]) + 't' + str(label[1]): 0 for label in unique_label}
            dict_steps_taken = {'s' + str(label[0]) + 't' + str(label[1]): 0 for label in unique_label}
            dict_eq_done = {'s' + str(label[0]) + 't' + str(label[1]): 0 for label in unique_label}
            dict_nodes = {'s' + str(label[0]) + 't' + str(label[1]): 0 for label in unique_label}

        if non_solved_set is not None:
            with open(non_solved_set, 'r') as fread:
                indicesstr = fread.readlines()
                indices = [int(ind) for ind in indicesstr]
            print('We have {} equations not solved by the classical algorithm'.format(len(indices)))

        # Loop on all the equations
        for i, scr_eq in enumerate(scr_expr):

            # Sanity check to see if all equations fit the model
            if len(env.obs_rep.sympy_to_prefix(scr_eq)) > env.obs_rep.dim:
                num_eqs_consider -= 1
                if distrib_info:
                    print('Equation:  s' + str(eq_info_complex[i][0]) + 't' + str(eq_info_complex[i][1]) +
                          ' does not fit')
                continue
            env.init_eq = scr_eq

            reward_tot = 0
            steps_done = 0
            cyclic_traj = False
            steps_done_non_cyclic = 0
            steps_done_distrib = {'0': 0, "1": 0, '2': 0, '3': 0}
            done = False
            obs = env.reset()
            min_poly = env.obs_rep.count_number_polylogs()
            if not verbose:
                if i % 10 == 0:
                    print('Doing expression {}'.format(str(i)))
            else:
                print('Doing expression {}'.format(str(i)))
                if non_solved_set is not None and i in indices:
                    print("Equation was not solved by the classical algorithm")

            # Do the episode
            while not done:
                # Random agent chooses randomly on the policy
                if random_agent:
                    action = random.randint(0, len(env.action_rep.actions)-1)
                    action_distr = model.policy.get_distribution(
                        model.policy.obs_to_tensor(obs)[0]).distribution.probs.detach().numpy()[0]

                else:
                    # Get the action distribution and apply the best one
                    action, _states = model.predict(obs, deterministic=deterministic)
                    action_distr = model.policy.get_distribution(
                        model.policy.obs_to_tensor(obs)[0]).distribution.probs.detach().numpy()[0]

                steps_done += 1
                steps_done_distrib[str(action)] += 1

                # Count the number of "active" actions
                if action != 2:
                    steps_done_non_cyclic += 1
                    if distrib_info:
                        dict_nodes['s' + str(eq_info_complex[i][0]) + 't' + str(eq_info_complex[i][1])] += 1
                if verbose:
                    print('Action taken is : {}'.format(env.action_rep.id_action_map()[str(action)]))
                    print('Distributions were {}'.format(str(str(action_distr))))

                obs, rewards, dones, info = env.step(action)

                # Check if the expression is smaller
                poly_num = env.obs_rep.count_number_polylogs()
                if poly_num < min_poly:
                    min_poly = poly_num

                # Check for cyclic trajectories
                if rewards < 0:
                    cyclic_traj = True
                reward_tot += rewards
                if verbose:
                    print('Total reward is {}'.format(reward_tot))
                    env.render()
                    print('Arguments are ordered as ')
                    print(env.obs_rep.sp_state.args)
                    print('\n')
                done = dones

            # Keep in memory the data for the episode run
            steps_taken.append(steps_done)
            steps_taken_non_cyclic.append(steps_done_non_cyclic)
            steps_taken_distrib.append(steps_done_distrib)
            count_poly_train.append(env.obs_rep.count_number_polylogs())
            count_words_train.append(env.obs_rep.get_length_expr())
            extra_poly = env.obs_rep.count_number_polylogs() - simple_expr[i].count(sp.polylog)
            poly_extra_train.append(extra_poly)
            count_poly_scr.append(scr_eq.count(sp.polylog))
            extra_words = env.obs_rep.get_length_expr() - len(env.obs_rep.sympy_to_prefix(simple_expr[i]))
            words_extra_train.append(extra_words)
            count_words_scr.append(len(env.obs_rep.sympy_to_prefix(scr_eq)))
            reward_tots_train.append(reward_tot)

            simple_len[str(simple_expr[i].count(sp.polylog))] += 1

            num_cyclic_traj += 1 if cyclic_traj else 0

            # If we solved the equation we take more statistics
            if extra_poly == 0:
                solved_num_poly += 1
                steps_taken_solved.append(steps_done)
                steps_taken_solved_non_cyclic.append(steps_done_non_cyclic)
                steps_taken_solved_distrib.append(steps_done_distrib)
                solved_simple_len[str(simple_expr[i].count(sp.polylog))] += 1
                if distrib_info:
                    dict_valid_count['s' + str(eq_info_complex[i][0]) + 't' + str(eq_info_complex[i][1])] += 1
                    dict_steps_taken['s' + str(eq_info_complex[i][0]) + 't' + str(eq_info_complex[i][1])] += steps_done_non_cyclic
                if non_solved_set is not None and i in indices:
                    print('Could solve Eq.{} not solved by classical algorithm'.format(i))
                    non_solved_resolved += 1
   
            # If we solved the equation at some point in the trajectory (relevant if end point is not 0)
            if min_poly <= simple_expr[i].count(sp.polylog):
                solved_simple_len_min[str(simple_expr[i].count(sp.polylog))] += 1
                if distrib_info:
                    dict_valid_count_min['s' + str(eq_info_complex[i][0]) + 't' + str(eq_info_complex[i][1])] += 1

            if extra_words <= 0:
                solved_num_words += 1

            if distrib_info:
                dict_eq_done['s' + str(eq_info_complex[i][0]) + 't' + str(eq_info_complex[i][1])] += 1

        # Make sure the stats account for any discarded input that is too long
        for key, value in solved_simple_len.items():
            if value > 0:
                solved_simple_len[key] = round(value / simple_len[key], 2)

        for key, value in solved_simple_len_min.items():
            if value > 0:
                solved_simple_len_min[key] = round(value / simple_len[key], 2)

        # Also normalize the distribution values
        if distrib_info:
            for key, value in dict_steps_taken.items():
                dict_steps_taken[key] = round(value/dict_valid_count[key], 2)
            for key, value in dict_valid_count.items():
                dict_valid_count[key] = round(value*100 / dict_eq_done[key], 2)
            for key, value in dict_valid_count_min.items():
                dict_valid_count_min[key] = round(value*100 / dict_eq_done[key], 2)
            for key, value in dict_nodes.items():
                dict_nodes[key] = value / dict_eq_done[key]

        # Output the entire statistics
        print('Mean Total reward:', np.array(reward_tots_train).mean())
        print('Mean Total Steps:', np.array(steps_taken).mean())
        print('Mean Total Non cyclic Steps:', np.array(steps_taken_non_cyclic).mean())
        print('Action distribution:',
              {key: sum(distr[key] for distr in steps_taken_distrib) for key in steps_taken_distrib[0].keys()})
        print('Mean Total Steps for solved:', np.array(steps_taken_solved).mean())
        print('Mean Total Non cyclic Steps for solved:', np.array(steps_taken_solved_non_cyclic).mean())
        print('Action distribution for solved:',
              {key: sum(distr[key] for distr in steps_taken_solved_distrib) for key in
               steps_taken_solved_distrib[0].keys()})
        print('Mean final number of polylogs:', np.array(count_poly_train).mean())
        print('Mean initial number of polylogs :', np.array(count_poly_scr).mean())
        print('Mean final number of polylogs extra:', np.array(poly_extra_train).mean())
        print('Mean final length of expression:', np.array(count_words_train).mean())
        print('Mean initial length of expression:', np.array(count_words_scr).mean())
        print('Mean final length of expression extra:', np.array(words_extra_train).mean())
        print('Found {} cyclic trajectories. So for {} % of the expressions:'.format(num_cyclic_traj, str(int(
            100 * num_cyclic_traj / num_eqs_consider))))
        print('Distribution for solved by initial length: {}'.format(solved_simple_len))
        print('Distribution for solved in trajectory by initial length: {}'.format(solved_simple_len_min))
        if distrib_info:
            print('Distribution for solved: {}'.format(dict_valid_count))
            print('Distribution for solved in trajectory: {}'.format(dict_valid_count_min))
            print('Distribution for solved (non cyclic) steps: {}'.format(dict_steps_taken))
            print('Distribution for nodes visited: {}'.format(dict_nodes))
        print('Solved expressions, num polylogs {} %'.format(str(int(100*solved_num_poly/num_eqs_consider))))
        print('Solved expressions, num words {} %'.format(str(int(100*solved_num_words / num_eqs_consider))))
        if non_solved_set is not None:
            print('Solved expressions not done by classical: {} %'.format(str(int(100*non_solved_resolved / len(indices)))))

    # If we don't start with a set to check then we just try to solve the current environment
    else:

        print('We start with')
        env.render()
        print('Arguments are ordered as ')
        print(env.obs_rep.sp_state.args)
        print('\n')

        count_poly = []
        count_words = []
        reward_tots = []
        resolved = 0

        for run in range(num_runs):
            reward_tot = 0
            done = False
            obs = env.reset()

            while not done:
                if random_agent:
                    action = random.randint(0, len(env.action_rep.actions)-1)
                else:
                    action, _states = model.predict(obs, deterministic=deterministic)
                if verbose:
                    print('Action taken is : {}'.format(env.action_rep.id_action_map()[str(action)]))
                    print('Distributions were {}'.format(str(str(model.policy.get_distribution(
                        model.policy.obs_to_tensor(obs)[0]).distribution.probs.detach().numpy()[0]))))
                obs, rewards, dones, info = env.step(action)
                reward_tot += rewards
                if verbose:
                    env.render()
                    print('Arguments are ordered as ')
                    print(env.obs_rep.sp_state.args)
                    print('\n')
                done = dones

            if env.obs_rep.count_number_polylogs() == 0:
                resolved += 1

            count_poly.append(env.obs_rep.count_number_polylogs())
            count_words.append(env.obs_rep.get_length_expr())
            reward_tots.append(reward_tot)

        print('Over {} run(s) we have '.format(num_runs))
        print('Mean Total reward:', np.array(reward_tots).mean())
        print('Mean final number of polylogs:', np.array(count_poly).mean())
        print('Mean final length of expression:', np.array(count_words).mean())
        print('Mean Resolution to 0 : {} %'.format(str(int(100*(resolved/num_runs)))))
