import logging
import numpy as np
import multiprocessing as mp
import os
import random
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import train_env as env
import a3c
import load_trace


S_INFO = 8  # bit_rate, buffer_size, transmissed chunk size, transmission delay, next_chunk_size, mask, downloaded low-quality chunk number, remain chunk number
S_LEN = 11
A_DIM = 11
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [250, 500, 650, 900, 1150 ,1350, 1750, 2000, 2350, 3100, 3900]
HD_REWARD = [1, 1.5, 2, 3, 4, 5, 6, 11, 13, 15, 18]
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
M_IN_B = 1024 * 1024
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # Initial video quality
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES_SAT = './cooked_traces/sat/'
TRAIN_TRACES_USER = './cooked_traces/user/'
NN_MODEL = None


def action_to_bitrate(action, mask, a_dim=A_DIM):
    assert len(mask) == a_dim
    assert action >= 0
    assert action < a_dim
    assert mask[action] == 1
    return np.sum(mask[:action])


def bitrate_to_action(bitrate, mask, a_dim=A_DIM):
    assert len(mask) == a_dim
    assert bitrate >= 0
    assert bitrate < np.sum(mask)
    cumsum_mask = np.cumsum(mask) - 1
    action = np.where(cumsum_mask == bitrate)[0][0]
    return action


def testing(epoch, nn_model, log_file):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    os.system('mkdir ' + TEST_LOG_FOLDER)
    
    # run test script
    os.system('python test.py ' + nn_model)
    
    # append test performance to the log
    rewards = []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward = []
        with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    if parse[-1] != b'!':
                        reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.sum(reward[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    fileinput = 'Epoch: ' + str(epoch) + '\t' + 'Rewards_mean: ' + str(rewards_mean) + '\t' + 'Rewards_median: ' + str(rewards_median) + '\t' + \
                'Rewards_min: ' + str(rewards_min) + '\t' + 'Rewards_5per: ' + str(rewards_5per) + '\t' + \
                'Rewards_95per: ' + str(rewards_95per) + '\t' + 'Rewards_max: ' + str(rewards_max) + '\n'
    log_file.write(fileinput.encode())
    log_file.flush()


def central_agent(net_params_queues, exp_queues):
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central', filemode='w', level=logging.INFO)

    with tf.Session() as sess, open(LOG_FILE + '_test', 'wb') as test_log_file:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver(max_to_keep = 1000)  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        epoch = 0

        # assemble experiences from agents, compute the gradients
        while epoch <= 50000:
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0 

            # assemble experiences from the agents
            actor_gradient_batch = []
            critic_gradient_batch = []

            for i in range(NUM_AGENTS):
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()

                #print(np.array(s_batch).shape, np.array(a_batch).shape, np.array(r_batch).shape)
                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)

                for i in range(len(actor_gradient)):
                    assert np.any(np.isnan(actor_gradient[i])) == False

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])

            # compute aggregated gradient
            assert NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)

            for i in range(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])
                critic.apply_gradients(critic_gradient_batch[i])

            # log training information
            epoch += 1
            avg_reward = total_reward  / total_agents
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len

            logging.info('Epoch: ' + str(epoch) +
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward: ' + str(avg_reward) +
                         ' Avg_entropy: ' + str(avg_entropy))

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: avg_entropy
            })

            writer.add_summary(summary_str, epoch)
            writer.flush()

            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt")
                logging.info("Model saved in file: " + save_path)
                testing(epoch, SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt", test_log_file)


def agent(agent_id, all_cooked_sat_time, all_cooked_sat_bw, all_cooked_user_time, all_cooked_user_bw, net_params_queue, exp_queue):

    net_env = env.Environment(all_cooked_user_time=all_cooked_user_time,
                              all_cooked_user_bw=all_cooked_user_bw,
                              all_cooked_sat_time=all_cooked_sat_time,
                              all_cooked_sat_bw=all_cooked_sat_bw,
                              random_seed=agent_id)

    with tf.Session() as sess, open(LOG_FILE + '_agent_' + str(agent_id), 'wb') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        mask = net_env.video_masks[net_env.video_idx]
        video_chunk_num = net_env.video_chunk_num

        bit_rate = DEFAULT_QUALITY

        action = bitrate_to_action(bit_rate, mask)
        last_action = action
        end_of_video = True

        action_vec = np.zeros(np.sum(mask))
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        time_stamp = 0
        quality_statistics = [0, 0, 0, 0]

        while True:  # experience video streaming forever
            if end_of_video:
                delay, user_delay, buffer_size, rebuf,  video_chunk_size_sr, \
                next_video_chunk_sizes, end_of_video, video_chunk_remain, mec_video_remain, \
                reward_set, quality_statistics, quality, mask = \
                    net_env.early_stage_trans(bit_rate)
                r_batch.extend(reward_set)
                sleep_time = 0

                bit_rate = last_bit_rate = quality
                action = last_action = bitrate_to_action(bit_rate, mask)

                line = ''
                for r in reward_set:
                    line = line + 'Reward: ' + str(r) + '\n'
                log_file.write(line.encode())
                log_file.flush()

                line = 'Early-Stage:' + '240P: ' + str(quality_statistics[0]) + '\t' + '480P: ' + str(quality_statistics[1]) + '\t' \
                       + '720P: ' + str(quality_statistics[2]) + '\t' + '1080P: ' + str(quality_statistics[3]) + '\n'
                log_file.write(line.encode())
                log_file.flush()

            # the action is from the last decision
            # this is to make the framework similar to the real
            else:
                delay, user_delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size_sr, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain, mec_video_remain, mask = \
                    net_env.get_video_chunk(bit_rate)

                quality_statistics[bit_rate] += 1


            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # -- linear reward --
            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[action] / M_IN_K \
                     - 4.3 * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[action] -
                                               VIDEO_BIT_RATE[last_action]) / M_IN_K

            # -- log scale reward --
            #log_bit_rate = np.log(VIDEO_BIT_RATE[action] / float(VIDEO_BIT_RATE[0]))
            #log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_action] / float(VIDEO_BIT_RATE[0]))

            #reward = log_bit_rate \
            #         - 2.66 * rebuf \
            #         - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

            # -- HD reward --
            #reward = HD_REWARD[action] \
            #         - 8.0 * rebuf \
            #         - SMOOTH_PENALTY * np.abs(HD_REWARD[action] - HD_REWARD[last_action])

            r_batch.append(reward)

            last_bit_rate = bit_rate
            last_action = action

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size_sr) / float(user_delay) / M_IN_B  # kilo byte / ms
            state[3, -1] = float(delay) / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :] = -1
            nxt_chnk_cnt = 0
            for i in range(A_DIM):
                if mask[i] == 1:
                    state[4, i] = next_video_chunk_sizes[nxt_chnk_cnt] / M_IN_B
                    nxt_chnk_cnt += 1
            assert (nxt_chnk_cnt) == np.sum(mask)
            state[5, -A_DIM:] = mask
            state[6, -1] = np.minimum(video_chunk_remain, video_chunk_num) / float(video_chunk_num)
            state[7, -1] = np.minimum(mec_video_remain, video_chunk_num) / float(video_chunk_num)

            # compute action probability vector
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            assert len(action_prob[0]) == np.sum(mask)
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            action = bitrate_to_action(bit_rate, mask)

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            # log time_stamp, bit_rate, buffer_size, reward
            line = 'Timestamp: ' + str(time_stamp) + '\t' + 'BitRate: ' + str(VIDEO_BIT_RATE[action]) + '\t' \
                   + 'BufferSize: ' + str(buffer_size) + '\t' + 'ReBuffer: ' + str(rebuf) + '\t' \
                   + 'SR_Video_Size: ' + str(video_chunk_size_sr) + '\t' + 'Delay: ' + str(delay) + '\t' \
                   + 'User-Delay: ' + str(user_delay) + '\t' + 'Reward: ' + str(reward) + '\n'
            log_file.write(line.encode())
            log_file.flush()

            # report experience to the coordinator
            if len(s_batch) >= TRAIN_SEQ_LEN or end_of_video:
                if len(s_batch) > 1:
                    exp_queue.put([s_batch[1:],  # ignore the first chuck
                                   a_batch[1:],  # since we don't have the
                                   r_batch[-len(a_batch) + 1:],  # control over it
                                   end_of_video,
                                   {'entropy': entropy_record}])

                    # synchronize the network parameters from the coordinator
                    actor_net_params, critic_net_params = net_params_queue.get()
                    actor.set_network_params(actor_net_params)
                    critic.set_network_params(critic_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]

                if len(r_batch) >= TRAIN_SEQ_LEN:
                    line = 'There is 100 chunk' + '\n' + '\n'
                    log_file.write(line.encode())  # so that in the log we know where video ends
                if end_of_video:
                    line = 'ALL-Stage:' + '240P: ' + str(quality_statistics[0]) + '\t' + '480P: ' + str(quality_statistics[1]) + '\t' \
                       + '720P: ' + str(quality_statistics[2]) + '\t' + '1080P: ' + str(quality_statistics[3]) + '\n' \
                       + 'Video is End' + '\n' + '\n'
                    log_file.write(line.encode())  # so that in the log we know where video ends

            # store the state and action into batches and reset initial state
            if end_of_video:
                bit_rate = DEFAULT_QUALITY  # use the default action here

                action = bitrate_to_action(bit_rate, mask)
                last_action = action

                action_vec = np.zeros(np.sum(mask))
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)

            else:
                s_batch.append(state)

                action_vec = np.zeros(np.sum(mask))
                action_vec[bit_rate] = 1
                a_batch.append(action_vec)


def main():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    all_cooked_user_time, all_cooked_user_bw, all_cooked_sat_time, all_cooked_sat_bw, _, _ \
        = load_trace.load_trace(TRAIN_TRACES_USER, TRAIN_TRACES_SAT)
    agents = []
    agent_rand = random.sample(range(0, 100), 16)
    print(agent_rand)
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(agent_rand[i], all_cooked_sat_time, all_cooked_sat_bw,
                                       all_cooked_user_time, all_cooked_user_bw,
                                       net_params_queues[i],
                                       exp_queues[i])))

    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
