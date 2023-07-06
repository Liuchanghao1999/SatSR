import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import tensorflow as tf
import load_trace
import a3c
import test_env as env
import warnings
warnings.filterwarnings('ignore')


S_INFO = 8  # bit_rate, buffer_size, transmissed chunk size, transmission delay, next_chunk_size, mask, downloaded low-quality chunk number, remain chunk number
S_LEN = 11
A_DIM = 11
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [250, 500, 650, 900, 1150 ,1350, 1750, 2000, 2350, 3100, 3900]
HD_REWARD = [1, 1.5, 2, 3, 4, 5, 6, 11, 13, 15, 18]
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
M_IN_B = 1024 * 1024
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
LOG_FILE = './test_results/log_sim_rl'
TEST_TRACES_SAT = './cooked_test_traces/sat/'
TEST_TRACES_USER = './cooked_test_traces/user/'
NN_MODEL = sys.argv[1]


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


def main():
    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    user_all_cooked_time, user_all_cooked_bw, sat_all_cooked_time, sat_all_cooked_bw, \
    user_all_file_names, sat_all_file_names = load_trace.load_trace(TEST_TRACES_USER, TEST_TRACES_SAT)

    net_env = env.Environment(all_cooked_user_time=user_all_cooked_time,
                              all_cooked_user_bw=user_all_cooked_bw,
                              all_cooked_sat_time=sat_all_cooked_time,
                              all_cooked_sat_bw=sat_all_cooked_bw)

    log_path = LOG_FILE + '_' + user_all_file_names[net_env.user_trace_idx] + '_' + sat_all_file_names[net_env.sat_trace_idx] + '_0'
    log_file = open(log_path, 'wb')

    with tf.Session() as sess:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")

        time_stamp = 0
        quality_statistics = [0, 0, 0, 0]

        mask = net_env.video_masks[net_env.video_idx]
        video_chunk_num = net_env.video_chunk_num

        bit_rate = DEFAULT_QUALITY
        end_of_video = True

        action = bitrate_to_action(bit_rate, mask)
        last_action = action

        action_vec = np.zeros(np.sum(mask))
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            if end_of_video:
                delay, user_delay, buffer_size, rebuf, video_chunk_size_sr, \
                next_video_chunk_sizes, end_of_video, video_chunk_remain, mec_video_remain, \
                reward_set, quality_statistics, quality, mask = \
                    net_env.early_stage_trans(bit_rate)
                sleep_time = 0
                r_batch.extend(reward_set)

                bit_rate = last_bit_rate = quality
                action = last_action = bitrate_to_action(bit_rate, mask)

                line = ''
                for r in reward_set:
                    line = line + 'Reward: ' + str(r) + '\n'
                log_file.write(line.encode())
                log_file.flush()

                line = 'Early-Stage:' + '240P: ' + str(quality_statistics[0]) + '\t' + '480P: ' + str(
                    quality_statistics[1]) + '\t' \
                       + '720P: ' + str(quality_statistics[2]) + '\t' + '1080P: ' + str(quality_statistics[3]) + '\t' + 'SR Start !' + '\n'
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
            #reward = VIDEO_BIT_RATE[action] / M_IN_K \
            #         - 4.3 * rebuf \
            #         - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[action] -
            #                                   VIDEO_BIT_RATE[last_action]) / M_IN_K

            # -- log scale reward --
            #log_bit_rate = np.log(VIDEO_BIT_RATE[action] / float(VIDEO_BIT_RATE[0]))
            #log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_action] / float(VIDEO_BIT_RATE[0]))

            #reward = log_bit_rate \
            #        - 2.66 * rebuf \
            #        - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

            # -- HD reward --
            reward = HD_REWARD[action] \
                    - 8.0 * rebuf \
                    - SMOOTH_PENALTY * np.abs(HD_REWARD[action] - HD_REWARD[last_action])

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

            s_batch.append(state)

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            line = 'Timestamp: ' + str(time_stamp) + '\t' + 'BitRate: ' + str(bit_rate) + '\t' \
                   + 'BufferSize: ' + str(buffer_size) + '\t' + 'ReBuffer: ' + str(rebuf) + '\t' \
                   + 'SR_Video: ' + str(video_chunk_size_sr) + '\t' + 'Delay: ' + str(delay) + '\t' \
                   + 'User-Delay: ' + str(user_delay) + '\t' + 'Reward: ' + str(reward) + '\n'
            log_file.write(line.encode())
            log_file.flush()

            if end_of_video:
                line = 'ALL-Stage:' + '240P: ' + str(quality_statistics[0]) + '\t' + '480P: ' + str(quality_statistics[1]) + '\t' \
                       + '720P: ' + str(quality_statistics[2]) + '\t' + '1080P: ' + str(quality_statistics[3]) + '\t' \
                       + 'Video is End !' + '\n'

                log_file.write(line.encode())
                log_file.close()

                bit_rate = DEFAULT_QUALITY  # use the default action here

                action = bitrate_to_action(bit_rate, mask)
                last_action = action

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(np.sum(mask))
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
                entropy_record = []

                video_count += 1

                if video_count >= max(len(user_all_file_names), len(sat_all_file_names)) * 5 - 1:
                    break

                log_path = LOG_FILE + '_' + user_all_file_names[net_env.user_trace_idx] + '_' + sat_all_file_names[net_env.sat_trace_idx] + '_' + str(video_count)

                log_file = open(log_path, 'wb')


if __name__ == '__main__':
    main()
