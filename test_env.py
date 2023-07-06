import numpy as np

B_IN_MB = 1024 * 1024
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4.0  # sec, every time add this amount to buffer
BITRATE_LEVELS = 4
BUFFER_THRESH = 60.0  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 0.5  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 0.08  # sec
SR_DELAY = {0:0, 1:0.88, 2:1.35, 3:2.05} # sec
BANDWIDTH = [[272569, 809338, 1511207, 2872440], [549166, 1443423, 2555485, 4455916],
             [396568, 1213394, 2309231, 4133744], [314275, 749665, 1401116, 2515021],
             [511663, 1578509, 2965426, 5305949]]
MBT = 4.0
DNN_SIZE = 4302035
SMOOTH_PENALTY = 1
VIDEO_BIT_RATE = [250, 500, 650, 900, 1150 ,1350, 1750, 2000, 2350, 3100, 3900]
HD_REWARD = [1, 1.5, 2, 3, 4, 5, 6, 11, 13, 15, 18]
M_IN_K = 1000.0
A_DIM = 11
SR_VIDEO_SIZE_FILE = ['./iPHONE_SR_size_', './challenge_SR_size_', './CatBox_SR_size_', './favorite_SR_size_', './PS4_SR_size_']
HD_VIDEO_SIZE_FILE = ['./iPHONE_HD_size_', './challenge_HD_size_', './CatBox_HD_size_', './favorite_HD_size_', './PS4_HD_size_']
VIDEO_CHUNK_NUM = [86, 202, 176, 180, 179]

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


class Environment:
    def __init__(self, all_cooked_user_time, all_cooked_user_bw, all_cooked_sat_time, all_cooked_sat_bw, random_seed=RANDOM_SEED):
        assert len(all_cooked_user_time) == len(all_cooked_user_bw)
        assert len(all_cooked_sat_time) == len(all_cooked_sat_bw)

        np.random.seed(random_seed)

        self.all_cooked_user_time = all_cooked_user_time
        self.all_cooked_user_bw = all_cooked_user_bw
        self.all_cooked_sat_time = all_cooked_sat_time
        self.all_cooked_sat_bw = all_cooked_sat_bw

        self.video_masks = {}
        self.video_masks[0] = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
        self.video_masks[1] = [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1]
        self.video_masks[2] = [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
        self.video_masks[3] = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
        self.video_masks[4] = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]

        self.sat_video_chunk_counter = 0
        self.user_video_chunk_counter = 0
        self.buffer_size = 0
        self.mec_video_remain = 0
        self.sat_last_trans_video_counter_sent = 0
        self.last_trans_dnn_counter_sent = 0
        self.first_sr_video_counter_sent = 0
        self.last_user_timestamp = 0

        # pick a random trace file
        self.sat_trace_idx = 0
        self.user_trace_idx = 0
        self.sat_time = self.all_cooked_sat_time[self.sat_trace_idx]
        self.sat_bw = self.all_cooked_sat_bw[self.sat_trace_idx]
        self.user_time = self.all_cooked_user_time[self.user_trace_idx]
        self.user_bw = self.all_cooked_user_bw[self.user_trace_idx]
        self.sat_cycle = 0
        self.user_cycle = 0

        self.mahimahi_ptr_sat = 1
        self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat - 1]
        self.mahimahi_ptr_user = self.mahimahi_ptr_sat
        self.last_mahimahi_time_user = self.user_time[self.mahimahi_ptr_user - 1]

        self.video_idx = 0
        self.video_chunk_num = VIDEO_CHUNK_NUM[self.video_idx]
        self.sr_video_size = {}
        self.hd_video_size = {}
        for bitrate in range(BITRATE_LEVELS):
            self.sr_video_size[bitrate] = []
            self.hd_video_size[bitrate] = []
            with open(SR_VIDEO_SIZE_FILE[self.video_idx] + str(bitrate) + '.log') as f:
                for line in f:
                    self.sr_video_size[bitrate].append(int(line.split()[0]))

            with open(HD_VIDEO_SIZE_FILE[self.video_idx] + str(bitrate) + '.log') as f:
                for line in f:
                    self.hd_video_size[bitrate].append(int(line.split()[0]))


    def early_stage_trans(self, first_quality):
        assert self.sat_video_chunk_counter == 0
        assert self.user_video_chunk_counter == 0

        video_chunk_size_hd = 0
        quality = first_quality
        delay = 0.0
        user_delay = 0.0
        rebuf = 0.0
        quality_statistics = [0, 0, 0, 0]
        reward_set = []
        action = last_action = bitrate_to_action(first_quality, self.video_masks[self.video_idx])

        while self.last_trans_dnn_counter_sent < DNN_SIZE:
            user_rate = []
            delay = 1.0
            user_delay = 0.0
            assert self.sat_video_chunk_counter == self.sat_video_chunk_counter
            assert quality < 4
            #print(quality, self.user_video_chunk_counter)
            if self.user_video_chunk_counter >= self.video_chunk_num - 1:
                print(self.sat_trace_idx, self.user_trace_idx, self.user_video_chunk_counter, self.last_trans_dnn_counter_sent, self.last_mahimahi_time_sat, self.last_mahimahi_time_user, self.sat_bw[0], self.user_bw[0])
            video_chunk_size_hd = self.hd_video_size[quality][self.user_video_chunk_counter]
            video_chunk_counter_sent = self.sat_last_trans_video_counter_sent

            while True:  # download HD-video chunk over sat-link
                if self.sat_bw[self.mahimahi_ptr_sat] < 0.2:
                    throughput = self.sat_bw[self.mahimahi_ptr_sat - 1] \
                                 * B_IN_MB / BITS_IN_BYTE
                else:
                    throughput = self.sat_bw[self.mahimahi_ptr_sat] \
                                 * B_IN_MB / BITS_IN_BYTE
                duration = self.sat_time[self.mahimahi_ptr_sat] \
                           - self.last_mahimahi_time_sat

                packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                if video_chunk_counter_sent + packet_payload > video_chunk_size_hd:

                    fractional_time = (video_chunk_size_hd - video_chunk_counter_sent) / \
                                      throughput / PACKET_PAYLOAD_PORTION
                    #delay += fractional_time
                    self.last_mahimahi_time_sat += fractional_time
                    assert(self.last_mahimahi_time_sat <= self.sat_time[self.mahimahi_ptr_sat])
                    self.mec_video_remain += 1
                    self.sat_last_trans_video_counter_sent = 0
                    assert self.sat_video_chunk_counter < self.video_chunk_num
                    self.sat_video_chunk_counter += 1
                    break

                video_chunk_counter_sent += packet_payload
                #delay += duration
                self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat]
                self.mahimahi_ptr_sat += 1

                if self.mahimahi_ptr_sat >= len(self.sat_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr_sat = 1
                    self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat - 1]
                    self.sat_cycle += 1


            timestamp_sat = self.sat_cycle * (self.sat_time[-1] - self.sat_time[0]) + self.last_mahimahi_time_sat - self.sat_time[0]
            timestamp_user = self.user_cycle * (self.user_time[-1] - self.user_time[0]) + self.last_mahimahi_time_user - self.user_time[0]

            assert timestamp_user <= timestamp_sat
            self.last_mahimahi_time_user = timestamp_sat % (self.user_time[-1] - self.user_time[0]) + self.user_time[0]
            self.user_cycle = int(timestamp_sat / (self.user_time[-1] - self.user_time[0]))
            for i in range(len(self.user_time)):
                self.mahimahi_ptr_user = i
                if self.last_mahimahi_time_user < self.user_time[i]:
                    break

            assert self.sat_video_chunk_counter > self.user_video_chunk_counter

            video_chunk_counter_sent_hd = 0
            while True:  # download HD-video chunk over user-link
                if self.user_bw[self.mahimahi_ptr_user] < 0.2:
                    throughput = self.user_bw[self.mahimahi_ptr_user - 1] \
                                 * B_IN_MB / BITS_IN_BYTE
                else:
                    throughput = self.user_bw[self.mahimahi_ptr_user] \
                                 * B_IN_MB / BITS_IN_BYTE
                duration = self.user_time[self.mahimahi_ptr_user] \
                           - self.last_mahimahi_time_user

                user_rate.append(throughput)

                packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                if video_chunk_counter_sent_hd + packet_payload > video_chunk_size_hd:
                    fractional_time = (video_chunk_size_hd - video_chunk_counter_sent_hd) / \
                                      throughput / PACKET_PAYLOAD_PORTION
                    #delay += fractional_time
                    user_delay += fractional_time
                    self.last_mahimahi_time_user += fractional_time
                    self.mec_video_remain -= 1
                    self.user_video_chunk_counter += 1
                    now_user_timestamp = self.user_cycle * (self.user_time[-1] - self.user_time[0]) + self.last_mahimahi_time_user - self.user_time[0]
                    delay = delay + now_user_timestamp - self.last_user_timestamp + LINK_RTT
                    assert delay > 0.0
                    self.last_user_timestamp = now_user_timestamp
                    assert self.last_mahimahi_time_user <= self.user_time[self.mahimahi_ptr_user]
                    break

                video_chunk_counter_sent_hd+= packet_payload
                #delay += duration
                user_delay += duration

                self.last_mahimahi_time_user = self.user_time[self.mahimahi_ptr_user]
                self.mahimahi_ptr_user += 1

                if self.mahimahi_ptr_user >= len(self.user_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr_user = 1
                    self.last_mahimahi_time_user = self.user_time[self.mahimahi_ptr_user - 1]
                    self.user_cycle += 1



            rebuf = np.maximum(delay - self.buffer_size, 0.0)
            self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

            timestamp_user = self.user_cycle * (self.user_time[-1] - self.user_time[0]) + self.last_mahimahi_time_user - self.user_time[0]
            assert timestamp_user > timestamp_sat
            self.new_last_mahimahi_time_sat = timestamp_user % (self.sat_time[-1] - self.sat_time[0]) + self.sat_time[0]
            self.new_sat_cycle = int(timestamp_user / (self.sat_time[-1] - self.sat_time[0]))
            for i in range(len(self.sat_time)):
                self.new_mahimahi_ptr_sat = i
                if self.new_last_mahimahi_time_sat < self.sat_time[i]:
                    break

            #if self.sat_cycle < self.new_sat_cycle:
            while self.sat_cycle < self.new_sat_cycle:
                dnn_counter_sent = self.last_trans_dnn_counter_sent
                #dnn_chunk_size = self.raw_video_size[0][self.sat_video_chunk_counter]

                while True:
                    if self.sat_bw[self.mahimahi_ptr_sat] < 0.2:
                        throughput = self.sat_bw[self.mahimahi_ptr_sat - 1] \
                                     * B_IN_MB / BITS_IN_BYTE
                    else:
                        throughput = self.sat_bw[self.mahimahi_ptr_sat] \
                                     * B_IN_MB / BITS_IN_BYTE
                    duration = self.sat_time[self.mahimahi_ptr_sat] \
                               - self.last_mahimahi_time_sat

                    packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                    if dnn_counter_sent + packet_payload > DNN_SIZE:
                        fractional_time = (DNN_SIZE - dnn_counter_sent) / \
                                          throughput / PACKET_PAYLOAD_PORTION
                        self.last_trans_dnn_counter_sent = DNN_SIZE
                        self.last_mahimahi_time_sat += fractional_time
                        assert self.last_mahimahi_time_sat <= self.sat_time[self.mahimahi_ptr_sat]
                        break

                    dnn_counter_sent += packet_payload
                    #self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat]
                    self.mahimahi_ptr_sat += 1

                    if self.mahimahi_ptr_sat >= len(self.sat_bw):
                        # loop back in the beginning
                        # note: trace file starts with time 0
                        self.mahimahi_ptr_sat = 1
                        self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat - 1]
                        self.sat_cycle += 1
                        if self.sat_cycle == self.new_sat_cycle:
                            self.last_trans_dnn_counter_sent = dnn_counter_sent
                            break

                if self.last_trans_dnn_counter_sent == DNN_SIZE:
                    #传输视频块
                    while self.sat_cycle < self.new_sat_cycle:
                        video_chunk_counter_sent = self.sat_last_trans_video_counter_sent
                        video_chunk_size_raw = self.hd_video_size[0][self.sat_video_chunk_counter]

                        while True:
                            if self.sat_bw[self.mahimahi_ptr_sat] < 0.2:
                                throughput = self.sat_bw[self.mahimahi_ptr_sat - 1] \
                                             * B_IN_MB / BITS_IN_BYTE
                            else:
                                throughput = self.sat_bw[self.mahimahi_ptr_sat] \
                                             * B_IN_MB / BITS_IN_BYTE
                            duration = self.sat_time[self.mahimahi_ptr_sat] \
                                       - self.last_mahimahi_time_sat

                            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                            if video_chunk_counter_sent + packet_payload > video_chunk_size_raw:
                                fractional_time = (video_chunk_size_raw - video_chunk_counter_sent) / \
                                                  throughput / PACKET_PAYLOAD_PORTION
                                self.last_mahimahi_time_sat += fractional_time
                                assert self.last_mahimahi_time_sat <= self.sat_time[self.mahimahi_ptr_sat]
                                self.mec_video_remain += 1
                                if self.mec_video_remain == 1:
                                    timestamp_sr = self.sat_cycle * (self.sat_time[-1] - self.sat_time[0]) + self.last_mahimahi_time_sat - self.sat_time[0]
                                    if quality > 0:
                                        timestamp_sr += SR_DELAY[quality]
                                assert self.sat_video_chunk_counter < self.video_chunk_num
                                self.sat_video_chunk_counter += 1
                                self.sat_last_trans_video_counter_sent = 0
                                break

                            video_chunk_counter_sent += packet_payload
                            self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat]
                            self.mahimahi_ptr_sat += 1

                            if self.mahimahi_ptr_sat >= len(self.sat_bw):
                                # loop back in the beginning
                                # note: trace file starts with time 0
                                self.mahimahi_ptr_sat = 1
                                self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat - 1]
                                self.sat_cycle += 1
                                if self.sat_cycle == self.new_sat_cycle:
                                    self.sat_last_trans_video_counter_sent = video_chunk_counter_sent
                                    break


            assert self.sat_cycle == self.new_sat_cycle

            while self.mahimahi_ptr_sat < self.new_mahimahi_ptr_sat:
                if self.last_trans_dnn_counter_sent < DNN_SIZE:
                    dnn_counter_sent = self.last_trans_dnn_counter_sent
                    while True:
                        if self.sat_bw[self.mahimahi_ptr_sat] < 0.2:
                            throughput = self.sat_bw[self.mahimahi_ptr_sat - 1] \
                                         * B_IN_MB / BITS_IN_BYTE
                        else:
                            throughput = self.sat_bw[self.mahimahi_ptr_sat] \
                                         * B_IN_MB / BITS_IN_BYTE
                        duration = self.sat_time[self.mahimahi_ptr_sat] \
                                   - self.last_mahimahi_time_sat

                        packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                        if dnn_counter_sent + packet_payload > DNN_SIZE:
                            fractional_time = (DNN_SIZE - dnn_counter_sent) / \
                                              throughput / PACKET_PAYLOAD_PORTION
                            self.last_trans_dnn_counter_sent = DNN_SIZE
                            self.last_mahimahi_time_sat += fractional_time
                            assert self.last_mahimahi_time_sat <= self.sat_time[self.mahimahi_ptr_sat]
                            break

                        dnn_counter_sent += packet_payload
                        self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat]
                        self.mahimahi_ptr_sat += 1

                        if self.mahimahi_ptr_sat == self.new_mahimahi_ptr_sat:
                            # self.last_trans_dnn_counter_sent = dnn_counter_sent
                            # GEO卫星系统存在请求时延，DNN可以多传输500ms
                            self.last_trans_dnn_counter_sent = dnn_counter_sent + throughput * 0.5
                            break

                if self.last_trans_dnn_counter_sent == DNN_SIZE:
                    while self.mahimahi_ptr_sat < self.new_mahimahi_ptr_sat:
                        if self.sat_video_chunk_counter >= self.video_chunk_num:
                            print(self.sat_trace_idx, self.user_trace_idx, self.user_video_chunk_counter,
                                  self.last_trans_dnn_counter_sent, self.last_mahimahi_time_sat,
                                  self.last_mahimahi_time_user, self.sat_bw[0], self.user_bw[0])

                        video_chunk_counter_sent = self.sat_last_trans_video_counter_sent
                        video_chunk_size_raw = self.hd_video_size[0][self.sat_video_chunk_counter]

                        while True:
                            if self.sat_bw[self.mahimahi_ptr_sat] < 0.2:
                                throughput = self.sat_bw[self.mahimahi_ptr_sat - 1] \
                                             * B_IN_MB / BITS_IN_BYTE
                            else:
                                throughput = self.sat_bw[self.mahimahi_ptr_sat] \
                                             * B_IN_MB / BITS_IN_BYTE
                            duration = self.sat_time[self.mahimahi_ptr_sat] \
                                       - self.last_mahimahi_time_sat

                            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                            if video_chunk_counter_sent + packet_payload > video_chunk_size_raw:
                                fractional_time = (video_chunk_size_raw - video_chunk_counter_sent) / \
                                                  throughput / PACKET_PAYLOAD_PORTION
                                self.last_mahimahi_time_sat += fractional_time
                                assert self.last_mahimahi_time_sat <= self.sat_time[self.mahimahi_ptr_sat]
                                self.mec_video_remain += 1
                                if self.mec_video_remain == 1:
                                    timestamp_sr = self.sat_cycle * (self.sat_time[-1] - self.sat_time[0]) + self.last_mahimahi_time_sat - self.sat_time[0]
                                    if quality > 0:
                                        timestamp_sr += SR_DELAY[quality]
                                assert self.sat_video_chunk_counter < self.video_chunk_num
                                self.sat_video_chunk_counter += 1
                                self.sat_last_trans_video_counter_sent = 0
                                break

                            video_chunk_counter_sent += packet_payload
                            self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat]
                            self.mahimahi_ptr_sat += 1

                            if self.mahimahi_ptr_sat == self.new_mahimahi_ptr_sat:
                                self.sat_last_trans_video_counter_sent = video_chunk_counter_sent
                                break

                        if self.mahimahi_ptr_sat == self.new_mahimahi_ptr_sat:
                            break

            if self.mahimahi_ptr_sat == self.new_mahimahi_ptr_sat:
                dnn_counter_sent = self.last_trans_dnn_counter_sent
                if self.sat_bw[self.mahimahi_ptr_sat] < 0.2:
                    throughput = self.sat_bw[self.mahimahi_ptr_sat - 1] \
                                 * B_IN_MB / BITS_IN_BYTE
                else:
                    throughput = self.sat_bw[self.mahimahi_ptr_sat] \
                                 * B_IN_MB / BITS_IN_BYTE
                duration = self.new_last_mahimahi_time_sat \
                           - self.last_mahimahi_time_sat
                assert duration >= 0
                packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                if dnn_counter_sent + packet_payload > DNN_SIZE:
                    fractional_time = (DNN_SIZE - dnn_counter_sent) / \
                                      throughput / PACKET_PAYLOAD_PORTION

                    self.last_trans_dnn_counter_sent = DNN_SIZE
                    self.sat_last_trans_video_counter_sent = throughput * (self.new_last_mahimahi_time_sat - fractional_time) * PACKET_PAYLOAD_PORTION

                else:
                    dnn_counter_sent += packet_payload
                    self.last_trans_dnn_counter_sent = dnn_counter_sent

                self.last_mahimahi_time_sat = self.new_last_mahimahi_time_sat

            self.buffer_size += VIDEO_CHUNCK_LEN

            if self.last_trans_dnn_counter_sent == DNN_SIZE:
                break

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

            reward_set.append(reward)

            quality_statistics[quality] += 1

            # if quality == last_quality:
            if self.buffer_size >= 10.0:
                bola_size = min(self.user_video_chunk_counter - 1,
                                self.video_chunk_num - self.user_video_chunk_counter + 1)
                bola_size = max(bola_size / 2, 3)
                bola_buffer_size = min(15.0, bola_size)  # max_buffer_siez = 60.0 = 4.0 * 15
                next_video_chunk_sizes = []
                for i in range(BITRATE_LEVELS):
                    next_video_chunk_sizes.append(self.hd_video_size[i][self.user_video_chunk_counter])
                video_size_min = min(next_video_chunk_sizes)
                video_size_max = max(next_video_chunk_sizes)
                bola_v = (bola_buffer_size - 1) / (np.log(video_size_max / video_size_min) + 1.0 * VIDEO_CHUNCK_LEN)
                bola_value = -9999.9
                next_quality = 0
                for i in range(len(next_video_chunk_sizes)):
                    bola_value_new = (bola_v * (np.log(next_video_chunk_sizes[
                                                           i] / video_size_min) + 1.0 * VIDEO_CHUNCK_LEN) - self.buffer_size / VIDEO_CHUNCK_LEN) / \
                                     next_video_chunk_sizes[i]
                    if bola_value_new > bola_value:
                        bola_value = bola_value_new
                        next_quality = i

            else:
                rate = sum(user_rate) / len(user_rate)
                bandwidth = rate * self.buffer_size / MBT
                next_quality = 0
                for i in range(4):
                    if bandwidth >= BANDWIDTH[self.video_idx][i]:
                        next_quality = i
                    else:
                        break

                # if self.buffer_size > RESEVOIR and quality < 3:
                #    next_quality += 1

            # else:
            #    next_quality = quality

            last_quality = quality
            quality = next_quality
            last_action = action
            action = bitrate_to_action(quality, self.video_masks[self.video_idx])


            #print('Train:  Video_idx: {}; DNN_download: {}; Bandwidth: {}; Buffer: {}; Rebuf: {}; Quality: {}'.format(self.user_video_chunk_counter, self.last_trans_dnn_counter_sent, bandwidth, self.buffer_size, rebuf, quality))

        return_quality = quality # 最后一个传输chunk的质量
        return_mec_video_remain = self.mec_video_remain

        if self.mec_video_remain == 0:
            video_chunk_counter_sent = self.sat_last_trans_video_counter_sent
            video_chunk_size_raw = self.hd_video_size[0][self.sat_video_chunk_counter]

            while True:
                if self.sat_bw[self.mahimahi_ptr_sat] < 0.2:
                    throughput = self.sat_bw[self.mahimahi_ptr_sat - 1] \
                                 * B_IN_MB / BITS_IN_BYTE
                else:
                    throughput = self.sat_bw[self.mahimahi_ptr_sat] \
                                 * B_IN_MB / BITS_IN_BYTE
                duration = self.sat_time[self.mahimahi_ptr_sat] \
                           - self.last_mahimahi_time_sat

                packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                if video_chunk_counter_sent + packet_payload > video_chunk_size_raw:
                    fractional_time = (video_chunk_size_raw - video_chunk_counter_sent) / \
                                      throughput / PACKET_PAYLOAD_PORTION
                    self.last_mahimahi_time_sat += fractional_time
                    assert (self.last_mahimahi_time_sat <= self.sat_time[self.mahimahi_ptr_sat])
                    self.mec_video_remain += 1
                    if self.mec_video_remain == 1:
                        timestamp_sr = self.sat_cycle * (self.sat_time[-1] - self.sat_time[0]) + self.last_mahimahi_time_sat - self.sat_time[0]
                        if quality > 0:
                            timestamp_sr += SR_DELAY[quality]
                    #print('self.sat_video_chunk_counter: {}'.format(self.sat_video_chunk_counter))

                    if self.sat_video_chunk_counter >= self.video_chunk_num - 1:
                        print(self.sat_trace_idx, self.user_trace_idx, self.user_video_chunk_counter,
                              self.last_trans_dnn_counter_sent, self.last_mahimahi_time_sat,
                              self.last_mahimahi_time_user, self.sat_bw[0], self.user_bw[0])

                    assert self.sat_video_chunk_counter < self.video_chunk_num - 1
                    self.sat_video_chunk_counter += 1
                    self.sat_last_trans_video_counter_sent = 0
                    break

                video_chunk_counter_sent += packet_payload
                self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat]
                self.mahimahi_ptr_sat += 1

                if self.mahimahi_ptr_sat >= len(self.sat_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr_sat = 1
                    self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat - 1]
                    self.sat_cycle += 1


        if timestamp_sr > self.user_cycle * (self.user_time[-1] - self.user_time[0]) + self.last_mahimahi_time_user - self.user_time[0]:
            self.last_mahimahi_time_user = timestamp_sr % (self.user_time[-1] - self.user_time[0]) + self.user_time[0]
            self.user_cycle = int(timestamp_sr / (self.user_time[-1] - self.user_time[0]))
            for i in range(len(self.user_time)):
                self.mahimahi_ptr_user = i
                if self.last_mahimahi_time_user < self.user_time[i]:
                    break

        #return_reward = sum(reward_set)
        return_buffer_size = self.buffer_size
        video_chunk_remain = self.video_chunk_num - self.user_video_chunk_counter
        bitrate_mask = self.video_masks[self.video_idx]
        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.sr_video_size[i][self.user_video_chunk_counter])

        assert self.user_video_chunk_counter < self.video_chunk_num
        end_of_video = False

        return delay, \
               user_delay, \
               return_buffer_size, \
               rebuf, \
               video_chunk_size_hd, \
               next_video_chunk_sizes, \
               end_of_video, \
               video_chunk_remain, \
               return_mec_video_remain, \
               reward_set, \
               quality_statistics, \
               return_quality, \
               bitrate_mask



    def get_video_chunk(self, quality):

        assert quality >= 0
        assert quality < BITRATE_LEVELS

        delay = 1.0
        user_delay = 0.0

        # user_link trans SR-video while do SR
        assert self.sat_video_chunk_counter > self.user_video_chunk_counter
        video_chunk_counter_sent_user = 0  # in bytes
        user_timestamp_start = self.user_cycle * (self.user_time[-1] - self.user_time[0]) + self.last_mahimahi_time_user - self.user_time[0]
        video_chunk_size_sr = self.sr_video_size[quality][self.user_video_chunk_counter]

        while True:  # download SR-video chunk over mahimahi
            if self.user_bw[self.mahimahi_ptr_user] < 0.2:
                throughput = self.user_bw[self.mahimahi_ptr_user - 1] \
                             * B_IN_MB / BITS_IN_BYTE
            else:
                throughput = self.user_bw[self.mahimahi_ptr_user] \
                             * B_IN_MB / BITS_IN_BYTE
            duration = self.user_time[self.mahimahi_ptr_user] \
                       - self.last_mahimahi_time_user

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent_user + packet_payload > video_chunk_size_sr:
                fractional_time = (video_chunk_size_sr - video_chunk_counter_sent_user) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                # delay += fractional_time
                user_delay += fractional_time
                self.last_mahimahi_time_user += fractional_time
                self.mec_video_remain -= 1
                self.user_video_chunk_counter += 1
                now_user_timestamp = self.user_cycle * (self.user_time[-1] - self.user_time[0]) + self.last_mahimahi_time_user - self.user_time[0]
                delay = delay + now_user_timestamp - self.last_user_timestamp + LINK_RTT
                assert delay > 0.0
                self.last_user_timestamp = now_user_timestamp
                assert self.last_mahimahi_time_user <= self.user_time[self.mahimahi_ptr_user]
                break

            video_chunk_counter_sent_user += packet_payload
            # delay += duration
            user_delay += duration

            self.last_mahimahi_time_user = self.user_time[self.mahimahi_ptr_user]
            self.mahimahi_ptr_user += 1

            if self.mahimahi_ptr_user >= len(self.user_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr_user = 1
                self.last_mahimahi_time_user = self.user_time[self.mahimahi_ptr_user - 1]
                self.user_cycle += 1

        timestamp_sat = self.sat_cycle * (self.sat_time[-1] - self.sat_time[0]) + self.last_mahimahi_time_sat - self.sat_time[0]
        timestamp_user = self.user_cycle * (self.user_time[-1] - self.user_time[0]) + self.last_mahimahi_time_user - self.user_time[0]
        assert timestamp_user > timestamp_sat
        self.new_last_mahimahi_time_sat = timestamp_user % (self.sat_time[-1] - self.sat_time[0]) + self.sat_time[0]
        self.new_sat_cycle = int(timestamp_user / (self.sat_time[-1] - self.sat_time[0]))
        for i in range(len(self.sat_time)):
            self.new_mahimahi_ptr_sat = i
            if self.new_last_mahimahi_time_sat < self.sat_time[i]:
                break

        while self.sat_cycle < self.new_sat_cycle:
            if self.sat_video_chunk_counter >= self.video_chunk_num:
                break
            video_chunk_counter_sent = self.sat_last_trans_video_counter_sent
            video_chunk_size_raw = self.hd_video_size[0][self.sat_video_chunk_counter]

            while True:
                if self.sat_bw[self.mahimahi_ptr_sat] < 0.2:
                    throughput = self.sat_bw[self.mahimahi_ptr_sat - 1] \
                                 * B_IN_MB / BITS_IN_BYTE
                else:
                    throughput = self.sat_bw[self.mahimahi_ptr_sat] \
                                 * B_IN_MB / BITS_IN_BYTE
                duration = self.sat_time[self.mahimahi_ptr_sat] \
                           - self.last_mahimahi_time_sat

                packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                if video_chunk_counter_sent + packet_payload > video_chunk_size_raw:
                    fractional_time = (video_chunk_size_raw - video_chunk_counter_sent) / \
                                      throughput / PACKET_PAYLOAD_PORTION
                    self.last_mahimahi_time_sat += fractional_time
                    assert self.last_mahimahi_time_sat <= self.sat_time[self.mahimahi_ptr_sat]
                    self.sat_last_trans_video_counter_sent = 0
                    self.mec_video_remain += 1
                    self.sat_video_chunk_counter += 1
                    if self.sat_video_chunk_counter >= self.video_chunk_num:
                        break

                    break

                video_chunk_counter_sent += packet_payload
                self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat]
                self.mahimahi_ptr_sat += 1

                if self.mahimahi_ptr_sat >= len(self.sat_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr_sat = 1
                    self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat - 1]
                    self.sat_cycle += 1
                    if self.sat_cycle == self.new_sat_cycle:
                        self.sat_last_trans_video_counter_sent = video_chunk_counter_sent
                        break


        while self.mahimahi_ptr_sat < self.new_mahimahi_ptr_sat:
            if self.sat_video_chunk_counter >= self.video_chunk_num:
                break
            video_chunk_counter_sent = self.sat_last_trans_video_counter_sent
            video_chunk_size_raw = self.hd_video_size[0][self.sat_video_chunk_counter]

            while True:
                if self.sat_bw[self.mahimahi_ptr_sat] < 0.2:
                    throughput = self.sat_bw[self.mahimahi_ptr_sat - 1] \
                                 * B_IN_MB / BITS_IN_BYTE
                else:
                    throughput = self.sat_bw[self.mahimahi_ptr_sat] \
                                 * B_IN_MB / BITS_IN_BYTE
                duration = self.sat_time[self.mahimahi_ptr_sat] \
                           - self.last_mahimahi_time_sat

                packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                if video_chunk_counter_sent + packet_payload > video_chunk_size_raw:
                    fractional_time = (video_chunk_size_raw - video_chunk_counter_sent) / \
                                      throughput / PACKET_PAYLOAD_PORTION
                    self.last_mahimahi_time_sat += fractional_time
                    assert self.last_mahimahi_time_sat <= self.sat_time[self.mahimahi_ptr_sat]
                    self.sat_last_trans_video_counter_sent = 0
                    self.mec_video_remain += 1
                    self.sat_video_chunk_counter += 1
                    if self.sat_video_chunk_counter >= self.video_chunk_num:
                        break

                    break

                video_chunk_counter_sent += packet_payload
                self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat]
                self.mahimahi_ptr_sat += 1

                if self.mahimahi_ptr_sat == self.new_mahimahi_ptr_sat:
                    self.sat_last_trans_video_counter_sent = video_chunk_counter_sent
                    break

            if self.mahimahi_ptr_sat == self.new_mahimahi_ptr_sat:
                break

        if self.mahimahi_ptr_sat == self.new_mahimahi_ptr_sat and self.sat_video_chunk_counter < self.video_chunk_num:

            video_chunk_size_raw = self.hd_video_size[0][self.sat_video_chunk_counter]

            if self.sat_bw[self.mahimahi_ptr_sat] < 0.2:
                throughput = self.sat_bw[self.mahimahi_ptr_sat - 1] \
                             * B_IN_MB / BITS_IN_BYTE
            else:
                throughput = self.sat_bw[self.mahimahi_ptr_sat] \
                             * B_IN_MB / BITS_IN_BYTE
            duration = self.new_last_mahimahi_time_sat \
                       - self.last_mahimahi_time_sat
            assert duration >= 0
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            self.sat_last_trans_video_counter_sent += packet_payload

            if self.sat_last_trans_video_counter_sent > video_chunk_size_raw:
                self.sat_last_trans_video_counter_sent -= video_chunk_size_raw
                self.mec_video_remain += 1
                self.sat_video_chunk_counter += 1


        return_mec_remain = self.mec_video_remain

        #Do SR while user_link trans SR-video
        assert self.mec_video_remain >= 0
        if self.mec_video_remain == 0 and self.sat_video_chunk_counter < self.video_chunk_num:
            video_chunk_size_raw = self.hd_video_size[0][self.sat_video_chunk_counter]
            video_chunk_counter_sent = self.sat_last_trans_video_counter_sent
            #self.sat_last_trans_video_counter_sent = 0  # in bytes

            while True:  # download video chunk over mahimahi
                if self.sat_bw[self.mahimahi_ptr_sat] < 0.2:
                    throughput = self.sat_bw[self.mahimahi_ptr_sat - 1] \
                                 * B_IN_MB / BITS_IN_BYTE
                else:
                    throughput = self.sat_bw[self.mahimahi_ptr_sat] \
                                 * B_IN_MB / BITS_IN_BYTE
                duration = self.sat_time[self.mahimahi_ptr_sat] \
                           - self.last_mahimahi_time_sat

                packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                if video_chunk_counter_sent + packet_payload > video_chunk_size_raw:

                    fractional_time = (video_chunk_size_raw - video_chunk_counter_sent) / \
                                      throughput / PACKET_PAYLOAD_PORTION
                    #delay += fractional_time
                    self.last_mahimahi_time_sat += fractional_time
                    assert self.last_mahimahi_time_sat <= self.sat_time[self.mahimahi_ptr_sat]
                    self.sat_video_chunk_counter += 1
                    self.mec_video_remain += 1
                    self.sat_last_trans_video_counter_sent = 0
                    break

                video_chunk_counter_sent += packet_payload
                #delay += duration
                self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat]
                self.mahimahi_ptr_sat += 1

                if self.mahimahi_ptr_sat >= len(self.sat_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr_sat = 1
                    self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat - 1]
                    self.sat_cycle += 1

        timestamp_sat = self.sat_cycle * (self.sat_time[-1] - self.sat_time[0]) + self.last_mahimahi_time_sat - self.sat_time[0]
        timestamp_user = self.user_cycle * (self.user_time[-1] - self.user_time[0]) + self.last_mahimahi_time_user - self.user_time[0]
        if quality > 0:
            if timestamp_sat > user_timestamp_start:
                timestamp_sat += SR_DELAY[quality]
                if timestamp_sat > timestamp_user:
                    self.last_mahimahi_time_user = timestamp_sat % (self.user_time[-1] - self.user_time[0]) +self.user_time[0]
                    self.user_cycle = int(timestamp_sat / (self.user_time[-1] - self.user_time[0]))
                    for i in range(len(self.user_time)):
                        self.mahimahi_ptr_user = i
                        if self.last_mahimahi_time_user < self.user_time[i]:
                            break

            else:
                if user_delay < SR_DELAY[quality]:
                    timestamp_sr = user_timestamp_start + SR_DELAY[quality]
                    self.last_mahimahi_time_user = timestamp_sr % (self.user_time[-1] - self.user_time[0]) + self.user_time[0]
                    self.user_cycle = int(timestamp_sr / (self.user_time[-1] - self.user_time[0]))
                    for i in range(len(self.user_time)):
                        self.mahimahi_ptr_user = i
                        if self.last_mahimahi_time_user < self.user_time[i]:
                            break

        else:
            if timestamp_sat > timestamp_user:
                self.last_mahimahi_time_user = timestamp_sat % (self.user_time[-1] - self.user_time[0]) + self.user_time[0]
                self.user_cycle = int(timestamp_sat / (self.user_time[-1] - self.user_time[0]))
                for i in range(len(self.user_time)):
                    self.mahimahi_ptr_user = i
                    if self.last_mahimahi_time_user < self.user_time[i]:
                        break


        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0) + VIDEO_CHUNCK_LEN


        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.user_time[self.mahimahi_ptr_user] \
                           - self.last_mahimahi_time_user
                if duration > sleep_time:
                    self.last_mahimahi_time_user += sleep_time
                    break
                sleep_time -= duration
                self.last_mahimahi_time_user = self.user_time[self.mahimahi_ptr_user]
                self.mahimahi_ptr_user += 1

                if self.mahimahi_ptr_user >= len(self.user_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr_user = 1
                    self.last_mahimahi_time_user = self.user_time[self.mahimahi_ptr_user - 1]
                    self.user_cycle += 1

        return_buffer_size = self.buffer_size

        # self.video_chunk_counter += 1
        video_chunk_remain = self.video_chunk_num - self.user_video_chunk_counter

        end_of_video = False
        if self.user_video_chunk_counter >= self.video_chunk_num:
            end_of_video = True

            self.sat_video_chunk_counter = 0
            self.user_video_chunk_counter = 0
            self.buffer_size = 0
            self.mec_video_remain = 0
            self.sat_last_trans_video_counter_sent = 0
            self.last_trans_dnn_counter_sent = 0
            self.first_sr_video_counter_sent = 0
            self.last_user_timestamp = 0

            # pick a random trace file
            self.video_idx += 1
            if self.video_idx == len(SR_VIDEO_SIZE_FILE):
                self.video_idx = 0
                self.user_trace_idx += 1
                if self.user_trace_idx >= len(self.all_cooked_user_time):
                    self.user_trace_idx = 0
                self.sat_trace_idx += 1
                if self.sat_trace_idx >= len(self.all_cooked_sat_time):
                    self.sat_trace_idx = 0

            self.sat_time = self.all_cooked_sat_time[self.sat_trace_idx]
            self.sat_bw = self.all_cooked_sat_bw[self.sat_trace_idx]
            self.user_time = self.all_cooked_user_time[self.user_trace_idx]
            self.user_bw = self.all_cooked_user_bw[self.user_trace_idx]
            self.sat_cycle = 0
            self.user_cycle = 0

            # randomize the start point of the video
            # note: trace file starts with time 0
            # self.mahimahi_ptr = np.random.randint(1, min(len(self.sat_bw), len(self.user_bw)))
            self.mahimahi_ptr_sat = 1
            self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat - 1]
            self.mahimahi_ptr_user = self.mahimahi_ptr_sat
            self.last_mahimahi_time_user = self.user_time[self.mahimahi_ptr_user - 1]


            self.video_chunk_num = VIDEO_CHUNK_NUM[self.video_idx]
            self.sr_video_size = {}
            self.hd_video_size = {}
            for bitrate in range(BITRATE_LEVELS):
                self.sr_video_size[bitrate] = []
                self.hd_video_size[bitrate] = []
                with open(SR_VIDEO_SIZE_FILE[self.video_idx] + str(bitrate) + '.log') as f:
                    for line in f:
                        self.sr_video_size[bitrate].append(int(line.split()[0]))

                with open(HD_VIDEO_SIZE_FILE[self.video_idx] + str(bitrate) + '.log') as f:
                    for line in f:
                        self.hd_video_size[bitrate].append(int(line.split()[0]))

        bitrate_mask = self.video_masks[self.video_idx]
        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.sr_video_size[i][self.user_video_chunk_counter])

        return delay, \
               user_delay, \
               sleep_time, \
               return_buffer_size, \
               rebuf, \
               video_chunk_size_sr, \
               next_video_chunk_sizes, \
               end_of_video, \
               video_chunk_remain, \
               return_mec_remain, \
               bitrate_mask