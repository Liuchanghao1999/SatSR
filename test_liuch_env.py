# -*-coding:utf-8-*-
import numpy as np

#MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 4
TOTAL_VIDEO_CHUNCK = 85
BUFFER_THRESH = 60.0  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 0.5  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 0.08  # millisec
SR_DELAY = {0:0, 1:0.88, 2:1.35, 3:2.05} # sec
BANDWIDTH = {0:272569, 1:809338, 2:1511207, 3:2872440}
MBT = 4.0
DNN_SIZE = 4302035
#PACKET_SIZE = 1500  # bytes
RESEVOIR = 10.0  # BB
CUSHION = 10.0  # BB
SMOOTH_PENALTY = 1
HD_REWARD = [1, 4, 10, 15]
VIDEO_BIT_RATE = [300,600,1200,2400]  # Kbps
M_IN_K = 1000.0

SR_VIDEO_SIZE_FILE = './iPHONE_SR_size_'
HD_VIDEO_SIZE_FILE = './iPHONE_HD_size_'


class Environment:
    def __init__(self, all_cooked_user_time, all_cooked_user_bw, all_cooked_sat_time, all_cooked_sat_bw, random_seed=RANDOM_SEED):
        assert len(all_cooked_user_time) == len(all_cooked_user_bw)
        assert len(all_cooked_sat_time) == len(all_cooked_sat_bw)

        np.random.seed(random_seed)

        self.all_cooked_user_time = all_cooked_user_time
        self.all_cooked_user_bw = all_cooked_user_bw
        self.all_cooked_sat_time = all_cooked_sat_time
        self.all_cooked_sat_bw = all_cooked_sat_bw

        self.sat_video_chunk_counter = 0
        self.user_video_chunk_counter = 0
        self.buffer_size = 0
        self.mec_video_remain = 0
        self.sat_last_trans_video_counter_sent = 0
        self.last_trans_dnn_counter_sent = 0
        self.first_sr_video_counter_sent = 0
        self.last_user_timestamp = 0

        # pick a random trace file
        self.sat_trace_idx = np.random.randint(len(self.all_cooked_sat_time))
        self.user_trace_idx = np.random.randint(len(self.all_cooked_user_time))
        self.sat_time = self.all_cooked_sat_time[self.sat_trace_idx]
        self.sat_bw = self.all_cooked_sat_bw[self.sat_trace_idx]
        self.user_time = self.all_cooked_user_time[self.user_trace_idx]
        self.user_bw = self.all_cooked_user_bw[self.user_trace_idx]
        self.sat_cycle = 0
        self.user_cycle = 0

        # randomize the start point of the trace
        # note: trace file starts with time 0
        # self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        #self.mahimahi_ptr = np.random.randint(1, min(len(self.sat_bw), len(self.user_bw)))
        self.mahimahi_ptr_sat = 1
        self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat - 1]
        self.mahimahi_ptr_user = self.mahimahi_ptr_sat
        self.last_mahimahi_time_user = self.user_time[self.mahimahi_ptr_user - 1]

        self.raw_video_size = {}  # in bytes
        self.sr_video_size = {}
        self.hd_video_size = {}
        for bitrate in range(BITRATE_LEVELS):
            self.raw_video_size[bitrate] = []
            self.sr_video_size[bitrate] = []
            self.hd_video_size[bitrate] = []
            with open(SR_VIDEO_SIZE_FILE + str(bitrate) + '.log') as f:
                for line in f:
                    self.raw_video_size[bitrate].append(int(line.split()[0]))
                    self.sr_video_size[bitrate].append(int(line.split()[0]))

            with open(HD_VIDEO_SIZE_FILE + str(bitrate) + '.log') as f:
                for line in f:
                    self.hd_video_size[bitrate].append(int(line.split()[0]))

    def early_stage_trans(self, first_quality):
        assert self.sat_video_chunk_counter == 0
        assert self.user_video_chunk_counter == 0

        video_chunk_size_hd = 0
        quality = first_quality
        last_quality = 0
        next_quality = 0
        quality_set = []
        delay = 0.0
        user_delay = 0.0
        rebuf = 0.0
        quality_statistics = [0, 0, 0, 0]
        reward_set = []
        rebuf_set = []
        downloads = 0
        #print('sat_trace_idx: {}; user_trace_idx: {}'.format(self.sat_trace_idx, self.user_trace_idx))

        while self.last_trans_dnn_counter_sent < DNN_SIZE:
            # 卫星链路下载请求视频
            user_rate = []
            delay = 0.0
            user_delay = 0.0
            assert self.sat_video_chunk_counter == self.sat_video_chunk_counter
            # print(quality, self.user_video_chunk_counter)
            video_chunk_size_hd = self.hd_video_size[quality][self.user_video_chunk_counter]
            video_chunk_counter_sent = self.sat_last_trans_video_counter_sent = 0

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
                    # delay += fractional_time
                    self.last_mahimahi_time_sat += fractional_time
                    assert (self.last_mahimahi_time_sat <= self.sat_time[self.mahimahi_ptr_sat])
                    self.mec_video_remain += 1
                    self.sat_last_trans_video_counter_sent = 0
                    assert self.sat_video_chunk_counter <= TOTAL_VIDEO_CHUNCK
                    self.sat_video_chunk_counter += 1
                    break

                video_chunk_counter_sent += packet_payload
                # delay += duration
                self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat]
                self.mahimahi_ptr_sat += 1

                if self.mahimahi_ptr_sat >= len(self.sat_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr_sat = 1
                    self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat - 1]
                    self.sat_cycle += 1

            timestamp_sat = self.sat_cycle * (self.sat_time[-1] - self.sat_time[0]) + self.last_mahimahi_time_sat - \
                            self.sat_time[0]
            timestamp_user = self.user_cycle * (self.user_time[-1] - self.user_time[0]) + self.last_mahimahi_time_user - \
                             self.user_time[0]

            assert timestamp_user <= timestamp_sat
            self.last_mahimahi_time_user = timestamp_sat % (self.user_time[-1] - self.user_time[0]) + self.user_time[0]
            self.user_cycle = int(timestamp_sat / (self.user_time[-1] - self.user_time[0]))
            for i in range(len(self.user_time)):
                self.mahimahi_ptr_user = i
                if self.last_mahimahi_time_user < self.user_time[i]:
                    break

            assert self.sat_video_chunk_counter > self.user_video_chunk_counter
            video_chunk_counter_sent_hd = 0

            downloads += video_chunk_size_hd

            while True:  # download HD-video chunk over user-link
                throughput = self.user_bw[self.mahimahi_ptr_user] \
                             * B_IN_MB / BITS_IN_BYTE
                duration = self.user_time[self.mahimahi_ptr_user] \
                           - self.last_mahimahi_time_user

                user_rate.append(throughput)

                packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                if video_chunk_counter_sent_hd + packet_payload > video_chunk_size_hd:
                    fractional_time = (video_chunk_size_hd - video_chunk_counter_sent_hd) / \
                                      throughput / PACKET_PAYLOAD_PORTION
                    # delay += fractional_time
                    user_delay += fractional_time
                    self.last_mahimahi_time_user += fractional_time
                    self.mec_video_remain -= 1
                    self.user_video_chunk_counter += 1
                    now_user_timestamp = self.user_cycle * (
                                self.user_time[-1] - self.user_time[0]) + self.last_mahimahi_time_user - self.user_time[
                                             0]
                    delay = now_user_timestamp - self.last_user_timestamp
                    assert delay > 0.0
                    self.last_user_timestamp = now_user_timestamp
                    assert self.last_mahimahi_time_user <= self.user_time[self.mahimahi_ptr_user]
                    break

                video_chunk_counter_sent_hd += packet_payload
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


            rebuf = np.maximum(delay - self.buffer_size, 0.0)
            self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

            timestamp_user = self.user_cycle * (self.user_time[-1] - self.user_time[0]) + self.last_mahimahi_time_user - \
                             self.user_time[0]
            assert timestamp_user > timestamp_sat
            self.new_last_mahimahi_time_sat = timestamp_user % (self.sat_time[-1] - self.sat_time[0]) + self.sat_time[0]
            self.new_sat_cycle = int(timestamp_user / (self.sat_time[-1] - self.sat_time[0]))
            for i in range(len(self.sat_time)):
                self.new_mahimahi_ptr_sat = i
                if self.new_last_mahimahi_time_sat < self.sat_time[i]:
                    break

            # if self.sat_cycle < self.new_sat_cycle:
            while self.sat_cycle < self.new_sat_cycle:
                dnn_counter_sent = self.last_trans_dnn_counter_sent
                # dnn_chunk_size = self.raw_video_size[0][self.sat_video_chunk_counter]

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
                    # self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat]
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
                    # 传输视频块
                    while self.sat_cycle < self.new_sat_cycle:
                        video_chunk_counter_sent = self.sat_last_trans_video_counter_sent
                        video_chunk_size_raw = self.raw_video_size[0][self.sat_video_chunk_counter]

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
                                    timestamp_sr = self.sat_cycle * (
                                                self.sat_time[-1] - self.sat_time[0]) + self.last_mahimahi_time_sat - \
                                                   self.sat_time[0]
                                    if quality > 0:
                                        timestamp_sr += SR_DELAY[quality]
                                assert self.sat_video_chunk_counter <= TOTAL_VIDEO_CHUNCK
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

            # assert self.mahimahi_ptr < self.new_mahimahi_ptr

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
                            self.last_trans_dnn_counter_sent = dnn_counter_sent
                            break

                if self.last_trans_dnn_counter_sent == DNN_SIZE:
                    while self.mahimahi_ptr_sat < self.new_mahimahi_ptr_sat:
                        video_chunk_counter_sent = self.sat_last_trans_video_counter_sent
                        video_chunk_size_raw = self.raw_video_size[0][self.sat_video_chunk_counter]

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
                                    timestamp_sr = self.sat_cycle * (
                                                self.sat_time[-1] - self.sat_time[0]) + self.last_mahimahi_time_sat - \
                                                   self.sat_time[0]
                                    if quality > 0:
                                        timestamp_sr += SR_DELAY[quality]
                                assert self.sat_video_chunk_counter <= TOTAL_VIDEO_CHUNCK
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
                    self.sat_last_trans_video_counter_sent = throughput * (
                                self.new_last_mahimahi_time_sat - fractional_time) * PACKET_PAYLOAD_PORTION

                else:
                    dnn_counter_sent += packet_payload
                    self.last_trans_dnn_counter_sent = dnn_counter_sent

                self.last_mahimahi_time_sat = self.new_last_mahimahi_time_sat

            self.buffer_size += VIDEO_CHUNCK_LEN

            if self.last_trans_dnn_counter_sent == DNN_SIZE:
                break

            # -- linear reward --
            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[quality] / M_IN_K \
                    - 4.3 * rebuf \
                    - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[quality] -
                                              VIDEO_BIT_RATE[last_quality]) / M_IN_K

            # -- log scale reward --
            #log_quality = np.log(VIDEO_BIT_RATE[quality] / float(VIDEO_BIT_RATE[0]))
            #log_last_quality = np.log(VIDEO_BIT_RATE[last_quality] / float(VIDEO_BIT_RATE[0]))

            #reward = log_quality \
            #        - 2.66 * rebuf \
            #        - SMOOTH_PENALTY * np.abs(log_quality - log_last_quality)

            # -- HD reward --
            #reward = HD_REWARD[quality] \
            #         - 8.0 * rebuf \
            #         - SMOOTH_PENALTY * np.abs(HD_REWARD[quality] - HD_REWARD[last_quality])

            reward_set.append(reward)
            rebuf_set.append(rebuf)

            #with open(log_path, 'w+') as log_file:
            #    line = 'Early-Stage:' + '\t' + 'BitRate: ' + str(quality) + '\t' \
            #       + 'BufferSize: ' + str(self.buffer_size) + '\t' + 'ReBuffer: ' + str(rebuf) + '\t' \
            #       + 'Delay: ' + str(delay) + '\t' + 'Reward: ' + str(reward) + '\n'
            #    log_file.write(line.encode())
            #    log_file.flush()


            quality_statistics[quality] += 1

            if quality == last_quality:
                rate = sum(user_rate) / len(user_rate)
                bandwidth = rate * self.buffer_size / MBT
                next_quality = 0
                for i in range(4):
                    if bandwidth >= BANDWIDTH[i]:
                        next_quality = i
                    else:
                        break

                if self.buffer_size > RESEVOIR and quality < 3:
                    next_quality += 1

                if self.buffer_size >= RESEVOIR + CUSHION:
                    next_quality = 3

            else:
                next_quality = quality

            last_quality = quality
            quality = next_quality
            quality_set.append(quality)

            # print('Train:  Video_idx: {}; DNN_download: {}; Bandwidth: {}; Buffer: {}; Rebuf: {}; Quality: {}'.format(self.user_video_chunk_counter, self.last_trans_dnn_counter_sent, bandwidth, self.buffer_size, rebuf, quality))

        return_mec_video_remain = self.mec_video_remain

        if self.mec_video_remain == 0:
            video_chunk_counter_sent = self.sat_last_trans_video_counter_sent
            video_chunk_size_raw = self.raw_video_size[0][self.sat_video_chunk_counter]

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
                        timestamp_sr = self.sat_cycle * (
                                    self.sat_time[-1] - self.sat_time[0]) + self.last_mahimahi_time_sat - self.sat_time[
                                           0]
                        if quality > 0:
                            timestamp_sr += SR_DELAY[quality]
                    assert self.sat_video_chunk_counter < TOTAL_VIDEO_CHUNCK
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

        if timestamp_sr > self.user_cycle * (self.user_time[-1] - self.user_time[0]) + self.last_mahimahi_time_user - \
                self.user_time[0]:
            self.last_mahimahi_time_user = timestamp_sr % (self.user_time[-1] - self.user_time[0]) + self.user_time[0]
            self.user_cycle = int(timestamp_sr / (self.user_time[-1] - self.user_time[0]))
            for i in range(len(self.user_time)):
                self.mahimahi_ptr_user = i
                if self.last_mahimahi_time_user < self.user_time[i]:
                    break

        #return_reward = sum(reward_set)
        return_buffer_size = self.buffer_size
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.user_video_chunk_counter
        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.sr_video_size[i][self.user_video_chunk_counter])

        assert self.user_video_chunk_counter < TOTAL_VIDEO_CHUNCK
        end_of_video = False

        #ave_quality = sum(quality_set) / len(quality_set)
        #print('Train:  Video_idx: {}; Ave_quality: {}'.format(self.user_video_chunk_counter, ave_quality))
        #print(reward_set, quality_statistics)

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
               rebuf_set, \
               downloads, \
               quality_statistics


    def get_video_chunk(self, quality):

        assert quality >= 0
        assert quality < BITRATE_LEVELS

        delay = 0.0
        user_delay = 0.0

        delay += LINK_RTT  # ABR decision delay

        # user_link trans SR-video while do SR
        assert self.sat_video_chunk_counter > self.user_video_chunk_counter
        video_chunk_counter_sent_user = 0  # in bytes
        user_timestamp_start = self.user_cycle * (
                    self.user_time[-1] - self.user_time[0]) + self.last_mahimahi_time_user - self.user_time[0]
        video_chunk_size_sr = self.sr_video_size[quality][self.user_video_chunk_counter]

        return_bandwidth_save = video_chunk_size_sr - self.sr_video_size[0][self.user_video_chunk_counter]

        while True:  # download SR-video chunk over mahimahi
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
                now_user_timestamp = self.user_cycle * (
                            self.user_time[-1] - self.user_time[0]) + self.last_mahimahi_time_user - self.user_time[0]
                delay = now_user_timestamp - self.last_user_timestamp
                # print(now_user_timestamp, self.last_user_timestamp)
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

        return_mec_remain = self.mec_video_remain

        # Do SR while user_link trans SR-video
        assert self.mec_video_remain >= 0
        if self.mec_video_remain == 0 and self.sat_video_chunk_counter <= TOTAL_VIDEO_CHUNCK:
            video_chunk_size_raw = self.raw_video_size[0][self.sat_video_chunk_counter]
            video_chunk_counter_sent = self.sat_last_trans_video_counter_sent
            # self.sat_last_trans_video_counter_sent = 0  # in bytes

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
                    # delay += fractional_time
                    self.last_mahimahi_time_sat += fractional_time
                    assert self.last_mahimahi_time_sat <= self.sat_time[self.mahimahi_ptr_sat]
                    # assert self.sat_video_chunk_counter < TOTAL_VIDEO_CHUNCK
                    self.sat_video_chunk_counter += 1
                    self.mec_video_remain += 1
                    self.sat_last_trans_video_counter_sent = 0
                    break

                video_chunk_counter_sent += packet_payload
                # delay += duration
                self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat]
                self.mahimahi_ptr_sat += 1

                if self.mahimahi_ptr_sat >= len(self.sat_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr_sat = 1
                    self.last_mahimahi_time_sat = self.sat_time[self.mahimahi_ptr_sat - 1]
                    self.sat_cycle += 1

        if quality > 0:
            timestamp_sat = self.sat_cycle * (self.sat_time[-1] - self.sat_time[0]) + self.last_mahimahi_time_sat - \
                            self.sat_time[0]
            if timestamp_sat > user_timestamp_start:
                timestamp_sat += SR_DELAY[quality]
                if timestamp_sat > self.user_cycle * (
                        self.user_time[-1] - self.user_time[0]) + self.last_mahimahi_time_user - self.user_time[0]:
                    self.last_mahimahi_time_user = timestamp_sat % (self.user_time[-1] - self.user_time[0]) + \
                                                   self.user_time[0]
                    self.user_cycle = int(timestamp_sat / (self.user_time[-1] - self.user_time[0]))
                    for i in range(len(self.user_time)):
                        self.mahimahi_ptr_user = i
                        if self.last_mahimahi_time_user < self.user_time[i]:
                            break

            else:
                if user_delay < SR_DELAY[quality]:
                    timestamp_sr = user_timestamp_start + SR_DELAY[quality]
                    self.last_mahimahi_time_user = timestamp_sr % (self.user_time[-1] - self.user_time[0]) + \
                                                   self.user_time[0]
                    self.user_cycle = int(timestamp_sr / (self.user_time[-1] - self.user_time[0]))
                    for i in range(len(self.user_time)):
                        self.mahimahi_ptr_user = i
                        if self.last_mahimahi_time_user < self.user_time[i]:
                            break

        else:
            timestamp_sat = self.sat_cycle * (self.sat_time[-1] - self.sat_time[0]) + self.last_mahimahi_time_sat - \
                            self.sat_time[0]
            if timestamp_sat > self.user_cycle * (
                    self.user_time[-1] - self.user_time[0]) + self.last_mahimahi_time_user - self.user_time[0]:
                self.last_mahimahi_time_user = timestamp_sat % (self.user_time[-1] - self.user_time[0]) + \
                                               self.user_time[0]
                self.user_cycle = int(timestamp_sat / (self.user_time[-1] - self.user_time[0]))
                for i in range(len(self.user_time)):
                    self.mahimahi_ptr_user = i
                    if self.last_mahimahi_time_user < self.user_time[i]:
                        break

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0) + VIDEO_CHUNCK_LEN

        return_buffer_size = self.buffer_size

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

        # self.video_chunk_counter += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.user_video_chunk_counter

        end_of_video = False
        if self.user_video_chunk_counter > TOTAL_VIDEO_CHUNCK:
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
            self.sat_trace_idx = np.random.randint(len(self.all_cooked_sat_time))
            self.user_trace_idx = np.random.randint(len(self.all_cooked_user_time))
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
               return_bandwidth_save, \
               return_mec_remain