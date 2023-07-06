import os


USER_COOKED_TRACE_FOLDER = './cooked_traces/user/'
SAT_COOKED_TRACE_FOLDER = './cooked_traces/sat/'
#USER_COOKED_TRACE_FOLDER = './cooked_test_traces/user/'
#SAT_COOKED_TRACE_FOLDER = './cooked_test_traces/sat/'

def load_trace(user_cooked_trace_folder=USER_COOKED_TRACE_FOLDER, sat_cooked_trace_folder=SAT_COOKED_TRACE_FOLDER):
    user_cooked_files = os.listdir(user_cooked_trace_folder)
    sat_cooked_files = os.listdir(sat_cooked_trace_folder)
    user_all_cooked_time = []
    user_all_cooked_bw = []
    user_all_file_names = []
    sat_all_cooked_time = []
    sat_all_cooked_bw = []
    sat_all_file_names = []
    for user_cooked_file in user_cooked_files:
        #print(user_cooked_file)
        file_path = user_cooked_trace_folder + user_cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        user_all_cooked_time.append(cooked_time)
        user_all_cooked_bw.append(cooked_bw)
        user_all_file_names.append(user_cooked_file)

    for sat_cooked_file in sat_cooked_files:
        #print(sat_cooked_file)
        file_path = sat_cooked_trace_folder + sat_cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        sat_all_cooked_time.append(cooked_time)
        sat_all_cooked_bw.append(cooked_bw)
        sat_all_file_names.append(sat_cooked_file)

    #print(sat_cooked_files[744], user_cooked_files[44])
    #print(sat_cooked_files[317], user_cooked_files[9])
    #print(sat_cooked_files[45], user_cooked_files[1])

    return user_all_cooked_time, user_all_cooked_bw, sat_all_cooked_time, sat_all_cooked_bw, \
           user_all_file_names, sat_all_file_names

if __name__ == '__main__':
    load_trace()
