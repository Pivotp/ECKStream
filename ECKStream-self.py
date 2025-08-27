import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

fl = pd.read_csv('Power.csv', low_memory=False)

V = fl.columns[fl.columns != 'time'].tolist()


def Event_triggered(epsilon):
    file_path = 'Power.csv'
    weather_file_path = 'Outdoor_weather.csv'

    file = pd.read_csv(file_path, low_memory=False)
    weather_file = pd.read_csv(weather_file_path, low_memory=False)

    file['time'] = pd.to_datetime(file['time'])
    weather_file['time'] = pd.to_datetime(weather_file['time'])

    U = {node: {} for node in V}
    Threshold = {}
    for node in V:
        mean_value = file[node].mean()
        Threshold[node] = mean_value

    print(Threshold)

    pre_trigger_index = {node: 0 for node in V}
    trigger_time_value = {node: 0 for node in V}
    next_val = {node: 0 for node in V}
    initialization = {node: False for node in V}
    trigger_times = {node: 0 for node in V}
    index = {node: 0 for node in V}
    is_trigger = {node: True for node in V}
    x = {node: [] for node in V}

    for index, row in file.sort_values(by='time').iterrows():
        time = row['time']
        index = int(index)

        for node in V:
            current_value = row[node]
            if not initialization[node]:
                trigger_time_value[node] = row[node]
                pre_trigger_index[node] = index
                initialization[node] = True

            x[node].append(current_value)

            arr = np.array(x[node], dtype=float)

            mean = arr.mean()
            std = arr.std(ddof=0)

            Difference = current_value - mean
            abs_value = abs(Difference)

            if abs_value > 1.0 * std:
                trigger_times[node] += 1
                current_time = time

                clip = []
                window_data = []

                for i in range(pre_trigger_index[node], index):
                    value = file.iloc[i][node]
                    window_data.append(value)

                window_size = len(window_data)
                max_value = max(window_data)
                min_value = min(window_data)
                noise_data = [0] * window_size

                if window_size == 1:
                    diff = window_data[0]
                    clip.append(diff)
                else:
                    for k in range(1, window_size):
                        dif = window_data[k] - window_data[k - 1]
                        clip.append(dif)
                dif_max = max(clip)
                dif_min = min(clip)
                information_list = []
                a = max_value - min_value
                b = dif_max - dif_min

                #  the first data in Xi
                if a <= 2 * b:
                    scale = a / epsilon
                    epsilon_1 = epsilon
                    epsilon_2 = 0
                    measure_var = 2 * (a / epsilon) ** 2
                    if window_size == 1:
                        scale = max_value / epsilon
                        measure_var = 2 * (max_value / epsilon) ** 2
                    if scale == 0 and window_size != 1:
                        scale = 1e-5 / epsilon
                        measure_var = 2 * scale ** 2
                else:
                    scale = a / (epsilon / 2)
                    epsilon_1 = 0
                    epsilon_2 = epsilon / 2
                    measure_var = 8 * ((b / epsilon) ** 2)
                    if window_size == 2:
                        measure_var = 2 * ((2 * dif_max) / epsilon) ** 2
                    if scale == 0 and window_size != 2:
                        scale = 1e-5 / epsilon
                        measure_var = 2 * scale ** 2
                    if b == 0 and window_size != 2 and scale != 0:
                        measure_var = 2 * ((2 * dif_max) / epsilon) ** 2
                process_var = 0.05 * measure_var
                noise_j = np.random.laplace(0, scale)
                noise_data[0] = window_data[0] + noise_j

                # other data
                for j in range(1, window_size):
                    if a <= 2 * b:
                        scale = a / epsilon_1
                        if scale == 0:
                            scale = 1e-5 / epsilon_1
                        noise_data[j] = window_data[j] + np.random.laplace(0, scale)
                    else:
                        scale = b / epsilon_2
                        if window_size == 2:
                            dif_max = abs(dif_max)
                            scale = dif_max / epsilon_2
                        if scale == 0 and window_size != 2:
                            scale = 1e-5 / epsilon_2
                        noise_data[j] = (window_data[j] - window_data[j - 1]) + np.random.laplace(0, scale) + noise_data[j - 1]

                '''smooth_data = kalman_filter(noise_data, process_var, measure_var)

                adj = 0
                for k in range(window_size):
                    adj += noise_data[k] - smooth_data[k]

                adjustment = adj / len(noise_data)

                for k in range(window_size):
                    smooth_data[k] = smooth_data[k] + adjustment'''

                for i in range(pre_trigger_index[node], index):

                    m = i - pre_trigger_index[node]
                    n_value = noise_data[m]

                    weather_info_judge = weather_file.loc[weather_file['time'] == file.loc[i]['time']]
                    if weather_info_judge.empty:
                        weather_info = {}
                    else:
                        weather_info = weather_info_judge.to_dict(orient='records')[0]

                    information = {
                        'time': file.iloc[i]['time'],
                        'node': node,
                        'value': n_value,
                        'weather': weather_info
                    }
                    information_list.append(information)

                if current_time not in U[node]:
                    U[node][current_time] = {
                        'information_list': []
                    }
                U[node][current_time]['information_list'].extend(information_list)

                trigger_time_value[node] = current_value
                pre_trigger_index[node] = index
                x[node] = [trigger_time_value[node]]

    for node in V:
        if pre_trigger_index[node] >= len(file):
            continue

        clip = []
        window_data = []

        current_time = file.iloc[pre_trigger_index[node]]['time']

        if pre_trigger_index[node] != len(file):
            for i in range(pre_trigger_index[node], len(file)):
                value = file.iloc[i][node]
                window_data.append(value)

        window_size = len(window_data)
        max_value = max(window_data)
        min_value = min(window_data)
        noise_data = [0] * window_size

        if window_size == 1:
            diff = window_data[0]
            clip.append(diff)
        else:
            for k in range(1, window_size):
                dif = window_data[k] - window_data[k - 1]
                clip.append(dif)
        dif_max = max(clip)
        dif_min = min(clip)
        information_list = []
        a = max_value - min_value
        b = dif_max - dif_min

        #  the first data in Xi
        if a <= 2 * b:
            scale = a / epsilon
            epsilon_1 = epsilon
            epsilon_2 = 0
            measure_var = 2 * (a / epsilon) ** 2
            if window_size == 1:
                scale = max_value / epsilon
                measure_var = 2 * (max_value / epsilon) ** 2
            if scale == 0 and window_size != 1:
                scale = 1e-5 / epsilon
                measure_var = 2 * scale ** 2
        else:
            scale = a / (epsilon / 2)
            epsilon_1 = 0
            epsilon_2 = epsilon / 2
            measure_var = 8 * ((b / epsilon) ** 2)
            if window_size == 2:
                measure_var = 2 * ((2 * dif_max) / epsilon) ** 2
            if scale == 0 and window_size != 2:
                scale = 1e-5 / epsilon
                measure_var = 2 * scale ** 2
            if b == 0 and window_size != 2 and scale != 0:
                measure_var = 2 * ((2 * dif_max) / epsilon) ** 2
        process_var = 0.05 * measure_var
        noise_j = np.random.laplace(0, scale)
        noise_data[0] = window_data[0] + noise_j

        # other data
        for j in range(1, window_size):
            if a <= 2 * b:
                scale = a / epsilon_1
                if scale == 0:
                    scale = 1e-5 / epsilon_1
                noise_data[j] = window_data[j] + np.random.laplace(0, scale)
            else:
                scale = b / epsilon_2
                if window_size == 2:
                    dif_max = abs(dif_max)
                    scale = dif_max / epsilon_2
                if scale == 0 and window_size != 2:
                    scale = 1e-5 / epsilon_2
                noise_data[j] = (window_data[j] - window_data[j - 1]) + np.random.laplace(0, scale) + noise_data[j - 1]

        '''smooth_data = kalman_filter(noise_data, process_var, measure_var)

        adj = 0
        for k in range(window_size):
            adj += noise_data[k] - smooth_data[k]

        adjustment = adj / len(noise_data)

        for k in range(window_size):
            smooth_data[k] = smooth_data[k] + adjustment'''

        for i in range(pre_trigger_index[node], len(file)):
            # print("start2", "i", i)
            m = i - pre_trigger_index[node]
            n_value = noise_data[m]

            weather_info_judge = weather_file.loc[weather_file['time'] == file.loc[i]['time']]
            if weather_info_judge.empty:
                weather_info = {}
            else:
                weather_info = weather_info_judge.to_dict(orient='records')[0]

            information = {
                'time': file.iloc[i]['time'],
                'node': node,
                'value': n_value,
                'weather': weather_info
            }
            information_list.append(information)

        if current_time not in U[node]:
            U[node][current_time] = {
                'information_list': []
            }
        U[node][current_time]['information_list'].extend(information_list)

    return U, Threshold


def differential_bound(C, v_j):
    while True:
        r_j = (1 - 2 * C) / ((1 - 2 * C) ** 2 + v_j)
        if (1 - r_j) + (r_j * 2 * C) >= 0:
            break
        C += 1
    return C


def kalman_filter(noise_data, process_var, measure_var):

    n = len(noise_data)
    smooth_data = np.zeros(n)
    x_hat = noise_data[0]
    P = measure_var

    smooth_data[0] = x_hat

    if n == 1:
        return smooth_data

    for k in range(1, n):

        x_hat_minus = x_hat
        P_minus = P + process_var

        S = P_minus + measure_var
        K = P_minus / S

        z = noise_data[k]
        x_hat = x_hat_minus + K * (z - x_hat_minus)
        P = (1 - K) * P_minus

        smooth_data[k] = x_hat

    return smooth_data


def MSE(df1, df2):
    real_values = list(df1.values())
    noise_values = [df2.get(key) for key in df1.keys()]

    mse = mean_squared_error(real_values, noise_values)

    return mse


def MAE(df1, df2):
    real_value = list(df1.values())
    noise_value = [df2.get(key) for key in df1.keys()]

    mae = mean_absolute_error(real_value, noise_value)

    return mae


def RMSE(M):
    return np.sqrt(M)


def NMAE(df1, A):
    real_value = list(df1.values())
    dif_value = np.max(real_value) - np.min(real_value)

    nmae = A / dif_value

    return nmae


def read_data(U, target_node):
    node_info = []
    for time, data in U[target_node].items():
        if 'information_list' in data:
            for info in data['information_list']:
                if info.get('node') == target_node:
                    node_info.append(info)
    return node_info


def noise_data(U, df):
    euclidean_distance = {}
    mae_error = {}

    for node in V:
        target_node = node
        node_information = read_data(U, target_node)

        n_df = pd.DataFrame([{
            'time': data['time'],
            'node': data['node'],
            'value': data['value'],
            'weather': data['weather']
        } for data in node_information])

        df['time'] = pd.to_datetime(df['time'])
        n_df['time'] = pd.to_datetime(n_df['time'])

        orig_df = df[['time', node]].rename(columns={node: 'orig_value'})
        perturbed_df = n_df[['time', 'value']].rename(columns={'value': 'perturbed_value'})

        merged = pd.merge(orig_df, perturbed_df, on='time', how='inner')

        orig_aligned = merged['orig_value'].values
        perturbed_aligned = merged['perturbed_value'].values
        diff = orig_aligned - perturbed_aligned
        mask = ~np.isnan(diff)
        vaild_count = np.sum(mask)
        abs_error = np.abs(orig_aligned - perturbed_aligned)
        mae_error[node] = np.sum(abs_error[mask]) / vaild_count
        euclidean_distance[node] = np.sqrt(np.sum((diff[mask]) ** 2))

    mean_mae = sum(mae_error.values()) / len(V)
    mean_distance = sum(euclidean_distance.values()) / len(euclidean_distance)
    return mean_distance, mean_mae


def noise_mean(U):
    noise_mean_value = {}
    noise_std_value = {}

    for node in V:
        target_node = node
        node_information = read_data(U, target_node)

        df = pd.DataFrame([{
            'time': data['time'],
            'node': data['node'],
            'value': data['value'],
            'weather': data['weather']
        } for data in node_information])

        node_mean = df.groupby('node')['value'].mean()
        node_std = df.groupby('node')['value'].std()

        noise_mean_value.update(node_mean.to_dict())
        noise_std_value.update(node_std.to_dict())

    return noise_mean_value, noise_std_value


def main():
    min_epsilon, max_epsilon, step_epsilon = 1.5, 5.0, 0.5

    with open("results_single_perturb.txt", "a") as f:
        f.write("Epsilon\tMAE\tRMSE\tNMAE\n")
        f.flush()

        for ep in np.arange(max_epsilon, min_epsilon, -step_epsilon):
            num_runs = 10
            all_mean_distances = []
            all_A_3 = []
            all_A_4 = []
            all_RM_3 = []
            all_RM_4 = []
            all_NA_3 = []
            all_NA_4 = []
            all_mae_error = []

            for run in range(num_runs):
                std_value = {}
                df = pd.read_csv('Power.csv')
                for node in V:
                    std_value[node] = df[node].std()

                perturb_data, Threshold = Event_triggered(ep)
                mean_distance, mean_mae = noise_data(perturb_data, df)
                all_mae_error.append(mean_mae)
                all_mean_distances.append(mean_distance)
                noise_mean_value, noise_std_value = noise_mean(perturb_data)

                print(mean_distance)
                print(mean_mae)

                M_3 = MSE(Threshold, noise_mean_value)
                M_4 = MSE(std_value, noise_std_value)

                A_3 = MAE(Threshold, noise_mean_value)
                A_4 = MAE(std_value, noise_std_value)

                RM_3 = RMSE(M_3)
                RM_4 = RMSE(M_4)

                NA_3 = NMAE(Threshold, A_3)
                NA_4 = NMAE(std_value, A_4)

                all_A_3.append(A_3)
                all_A_4.append(A_4)
                all_RM_3.append(RM_3)
                all_RM_4.append(RM_4)
                all_NA_3.append(NA_3)
                all_NA_4.append(NA_4)

            avg_mean_distance = np.mean(all_mean_distances)
            avg_mae_error = np.mean(all_mae_error)
            print(avg_mean_distance)
            print(avg_mae_error)

            avg_A_3 = np.mean(all_A_3)
            avg_A_4 = np.mean(all_A_4)

            avg_RM_3 = np.mean(all_RM_3)
            avg_RM_4 = np.mean(all_RM_4)

            avg_NA_3 = np.mean(all_NA_3)
            avg_NA_4 = np.mean(all_NA_4)

            print("MAE", avg_A_3, ep)
            print("MAE", avg_A_4, ep)

            print("RMSE", avg_RM_3, ep)
            print("RMSE", avg_RM_4, ep)

            print("NMAE", avg_NA_3, ep)
            print("NMAE", avg_NA_4, ep)

            f.write("mean_distance"f"{ep}\t{avg_mean_distance}\n")
            f.write("mean_mae"f"{ep}\t{avg_mae_error}\n")
            f.write("mean_value"f"{ep}\t{avg_A_3}\t{avg_RM_3}\t{avg_NA_3}\n")
            f.write("std_value"f"{ep}\t{avg_A_4}\t{avg_RM_4}\t{avg_NA_4}\n")

            f.flush()


main()
