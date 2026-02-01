import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

fl = pd.read_csv('Power.csv', low_memory=False)
V = fl.columns[fl.columns != 'time'].tolist()


def Event_triggered(epsilons, kt):
    file_path = 'Power.csv'
    weather_file_path = 'Outdoor_weather.csv'
    file = pd.read_csv(file_path, low_memory=False)
    weather_file = pd.read_csv(weather_file_path, low_memory=False)
    file['time'] = pd.to_datetime(file['time'])
    weather_file['time'] = pd.to_datetime(weather_file['time'])
    U = {node: {} for node in V}
    Delta = {node: 0 for node in V}
    Threshold = {}
    for node in V:
        node_data = file[node]
        max_node = max(node_data)
        min_node = min(node_data)
        Delta[node] = max_node - min_node
        mean_value = node_data.mean()
        Threshold[node] = mean_value
    pre_trigger_index = {node: 0 for node in V}
    trigger_time_value = {node: 0 for node in V}
    initialization = {node: False for node in V}
    trigger_times = {node: 0 for node in V}
    x = {node: [] for node in V}
    ep = epsilons * 0.2
    epsilon = epsilons * 0.8
    upper = {node: 0 for node in V}
    lower = {node: 0 for node in V}

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
            delta = 1e-6
            sigma = (Delta[node] / ep) * np.sqrt(2 * np.log(1.25 / delta))
            noise = np.random.normal(0, sigma)
            if (abs_value - kt * std) + noise > 0:

                trigger_times[node] += 1
                current_time = time
                upper_bound = mean + kt * std
                lower_bound = mean - kt * std
                upper[node] = upper_bound
                lower[node] = lower_bound
                clip = []
                window_data = []

                if pre_trigger_index[node] == 0 and index == 0:
                    value = file.iloc[0][node]
                    window_data.append(value)
                else:
                    for i in range(pre_trigger_index[node], index):
                        value = file.iloc[i][node]
                        window_data.append(value)

                window_size = len(window_data)
                max_value = max(window_data)
                min_value = min(window_data)
                noise_data = [0] * window_size

                ls_Independent0 = max_value - min_value
                ls_Independent1 = max(upper_bound - min_value, max_value - lower_bound)
                ls_Independent2 = upper_bound - lower_bound
                delta = 1e-6
                beta = epsilon / (2 * math.log(1 / delta))

                a = max(ls_Independent0,
                        ls_Independent1 * math.exp(-beta),
                        ls_Independent2 * math.exp(-2 * beta))

                alpha = epsilon / 2
                scale = a / alpha
                measure_var = 2 * (a / alpha) ** 2
                if window_size == 1:
                    scale = max_value / alpha
                    measure_var = 2 * (max_value / alpha) ** 2
                    if max_value == 0:
                        scale = 1e-5 / alpha
                        measure_var = 2 * scale ** 2
                if scale == 0 and window_size != 1:
                    scale = 1e-5 / alpha
                    measure_var = 2 * scale ** 2
                process_var = 0.05 * measure_var
                noise_j = np.random.laplace(0, scale)
                noise_data[0] = window_data[0] + noise_j
                for j in range(1, window_size):
                    scale = a / alpha
                    if scale == 0:
                        scale = 1e-5 / alpha
                    noise_data[j] = window_data[j] + np.random.laplace(0, scale)

                smooth_data = kalman_filter(noise_data, process_var, measure_var)
                adj = 0
                for k in range(window_size):
                    adj += noise_data[k] - smooth_data[k]
                adjustment = adj / len(noise_data)
                for k in range(window_size):
                    smooth_data[k] = smooth_data[k] + adjustment

                for i in range(pre_trigger_index[node], index):
                    m = i - pre_trigger_index[node]
                    n_value = smooth_data[m]

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
        upper_bound = 0
        lower_bound = 0
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

        if max_value < lower[node]:
            upper_bound = lower[node]
            lower_bound = min_value
        elif min_value > upper[node]:
            lower_bound = upper[node]
            upper_bound = max_value

        ls_Independent0 = max_value - min_value
        ls_Independent1 = max(upper_bound - min_value, max_value - lower_bound)
        ls_Independent2 = upper_bound - lower_bound
        delta = 1e-6
        beta = epsilon / (2 * math.log(1 / delta))
        a = max(ls_Independent0,
                ls_Independent1 * math.exp(-beta),
                ls_Independent2 * math.exp(-2 * beta))

        alpha = epsilon / 2
        scale = a / alpha
        measure_var = 2 * (a / alpha) ** 2
        if window_size == 1:
            scale = max_value / alpha
            measure_var = 2 * (max_value / alpha) ** 2
            if max_value == 0:
                scale = 1e-5 / alpha
                measure_var = 2 * scale ** 2
        if scale == 0 and window_size != 1:
            scale = 1e-5 / alpha
            measure_var = 2 * scale ** 2
        process_var = 0.05 * measure_var
        noise_j = np.random.laplace(0, scale)
        noise_data[0] = window_data[0] + noise_j
        for j in range(1, window_size):
            scale = a / alpha
            if scale == 0:
                scale = 1e-5 / alpha
            noise_data[j] = window_data[j] + np.random.laplace(0, scale)

        smooth_data = kalman_filter(noise_data, process_var, measure_var)
        adj = 0
        for k in range(window_size):
            adj += noise_data[k] - smooth_data[k]
        adjustment = adj / len(noise_data)
        for k in range(window_size):
            smooth_data[k] = smooth_data[k] + adjustment

        for i in range(pre_trigger_index[node], len(file)):
            # print("start2", "i", i)
            m = i - pre_trigger_index[node]
            n_value = smooth_data[m]

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
        P_minus = P + max(process_var, 1e-5)
        S = P_minus + max(measure_var, 1e-5)
        K = P_minus / S
        z = noise_data[k]
        x_hat = x_hat_minus + K * (z - x_hat_minus)
        P = (1 - K) * P_minus
        smooth_data[k] = x_hat

    return smooth_data


def compute_smooth_sensitivity(current_sen, sen, max_k, beta):
    if len(sen) == 0:
        return current_sen
    actual_k = min(max_k, len(sen))
    smooth_sens = current_sen
    for k in range(1, actual_k + 1):
        historical_sen = sen[k - 1]
        decayed_sen = historical_sen * math.exp(-beta * k)
        smooth_sens = max(smooth_sens, decayed_sen)
    return smooth_sens


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


def main():
    min_kt, max_kt, step_epsilon = 1.0, 5.5, 1.0
    ep = 1.0

    with open("results_single_threshold.txt", "a") as f:
        f.write("-----------------------\n")
        f.flush()

        for kt in np.arange(max_kt, min_kt, -step_epsilon):
            num_runs = 20
            all_mean_distances = []
            all_mae_error = []

            for run in range(num_runs):
                df = pd.read_csv('Power.csv')
                perturb_data, Threshold = Event_triggered(ep, kt)

                mean_distance, mean_mae = noise_data(perturb_data, df)
                all_mae_error.append(mean_mae)
                all_mean_distances.append(mean_distance)
            avg_mean_distance = np.mean(all_mean_distances)
            avg_mae_error = np.mean(all_mae_error)
            f.write("mean_distance"f"{kt}\t{avg_mean_distance}\n")
            f.write("mean_mae"f"{kt}\t{avg_mae_error}\n")
            f.flush()


main()

