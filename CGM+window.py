import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import agm


def correlated_data(sampled_data, mean_value):
    for col in sampled_data.columns[1:]:
        C = mean_value[col]
        values = sampled_data[col].values.tolist()
        for i in range(1, len(values)):
            if values[i] - values[i - 1] > C:
                values[i] = values[i - 1] + C
            elif values[i] - values[i - 1] < -C:
                values[i] = values[i - 1] - C
            else:
                values[i] = values[i]

        sampled_data.loc[sampled_data.index, col] = values

    return sampled_data


def roll_window_perturb(sampled_data, epsilon, w, mean_value):  # 利用固定的滑动窗口，来对数据进行加噪，其中利用Wiener 滤波器
    sampled_data = correlated_data(sampled_data, mean_value)
    df_sample = sampled_data
    for col in df_sample.columns[1:]:
        window_size = w
        values = df_sample[col].values.tolist()

        num_windows = len(values) // window_size
        for i in range(num_windows):
            clip = []
            noise_data = []
            window_data = values[i * window_size:(i + 1) * window_size]
            max_value = max(window_data)
            min_value = min(window_data)
            for k in range(1, window_size):
                dif = abs(window_data[k] - window_data[k - 1])
                clip.append(dif)
            dif_max = max(clip)
            dif_min = min(clip)
            a = max_value - min_value
            b = dif_max - dif_min
            sigma = agm.AnalyticGaussian(epsilon, 1.e-5, a)
            v_j = 1
            noise_j = np.random.normal(loc=0, scale=sigma)
            noise_value = window_data[0] + noise_j
            noise_data.append(noise_value)

            for j in range(1, window_size):
                r_j = (1 - b) / ((1 - b) ** 2 + v_j)
                sigma_j = float(((1 - r_j) * a + r_j * b) * sigma)
                noise_j = np.random.normal(loc=0, scale=sigma_j) + r_j * noise_j
                noise_value = window_data[j] + noise_j
                noise_data.append(noise_value)
                v_j = v_j / ((1 - b) ** 2 + v_j)

            values[i * window_size:(i + 1) * window_size] = noise_data

        if len(values) % window_size != 0:
            clip = []
            noise_data = []
            window_data = values[num_windows * window_size:]
            max_value = max(window_data)
            min_value = min(window_data)
            if len(window_data) == 1:
                diff = window_data[0]
                clip.append(diff)
            else:
                for k in range(1, len(window_data)):
                    dif = abs(window_data[k] - window_data[k - 1])
                    clip.append(dif)
            dif_max = max(clip)
            dif_min = min(clip)
            a = max_value - min_value
            b = dif_max - dif_min
            sigma = agm.AnalyticGaussian(epsilon, 1.e-5, a)
            v_j = 1
            noise_j = np.random.normal(loc=0, scale=sigma)
            noise_value = window_data[0] + noise_j
            noise_data.append(noise_value)

            for j in range(1, len(window_data)):
                r_j = (1 - b) / ((1 - b) ** 2 + v_j)
                sigma_j = float(((1 - r_j) * a + r_j * b) * sigma)
                noise_j = np.random.normal(loc=0, scale=sigma_j) + r_j * noise_j
                noise_value = window_data[j] + noise_j
                noise_data.append(noise_value)
                v_j = v_j / ((1 - b) ** 2 + v_j)

            values[num_windows * window_size:] = noise_data

        df_sample.loc[df_sample.index, col] = values
    return df_sample


def main():
    min_epsilon, max_epsilon, step_epsilon = 0.5, 5.5, 0.5

    with open("results_4.txt", "a") as f:
        f.write("Epsilon\tMAE\tRMSE\tNMAE\n")
        f.flush()

        for ep in np.arange(min_epsilon, max_epsilon, step_epsilon):
            num_runs = 20
            all_mean_distances = []
            all_mae_error = []
            w = 10

            for run in range(num_runs):
                euclidean_distance = {}
                mae_error = {}
                df = pd.read_csv('Power.csv', low_memory=False)
                nodes = df.columns[1:]
                roll_data = roll_window_perturb(df.copy(), ep, w, mean_value)

                for node in nodes:
                    orig = df[node].values
                    perturbed = roll_data[node].values
                    diff = orig - perturbed
                    mask = ~np.isnan(diff)
                    vaild_count = np.sum(mask)
                    abs_error = np.abs(orig - perturbed)
                    mae_error[node] = np.sum(abs_error[mask]) / vaild_count
                    euclidean_distance[node] = np.sqrt(np.sum((diff[mask]) ** 2))

                mean_mae = sum(mae_error.values()) / len(nodes)
                all_mae_error.append(mean_mae)
                mean_distance = sum(euclidean_distance.values()) / len(euclidean_distance)
                all_mean_distances.append(mean_distance)

            avg_mean_distance = np.mean(all_mean_distances)
            avg_mae_error = np.mean(all_mae_error)
            print(avg_mean_distance)
            print(avg_mae_error)
            f.write("mean_distance"f"{ep}\t{avg_mean_distance}\n")
            f.write("mean_mae"f"{ep}\t{avg_mae_error}\n")

            f.flush()


main()
