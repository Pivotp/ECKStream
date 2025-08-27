import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import agm


def direct_perturb(sampled_data, epsilon):  # 直接扰动
    for col in sampled_data.columns[1:]:
        values = sampled_data[col].values.tolist()
        noise_data = [0] * len(values)
        max_value = max(map(float, values))
        min_value = min(map(float, values))
        a = max_value - min_value
        max_dif = np.max(np.diff(values))
        min_dif = np.min(np.diff(values))
        b = max_dif - min_dif
        if a <= 2 * b:
            scale = a / epsilon
            epsilon_1 = epsilon
            epsilon_2 = 0
            measure_var = 2 * (a / epsilon) ** 2
        else:
            scale = a / (epsilon / 2)
            epsilon_1 = 0
            epsilon_2 = epsilon / 2
            measure_var = 8 * ((b / epsilon) ** 2)
        process_var = 0.05 * measure_var
        noise_j = np.random.laplace(0, scale)
        noise_data[0] = values[0] + noise_j
        for j in range(1, len(values)):
            if a <= 2 * b:
                scale = a / epsilon_1
                noise_data[j] = values[j] + np.random.laplace(0, scale)
            else:
                scale = b / epsilon_2
                noise_data[j] = (values[j] - values[j - 1]) + np.random.laplace(0, scale) + noise_data[j - 1]
        smooth_data = kalman_filter(noise_data, process_var, measure_var)
        adj = 0
        for k in range(len(values)):
            adj += noise_data[k] - smooth_data[k]
        adjustment = adj / len(noise_data)
        for k in range(len(values)):
            smooth_data[k] = smooth_data[k] + adjustment
        sampled_data.loc[sampled_data.index, col] = smooth_data
    return sampled_data


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


def main():
    min_epsilon, max_epsilon, step_epsilon = 0.5, 5.5, 0.5

    with open("results_perturb.txt", "a") as f:
        f.write("Epsilon\tMAE\tRMSE\tNMAE\n")
        f.flush()

        for ep in np.arange(min_epsilon, max_epsilon, step_epsilon):
            num_runs = 20
            all_mean_distances = []
            all_mae_error = []

            for run in range(num_runs):
                euclidean_distance = {}
                mae_error = {}
                df = pd.read_csv('Power.csv', low_memory=False)
                nodes = df.columns[1:]
                direct_data = direct_perturb(df.copy(), ep)

                for node in nodes:
                    orig = df[node].values
                    perturbed = direct_data[node].values
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
