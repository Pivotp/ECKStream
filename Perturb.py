import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def direct_perturb(sampled_data, epsilon): 
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
        else:
            scale = a / (epsilon / 2)
            epsilon_1 = 0
            epsilon_2 = epsilon / 2
        noise_j = np.random.laplace(0, scale)
        noise_data[0] = values[0] + noise_j
        for j in range(1, len(values)):
            if a <= 2 * b:
                scale = a / epsilon_1
                noise_data[j] = values[j] + np.random.laplace(0, scale)
            else:
                scale = b / epsilon_2
                noise_data[j] = (values[j] - values[j - 1]) + np.random.laplace(0, scale) + noise_data[j - 1]

        sampled_data.loc[sampled_data.index, col] = noise_data
    return sampled_data


def main():
    min_epsilon, max_epsilon, step_epsilon = 0.5, 5.5, 0.5
    with open("results_1.txt", "a") as f:
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
            f.write("mean_distance"f"{ep}\t{avg_mean_distance}\n")
            f.write("mean_mae"f"{ep}\t{avg_mae_error}\n")
            f.flush()


main()
