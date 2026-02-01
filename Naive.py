import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def direct_perturb(sampled_data, epsilon):
    float_cols = sampled_data.columns[1:]
    sampled_data[float_cols] = sampled_data[float_cols].astype('float64')
    for col in sampled_data.columns[1:]:
        values = sampled_data[col].values.tolist()
        max_value = max(map(float, values))
        min_value = min(map(float, values))
        a = np.linalg.norm(max_value - min_value)
        delta = 2 * 1e-6
        sigma = (a / epsilon) * np.sqrt(2 * np.log(1.25 / delta))
        for i in range(0, len(values)):
            values[i] += np.random.normal(0, sigma)
        sampled_data.loc[sampled_data.index, col] = values

    return sampled_data


def main():
    min_epsilon, max_epsilon, step_epsilon = 0.5, 5.5, 0.5
    file = pd.read_csv('Power.csv')

    mean_value = {}
    for node in file.columns[1:]:
        mean_value[node] = file[node].mean()

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

                direct_data = direct_perturb(df.copy(), ep, mean_value)
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

