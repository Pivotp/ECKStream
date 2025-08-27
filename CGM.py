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


def cgm(sampled_data, epsilon, mean_value):
    sampled_data = correlated_data(sampled_data, mean_value)
    for col in sampled_data.columns[1:]:
        values = sampled_data[col].values.tolist()
        max_value = max(map(float, values))
        min_value = min(map(float, values))
        a = max_value - min_value
        max_dif = np.max(np.abs(np.diff(values)))
        min_dif = np.min(np.diff(values))
        b = max_dif - min_dif
        sigma = agm.AnalyticGaussian(epsilon, 1.e-5, a)
        if sigma <= 0:
            sigma = 0.1
        noise_values = []
        v_i = 1
        noise_i = np.random.normal(loc=0, scale=sigma)
        per_value = values[0] + noise_i
        noise_values.append(per_value)
        for i in range(1, len(values)):
            r_i = (1 - b) / ((1 - b) ** 2 + v_i)
            sigma_i = float(((1 - r_i) * a + r_i * b) * sigma)
            noise_i = np.random.normal(loc=0, scale=sigma_i) + r_i * noise_i
            per_value = values[i] + noise_i
            v_i = v_i / ((1 - b) ** 2 + v_i)
            noise_values.append(per_value)

        sampled_data.loc[sampled_data.index, col] = noise_values
    return sampled_data


def main():
    min_epsilon, max_epsilon, step_epsilon = 0.5, 5.5, 0.5

    with open("results_3.txt", "a") as f:
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
                cgm_data = cgm(df.copy(), ep, mean_value)

                for node in nodes:
                    orig = df[node].values
                    perturbed = cgm_data[node].values
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
