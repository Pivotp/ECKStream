import numpy as np
import common_tools
import sensitivity_calc
import pandas as pd


def whether_group(groupi, old_avg, new_data, tau, eps_group, sensitivity_):
    total_num = len(groupi)
    # print(total_num)
    new_avg = (old_avg * total_num + new_data) / (total_num + 1)

    dev = 0
    for i in range(total_num):
        dev += abs(groupi[i] - new_avg)

    dev += abs(new_data - new_avg)

    # print(dev / total_num)

    if (dev / total_num) + common_tools.add_noise(sensitivity_ / total_num, eps_group / 4,
                                                  1) > tau + common_tools.add_noise(
        sensitivity_ / total_num, eps_group / 2, 1):
        return 0, new_avg
    else:
        return 1, new_avg


# ------------orginial pegasus--------------
def pegasus_nodelay(ex, domain_low, domain_high, eps, tau, flag=0, interval_=5, num_=100):
    total_time = len(ex)
    dim = len(ex[0])
    eps_group = eps / 5
    eps_pub = eps - eps_group
    published_result = []

    i = 0
    group_ = []
    group_index = []
    group_noisy = []
    avg_ = 0

    whether_update = 0
    sensitivity_ = domain_high

    for i in range(total_time):
        if whether_update == 0:
            eps_pub = eps - eps_group

        if i % num_ == 0:
            if (i / num_) % interval_ == 0 and flag == 1 and (i + 1) * num_ <= total_time:
                eps_s = eps_pub / 2
                eps_pub = eps_pub - eps_s

                data_sens = np.zeros(num_, dtype=int)
                cc = 0
                for qq in range(i * num_, (i + 1) * num_):
                    data_sens[cc] = ex[qq][0]
                    cc += 1

                sensitivity_ = sensitivity_calc.quality_func(data_sens, domain_low, domain_high, interval_,
                                                             eps_s)
                whether_update = 1
            else:
                whether_update = 0

        if len(group_) == 0:
            group_.append(ex[i][0])
            avg_ = ex[i][0]
            group_index.append(i)
            if ex[i][0] > sensitivity_:
                noisy_result = sensitivity_ + common_tools.add_noise(sensitivity_, eps_pub, dim)
            else:
                noisy_result = ex[i][0] + common_tools.add_noise(sensitivity_, eps_pub, dim)

            published_result.append(noisy_result)
            group_noisy.append(noisy_result)

        else:
            ifadd_, newavg = whether_group(group_, avg_, ex[i][0], tau, eps_group, sensitivity_)
            if ifadd_ == 1:
                group_.append(ex[i][0])
                group_index.append(i)
                avg_ = newavg
                if ex[i][0] > sensitivity_:
                    sum_ = sensitivity_ + common_tools.add_noise(sensitivity_, eps_pub, dim)
                else:
                    sum_ = ex[i][0] + common_tools.add_noise(sensitivity_, eps_pub, dim)

                group_noisy.append(sum_)
                for k in range(len(group_) - 1):
                    sum_ += group_noisy[group_index[k]]
                sum_ = sum_ / len(group_)
                published_result.append(sum_)
            else:
                if ex[i][0] > sensitivity_:
                    noisy_result = sensitivity_ + common_tools.add_noise(sensitivity_, eps_pub, dim)
                else:
                    noisy_result = ex[i][0] + common_tools.add_noise(sensitivity_, eps_pub, dim)

                published_result.append(noisy_result)
                group_noisy.append(noisy_result)
                group_ = []
                group_index = []

    return published_result


# ---------pegasus with delay, post-processing------------
def pegasus_delay(ex, domain_low, domain_high, eps, tau, flag=0, interval_=5, num_=100):
    total_time = len(ex)
    dim = len(ex[0])
    eps_group = eps / 5
    eps_pub = eps - eps_group
    published_result = []

    i = 0
    group_ = []
    group_index = []
    avg_ = 0

    whether_update = 0
    sensitivity_ = domain_high

    for i in range(total_time):
        if whether_update == 0:
            eps_pub = eps - eps_group

        if i % num_ == 0:
            if (i / num_) % interval_ == 0 and flag == 1 and (i + 1) * num_ <= total_time:
                eps_s = eps_pub / 2
                eps_pub = eps_pub - eps_s

                data_sens = np.zeros(num_, dtype=int)
                cc = 0
                for qq in range(i * num_, (i + 1) * num_):
                    data_sens[cc] = ex[qq][0]
                    cc += 1

                sensitivity_ = sensitivity_calc.quality_func(data_sens, domain_low, domain_high, interval_,
                                                             eps_s)
                whether_update = 1
            else:
                whether_update = 0

        if len(group_) == 0:
            group_.append(ex[i][0])
            avg_ = ex[i][0]
            group_index.append(i)

        else:
            ifadd_, newavg = whether_group(group_, avg_, ex[i][0], tau, eps_group, sensitivity_)
            if ifadd_ == 1:
                group_.append(ex[i][0])
                avg_ = newavg
            else:
                sum_ = 0
                for k in range(len(group_)):
                    if group_[k] > sensitivity_:
                        sum_ += sensitivity_ + common_tools.add_noise(sensitivity_, eps_pub, dim)
                    else:
                        sum_ += group_[k] + common_tools.add_noise(sensitivity_, eps_pub, dim)

                sum_ = sum_ / len(group_)
                for k in range(len(group_)):
                    published_result.append(sum_)

                noisy_result = ex[i][0] + common_tools.add_noise(sensitivity_, eps_pub, dim)
                published_result.append(noisy_result)
                group_ = []

    if len(published_result) < total_time:
        sum_ = 0
        for k in range(len(group_)):
            if group_[k] > sensitivity_:
                sum_ += sensitivity_ + common_tools.add_noise(sensitivity_, eps_pub, dim)
            else:
                sum_ += group_[k] + common_tools.add_noise(sensitivity_, eps_pub, dim)

        sum_ = sum_ / len(group_)
        for k in range(len(group_)):
            published_result.append(sum_)

    return published_result


if __name__ == "__main__":
    min_epsilon, max_epsilon, step_epsilon = 0.5, 5.5, 0.5

    with open("pegasus_result", "a") as f:
        f.write("Epsilon\tMSE\tMAE\n")
        f.flush()
        for ep in np.arange(min_epsilon, max_epsilon, step_epsilon):
            round_ = 20
            all_mean_distances = []
            all_mae_error = []
            for i in range(round_):

                filename = "Power.csv"
                df = pd.read_csv(filename, low_memory=False)
                original_data = {}
                noise_data = {}
                euclidean_distance = {}
                mae_error = {}
                nodes = df.columns[1:]

                for col in df.columns[1:]:
                    ex = df[[col]].astype(float).values.tolist()
                    data = np.zeros(len(ex), dtype=float)
                    for j in range(len(ex)):
                        data[j] = ex[j][0]
                    delay_time = 5
                    sensitivity_ = max(data)
                    domain_low = min(data)
                    domain_high = max(data)
                    original_data[col] = df[col].astype(float).values.tolist()
                    noise_data[col] = pegasus_delay(ex, domain_low, domain_high, ep, delay_time, flag=0, interval_=5,
                                                      num_=100)
                    orig = np.array(original_data[col])
                    perturbed = np.array(noise_data[col])
                    diff = orig - perturbed
                    mask = ~np.isnan(diff)
                    vaild_count = np.sum(mask)
                    abs_error = np.abs(orig - perturbed)
                    mae_error[col] = np.sum(abs_error[mask]) / vaild_count
                    euclidean_distance[col] = np.sqrt(np.sum((diff[mask]) ** 2))
                mean_mae = sum(mae_error.values()) / len(nodes)
                all_mae_error.append(mean_mae)
                mean_distance = sum(euclidean_distance.values()) / len(euclidean_distance)
                all_mean_distances.append(mean_distance)
            avg_mean_distance = np.mean(all_mean_distances)
            avg_mae_error = np.mean(all_mae_error)
            print('avg_mean_distance', avg_mean_distance)
            print('avg_mae_error', avg_mae_error)
            f.write("mean_distance"f"{ep}\t{avg_mean_distance}\n")
            f.write("mean_mae"f"{ep}\t{avg_mae_error}\n")
            f.flush()
