import matplotlib.pyplot as plt
import csv
import numpy as np

# ss = [0.1, 0.3, 0.5]
# sns = ['01', '03', '05']
ss = [0.1]
sns = ['01']
# methods = ["ffgd", "fed-avg", "scaffold"]
methods = ["ffgd", "fed-avg"]
# colors = ['r', 'g', 'b']
colors = ['r', 'g']
for s, s_n in zip(ss, sns):
    for method, color in zip(methods, colors):
        comms = []
        accus = []
        min_len = 10000000
        for j in range(5):
            with open(f'csv/s{s_n}/{method}{j}.csv', 'r') as csvfile:
                plots = csv.reader(csvfile, delimiter=',')
                i = 0
                comm = []
                accu = []
                for row in plots:
                    if i == 0:
                        i += 1
                        continue
                    comm.append(float(row[1]))
                    accu.append(float(row[2]))
                min_len = min(min_len, len(accu))
            accus.append(accu)
        accus = [accu[:min_len] for accu in accus]
        comm = comm[:min_len]
        accus_array = np.asarray(accus)
        accu_mean = np.mean(accus_array)
        accu_mean = np.mean(accus_array, axis=0)
        accu_std = np.std(accus_array, axis=0)
        plt.plot(comm, accu_mean, label=method, color=color)

        plt.fill_between(comm, accu_mean - accu_std, accu_mean + accu_std, facecolor=color, alpha=0.15)

        # plt.plot(comm[1:], accu[1:], label=method)

    plt.xlabel('communication complexity')
    plt.xlim(0, 2000)
    plt.ylabel('testing accuracy')
    plt.title(f'CIFAR10, N=56, s={s}')
    plt.legend()
    plt.savefig(f'plot/test_acc_comm_s{s_n}.pdf', bbox_inches='tight')

    plt.show()

# plt.errorbar(ffgd_comm[1:], ffgd_accu[1:], yerr=ffgd_yerr[1:], label='ffgd-0.3')
# plt.plot(sgd_time[1:], sgd_loss[1:], label='Adam')
# ax.set_yscale('log')
# plt.yscale('log')
