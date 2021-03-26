import matplotlib.pyplot as plt
import csv
import numpy as np

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 25

# ss = [0.1, 0.3]
# sns = ['01', '03']
ss = [0.1]
sns = ['01']
# methods = ["ffgd", "fed-avg", "scaffold"]
methods = ["ffgb", "fed-avg", "fedprox", "scaffold", "mime"]
# colors = ['r', 'g', 'b']
colors = ['r', 'g', 'b', 'c', 'm']
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
        plt.plot(comm, accu_mean, label=method, color=color, linewidth=5.0)

        plt.fill_between(comm, accu_mean - accu_std, accu_mean + accu_std, facecolor=color, alpha=0.15)

        # plt.plot(comm[1:], accu[1:], label=method)

    # ax = plt.gca()
    x = np.arange(0, 2000)
    y = np.ones(2000)*0.65
    plt.plot(x, y, 'k-.', label="sgd-opt", linewidth=5.0)
    x = np.arange(0, 2000)
    y = np.ones(2000) * 0.73
    plt.plot(x, y, 'y-.', label="ffgb-opt", linewidth=5.0)

    plt.ticklabel_format(axis='x', style='sci', scilimits=(3, 3))

    plt.xlabel('communication cost', size=BIGGER_SIZE)
    plt.xlim(0, 2000)
    plt.ylabel('testing accuracy', size=BIGGER_SIZE)
    plt.title(f'CIFAR10, N=56, s={s}', size=BIGGER_SIZE)
    plt.xticks(size=BIGGER_SIZE)
    plt.yticks(size=BIGGER_SIZE)
    plt.legend()
    plt.legend(fontsize='x-large')
    plt.xticks(np.arange(0, 2001, 400))
    plt.savefig(f'plot/test_acc_comm_s{s_n}.pdf', bbox_inches='tight')

    plt.show()

# plt.errorbar(ffgd_comm[1:], ffgd_accu[1:], yerr=ffgd_yerr[1:], label='ffgd-0.3')
# plt.plot(sgd_time[1:], sgd_loss[1:], label='Adam')
# ax.set_yscale('log')
# plt.yscale('log')
