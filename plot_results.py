import matplotlib.pyplot as plt
import csv


ss = [0.1, 0.3, 0.5]
methods = ["ffgd", "fedavg", "scaffold"]


for s in ss:
    # comms = []
    # accus = []
    for method in methods:
        with open(f'csv/{method}_s{s}.csv', 'r') as csvfile:
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
        # comms.append(comm)
        # accus.append(accu)

            plt.plot(comm[1:], accu[1:], label=method)

    plt.xlabel('communication complexity')
    plt.xlim(0, 5000)
    plt.ylabel('testing accuracy')
    plt.title(f'CIFAR10, N=56, s={s}')
    plt.legend()
    plt.savefig(f'plot/test_acc_comm_s{s}.pdf', bbox_inches='tight')

    plt.show()

# plt.errorbar(ffgd_comm[1:], ffgd_accu[1:], yerr=ffgd_yerr[1:], label='ffgd-0.3')
# plt.plot(sgd_time[1:], sgd_loss[1:], label='Adam')
# ax.set_yscale('log')
# plt.yscale('log')
