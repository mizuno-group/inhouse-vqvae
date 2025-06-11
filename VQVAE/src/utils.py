import matplotlib.pyplot as plt

def plot_loss(train_loss_log, test_loss_log):
    plt.suptitle('Loss')
    plt.plot(train_loss_log, label='train_loss')
    plt.plot(test_loss_log, label='test_loss')
    plt.grid(axis='y')
    plt.legend()
    plt.savefig("/workspace/inhouse-vqvae/VQVAE/results/test.png")