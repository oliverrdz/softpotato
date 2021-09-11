import matplotlib.pyplot as plt

def format(xlab, ylab, show):
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(xlab, fontsize=18)
    plt.ylabel(ylab, fontsize=18)
    plt.grid()
    plt.tight_layout()
    if show:
        plt.show()

def plot(x, y, xlab='$E$ / V', ylab='$i$ / A', mark='-', fig=1, show=False):
    plt.figure(fig)
    plt.plot(x, y, mark)
    format(xlab, ylab, show)
