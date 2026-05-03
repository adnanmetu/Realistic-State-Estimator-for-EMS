import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(dx_list):

    plt.figure()
    plt.plot(dx_list, marker='o')
    plt.title("Convergence of State Update")
    plt.xlabel("Iteration")
    plt.ylabel("Max |dx|")
    plt.grid()
    plt.show()


def plot_residuals(r):

    plt.figure()
    plt.stem(r)
    plt.title("Residuals")
    plt.xlabel("Measurement Index")
    plt.ylabel("Residual")
    plt.grid()
    plt.show()