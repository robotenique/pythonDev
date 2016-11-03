import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

# System backward compatibility
import sys
if sys.version_info[0] < 3:
    from Tkinter import *
    from Tkinter import ttk
    from Tkinter import tkMessageBox as messagebox

else:
    from tkinter import *
    from tkinter import ttk
    from tkinter import messagebox


# plotting function: clear current, plot & redraw
def plot(lim):
    # Clear current plot
    plt.clf()
    # Create the Data to plot
    lim = int(lim.get())
    fRelMatrix = [[] for y in range(6)]
    x_axis = [x for x in range(1, lim+1)]
    uniform_list = np.random.randint(1, size=lim, high=7)
    color = [[1, 0, 0.85], [0.4, 0, 0.8520772], [0, 0, 1], [0.2, 1, 1],
    [0.2, 1, 0], [0.9, 0.973553065884, 0.09], [1, 0.407421725309, 0], [1, 0, 0]]
    z = np.arange(6)+2
    data_freq = np.array([0, 0, 0, 0, 0, 0])
    i = 0
    for num in uniform_list:
        i += 1
        data_freq[num-1] += 1
        for j in range(6):
            fRelMatrix[j].append(data_freq[j]/i)
    for j in range(6):
        if(i < 10000):
            plt.plot(x_axis, fRelMatrix[j], linewidth=3, color=color[j])
        plt.scatter(x_axis, fRelMatrix[j], marker='+', c=color[j], zorder=z[j])

    # Axis configuration
    plt.ylim(0, 1)
    plt.xlim(0, lim)
    plt.xlabel('Nº jogadas', fontsize=18)
    plt.ylabel('Freq. Relativa', fontsize=18)
    ax = plt.gca()
    plt.tight_layout()
    ax.spines['bottom'].set_color('k')
    ax.spines['left'].set_color('k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_axis_bgcolor('k')
    ax.xaxis.label.set_color('k')
    ax.yaxis.label.set_color('k')
    ax.tick_params(axis='x', colors='k')
    ax.tick_params(axis='x', colors='k')

    # Draw graphic
    plt.gcf().canvas.draw()


def onCalc(nEntry):
        errmsg = "Digite um número inteiro válido!"
        try:
            n = int(nEntry.get())
            plot(nEntry)
        except ValueError:
            messagebox.showerror("Entrada inválida", errmsg)
            return


def createInterface():
    # GUI
    root = Tk()
    root.title('Gráfico de frequência')
    root.resizable(0, 0)
    nLabel = ttk.Label(root, text="N =")
    nEntry = ttk.Entry(root, width=12)
    nEntry.insert(0, "60")
    draw_button = ttk.Button(root, text="Plot!", width=15, command=lambda: onCalc(nEntry))
    # Layout
    nLabel.grid(row=0,column=0, pady=5, sticky=N)
    nEntry.grid(row=0,column=1, pady=5, padx=10, sticky=N)
    draw_button.grid(row=0, column=0, padx=5, sticky="we")

    # init figure
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)

    canvas = FigureCanvasTkAgg(fig, master=root)
    toolbar = NavigationToolbar2TkAgg(canvas, root)
    canvas.get_tk_widget().grid(row=0, column=2)
    toolbar.grid(row=1, column=2)

    root.mainloop()


def main():
    createInterface()

if __name__ == '__main__':
    main()
