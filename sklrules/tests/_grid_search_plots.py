import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# dimensions: structure, avg_rule_length
rnc_data = np.array([
    [0.709615, 0.802485, 0.87766, 0.890135, 0.885745, 0.87646, 0.86455],
    [0.71411, 0.725585, 0.831385, 0.882475, 0.898385, 0.89717, 0.880515],
    [0.758105, 0.734335, 0.738565, 0.78075, 0.841805, 0.871335, 0.877545],
    [0.768065, 0.793845, 0.764305, 0.773985, 0.784775, 0.80986, 0.859275],
    [0.786955, 0.808205, 0.799115, 0.79536, 0.796545, 0.79004, 0.789015],
    [0.79311, 0.823735, 0.83418, 0.830225, 0.80548, 0.80953, 0.79302]
])
rnc_structure = [10, 20, 50, 100, 200, 500]
rnc_avg_rule_length = [1, 2, 3, 4, 5, 6, 7]

# dimensions: init_prob, structure, avg_rule_length
drnc_data = np.array([
    [
        [0.90084, 0.907285, 0.89322],
        [0.876175, 0.89526, 0.88603],
        [0.80331, 0.88755, 0.889935],
        [0.87945, 0.86484, 0.86533],
        [0.88233, 0.87872, 0.85537],
        [0.85381, 0.86571, 0.847465]
    ],
    [
        [0.77203, 0.891605, 0.861135],
        [0.836475, 0.892825, 0.88858],
        [0.742875, 0.877735, 0.898285],
        [0.879885, 0.88066, 0.873285],
        [0.893265, 0.872995, 0.87261],
        [0.839355, 0.874565, 0.860495]
    ],
    [
        [0.666555, 0.825735, 0.85806],
        [0.77593, 0.890185, 0.89722],
        [0.72178, 0.86479, 0.889695],
        [0.839065, 0.862745, 0.88027],
        [0.863095, 0.88106, 0.88062],
        [0.789895, 0.86016, 0.852925]
    ]
])
drnc_init_prob = [0.025, 0.075, 0.125]
drnc_structure_labels = ['[72, 36, 12, 6, 2]', '[32, 16, 8, 4, 2]',
                         '[36, 12, 6, 2]', '[16, 8, 4, 2]', '[12, 6, 2]',
                         '[8, 4, 2]']
drnc_structure = [1, 2, 3, 4, 5, 6]
drnc_avg_rule_length = [1, 2, 3]


def plot_accuracy_graph(fig, ax, x_values, y_values, label):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
        ax.set(xlabel=label.capitalize(), ylabel='Accuracy',
               title=f'Accuracies over different {label}s')
    ax.plot(x_values, y_values, linewidth='1')
    plt.close(fig)
    return fig, ax


def plot_accuracy_wireframe(fig, ax, x_values, y_values, z_values, x_label,
                            y_label):
    if fig is None or ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set(xlabel=x_label.capitalize(), ylabel=y_label.capitalize(),
               zlabel='Accuracy',
               title=f'Accuracies over different {x_label}s and {y_label}s')
    X, Y = np.meshgrid(x_values, y_values)
    ax.axes.set_xticks(x_values)
    ax.axes.set_yticks(y_values)
    for Z, color in zip(z_values, itertools.cycle(['red', 'orange', 'green',
                                                   'cyan', 'blue', 'purple'
                                                   ])):
        ax.plot_wireframe(X.T, Y.T, Z, color=color)
    plt.close(fig)
    return fig, ax


def format_plot_2D(fig, ax, labels, title, name):
    for line, marker, linestyle in zip(
            ax.lines, itertools.cycle('so^PDX*'),
            itertools.cycle(["-", "--", "-.", ":"])):
        line.set_marker(marker)
        line.set_linestyle(linestyle)
    ax.legend(labels, title=title.capitalize(), loc='center left',
              bbox_to_anchor=(1, 0.5))
    fig.savefig(f'plots/{name}.png', bbox_inches='tight')
    plt.close('all')


def format_plot_3D(fig, ax, labels, title, name):
    ax.legend(labels, title=title.capitalize(), loc='center left',
              bbox_to_anchor=(1.2, 0.5))
    fig.savefig(f'plots/{name}.png', bbox_inches='tight')
    plt.close('all')


if __name__ == '__main__':
    rnc_fig, rnc_ax = plot_accuracy_graph(
        None, None, rnc_avg_rule_length, rnc_data.transpose(), 'Ø rule length')
    format_plot_2D(rnc_fig, rnc_ax, rnc_structure, 'layer size', 'rnc_1')

    rnc_fig, rnc_ax = plot_accuracy_graph(
        None, None, rnc_structure, rnc_data, 'layer size')
    rnc_ax.set_xscale('log')
    rnc_ax.axes.set_xticks(rnc_structure)
    rnc_ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    format_plot_2D(rnc_fig, rnc_ax, rnc_avg_rule_length, 'Ø rule length',
                   'rnc_2')

    drnc_fig, drnc_ax = plot_accuracy_wireframe(
        None, None, drnc_init_prob, drnc_avg_rule_length,
        drnc_data.transpose((1, 0, 2)), 'init prob', 'Ø rule length')
    format_plot_3D(drnc_fig, drnc_ax, drnc_structure_labels, 'layer structure',
                   'drnc_1')

    drnc_fig, drnc_ax = plot_accuracy_wireframe(
        None, None, drnc_structure, drnc_init_prob,
        drnc_data.transpose((2, 1, 0)), 'layer structure', 'init prob')
    drnc_ax.axes.set_xticklabels(drnc_structure_labels, {'fontsize': 7})
    format_plot_3D(drnc_fig, drnc_ax, drnc_avg_rule_length, 'Ø rule length',
                   'drnc_2')

    drnc_fig, drnc_ax = plot_accuracy_wireframe(
        None, None, drnc_structure, drnc_avg_rule_length,
        drnc_data, 'layer structure', 'Ø rule length')
    drnc_ax.axes.set_xticklabels(drnc_structure_labels, {'fontsize': 7})
    format_plot_3D(drnc_fig, drnc_ax, drnc_init_prob, 'init prob', 'drnc_3')
