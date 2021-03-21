import torch
from matplotlib.axes import Axes


def countplot(plot_id: Axes,
              data: list,
              labels: list,
              title: str) -> None:
    """
    Create a bar plot.

    :param plot_id: sub plot axes
    :param data: the list containing the values
    :param labels: the list containing the labels
    :param title: the title of the bar plot
    """

    data_torch = torch.tensor(data, dtype=torch.float)
    plot_id.title.set_text(title)
    plot_id.bar(labels, data_torch.numpy())


def lineplot(plot_id: Axes,
             data: list,
             title: str,
             xlabel: str,
             ylabel: str,
             points: list = (),
             points_style: list = (),
             hline: float = None) -> None:
    """
    Plot the data of the last episodes.

    :param plot_id: sub plot axes
    :param data: list to plot
    :param title: title of the plot
    :param xlabel: title of the x-axis
    :param ylabel: title of the y-axis
    :param points: list of lists of positions where to draw points on the line
    :param points_style: list of dicts representing the style of each list of positions
    :param hline: vertical coordinate where to draw a line
    """

    data_torch = torch.tensor(data, dtype=torch.float)
    plot_id.title.set_text(title)
    plot_id.set_xlabel(xlabel)
    plot_id.set_ylabel(ylabel)
    plot_id.plot(data_torch.numpy())
    for i, p in enumerate(points):
        plot_id.scatter(p, data_torch[p], **points_style[i])
    if hline is not None:
        plot_id.axhline(hline, color='red', linestyle='dashed')
