import matplotlib.pyplot as plt


def tensor_gridshow(
    tensors,
    nrow,
    ncol,
    figsize,
    cmap="gray",
    vmin=None,
    vmax=None,
    axis=True,
    title=None,
):

    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    for i, ax in enumerate(axs.ravel()):
        if i >= len(tensors):
            ax.axis("off")
            continue
        ax.imshow(tensors[i], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.locator_params(axis="y", nbins=4)
        ax.locator_params(axis="x", nbins=4)
        ax.tick_params(labelleft=False, labelbottom=False, length=0)
        ax.grid(True, color="w", linestyle=":", alpha=0.75)
        if title is not None and not isinstance(title, str):
            title = str(title)
        ax.set_title(title)
        ax.axis(axis)

    return fig
