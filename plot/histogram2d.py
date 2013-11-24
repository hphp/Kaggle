import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib


def scatter_hist(x,y):
    fig, axScatter = plt.subplots(figsize=(5.5,5.5))

    # the scatter plot:
    axScatter.scatter(x, y)
    axScatter.set_aspect(1.)

    # create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    divider = make_axes_locatable(axScatter)
    axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=axScatter)
    axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=axScatter)

    # make some labels invisible
    plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),
             visible=False)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    lim = ( int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')

    # the xaxis of axHistx and yaxis of axHisty are shared with axScatter,
    # thus there is no need to manually adjust the xlim and ylim of these
    # axis.

    #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
    for tl in axHistx.get_xticklabels():
        tl.set_visible(False)
    axHistx.set_yticks([0, 50, 100])

    #axHisty.axis["left"].major_ticklabels.set_visible(False)
    for tl in axHisty.get_yticklabels():
        tl.set_visible(False)
    axHisty.set_xticks([0, 50, 100])

    plt.draw()
    plt.show()

def hist2d_bubble(x_data, y_data, bins=10):
    ax = np.histogram2d(x_data, y_data, bins=bins)
    xs = ax[1]
    dx = xs[1] - xs[0]
    ys = ax[2]
    dy = ys[1] - ys[0]
    def rdn():
        return (1-(-1))*np.random.random() + -1
    points = []
    for (i, j),v in np.ndenumerate(ax[0]):
        points.append((xs[i], ys[j], v))
    points = np.array(points)
    fig = pyplot.figure()
    sub = pyplot.scatter(points[:, 0],points[:, 1],
            color='black', marker='o', s=128*points[:, 2])
    sub.axes.set_xticks(xs)
    sub.axes.set_yticks(ys)
    #pyplot.ion()
    pyplot.grid()
    #pyplot.draw()
    pyplot.show()
    return points, sub

if __name__=="__main__":
# the random data
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    #scatter_hist(x,y)

    temperature = [4,   3,   1,   4,   6,   7,   8,   3,   1]
    radius      = [0,   2,   3,   4,   0,   1,   2,  10,   7]
    density     = [1,  10,   2,  24,   7,  10,  21, 102, 203]

    matplotlib.rcParams.update({'font.size':14})

    points, sub = hist2d_bubble(radius, density, bins=4)
    sub.axes.set_xlabel('radius')
    sub.axes.set_ylabel('density')


