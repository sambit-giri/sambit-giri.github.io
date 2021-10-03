import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from array2gif import write_gif
from skimage.color import label2rgb

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

class AnimatedGif:
    def __init__(self, size=(640, 480)):
        self.fig = plt.figure()
        self.fig.set_size_inches(size[0] / 100, size[1] / 100)
        ax = self.fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.set_xticks([])
        ax.set_yticks([])
        self.images = []
 
    def add(self, image, label=''):
        plt_im = plt.imshow(image, cmap='jet', animated=True)#, vmin=0, vmax=1)
        plt_txt = plt.text(10, 310, label, color='red')
        self.images.append([plt_im, plt_txt])
 
    def save(self, filename):
        animation = anim.ArtistAnimation(self.fig, self.images)
        animation.save(filename, writer='imagemagick', fps=1)

lc_dir = '/Users/sambitgiri/Desktop/Work/simulations/lightcones/'

lc_xf = np.load(lc_dir+'244Mpc_f2_0_250_xfrac_lightcone.npy')
lc_dt = np.load(lc_dir+'244Mpc_f2_0_250_dt_lightcone.npy')
lc_zs = np.load(lc_dir+'244Mpc_f2_0_250_dt_redshifts.npy') 
lc_xs = lc_xf.mean(axis=0).mean(axis=0)


plt.plot(lc_zs, lc_xs)
plt.plot(lc_zs, smooth(lc_xs,15), '--')

xi = np.arange(0,1.1,0.05)

#### mandelbrot
m = 480
n = 320
x = np.linspace(-2, 1, num=m).reshape((1, m))
y = np.linspace(-1, 1, num=n).reshape((n, 1))
C = np.tile(x, (n, 1)) + 1j * np.tile(y, (1, m))
Z = np.zeros((n, m), dtype=complex)
M = np.full((n, m), True, dtype=bool)

animated_gif = AnimatedGif(size=(m, n))
animated_gif.add(M, label='0')
images = []
for i in range(1, 151):
    Z[M] = Z[M] * Z[M] + C[M]
    M[np.abs(Z) > 2] = False
    if i <= 15 or not (i % 10):
        animated_gif.add(M, label=str(i))

animated_gif.save('mandelbrot.gif')

#### Reionization
animated_gif = AnimatedGif(size=(m, n))
animated_gif.add(M, label='0')
dataset_xf = [lc_xf[:,:,np.abs(lc_xs-xx).argmin()] for xx in xi]
# dataset_xf = [label2rgb(xx) for xx in dataset_xf]

for i in range(xi.size):
	animated_gif.add(dataset_xf[i], label=str(i))

animated_gif.save('ReionSim_xf.gif')



dataset_xf = [label2rgb(lc_xf[:,:,np.abs(lc_xs-xx).argmin()]) for xx in xi]
write_gif(dataset_xf, 'ReionSim_xf.gif', fps=5)

dataset = [
    np.array([
        [[255, 0, 0], [255, 0, 0]],  # red intensities
        [[0, 255, 0], [0, 255, 0]],  # green intensities
        [[0, 0, 255], [0, 0, 255]]   # blue intensities
    ]),
    np.array([
        [[0, 0, 255], [0, 0, 255]],
        [[0, 255, 0], [0, 255, 0]],
        [[255, 0, 0], [255, 0, 0]]
    ])
]
write_gif(dataset, 'ReionSim.gif', fps=5)