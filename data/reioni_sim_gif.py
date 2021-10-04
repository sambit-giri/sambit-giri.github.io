import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import pickle
from skimage.color import label2rgb
from glob import glob
from tqdm import tqdm 
import tools21cm as t2c

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# lc_dir = '/Users/sambitgiri/Desktop/Work/simulations/lightcones/'

# lc_xf = np.load(lc_dir+'244Mpc_f2_0_250_xfrac_lightcone.npy')
# lc_dt = np.load(lc_dir+'244Mpc_f2_0_250_dt_lightcone.npy')
# lc_zs = np.load(lc_dir+'244Mpc_f2_0_250_dt_redshifts.npy') 
# lc_xs = lc_xf.mean(axis=0).mean(axis=0)


# plt.plot(lc_zs, lc_xs)
# plt.plot(lc_zs, smooth(lc_xs,15), '--')

# Get data from dawn-1

class AnimatedGif:
    def __init__(self, size=(640, 480), figsize=None, fps=1, cmap='jet', text_color='red'):
        self.fig = plt.figure(figsize=figsize)
        if figsize is None: self.fig.set_size_inches(size[0] / 100, size[1] / 100)
        ax = self.fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.set_xticks([])
        ax.set_yticks([])
        self.images = []
        self.fps = fps
        self.cmap = cmap
        self.text_color = text_color
 
    def add(self, image, label=''):
        plt_im = plt.imshow(image, cmap=self.cmap, animated=True)#, vmin=0, vmax=1)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.95)
        plt_txt = plt.text(120, 25, label, color=self.text_color, fontsize=14, bbox=props)
        plt.title(label)
        self.images.append([plt_im, plt_txt])
 
    def save(self, filename):
        animation = anim.ArtistAnimation(self.fig, self.images)
        animation.save(filename, writer='imagemagick', fps=self.fps)


xf_dir = '/disk/dawn-1/garrelt/Reionization/C2Ray_WMAP7/500Mpc/500Mpc_z50_0_300/results/' #'/disk/dawn-1/garrelt/Reionization/C2Ray_WMAP7/500Mpc/500Mpc_f2_0_300/retrieved_from_khagolaz/results/'
dn_dir = '/disk/dawn-1/garrelt/Reionization/C2Ray_WMAP7/500Mpc/coarser_densities/nc300/'
xf_files = glob(xf_dir+'xfrac3d_*')
dn_files = glob(dn_dir+'*n_all.dat')
xf_zs = np.array([ff.split('xfrac3d_')[-1].split('.bin')[0] for ff in xf_files]).astype(float)
dn_zs = np.array([ff.split('/')[-1].split('n_all')[0] for ff in dn_files]).astype(float)
zs = np.intersect1d(xf_zs,dn_zs)

filename = './slices_dataset_500Mpc_z50_0_300.pkl' #'./slices_dataset_500Mpc_f2_0_300.pkl'
dataset  = pickle.load(open(filename, 'rb')) if glob(filename) else {}
if len(dataset.keys())==0:
	for zz in tqdm(zs):
		xx = t2c.XfracFile(xf_dir+'xfrac3d_{:.3f}.bin'.format(zz)).xi 
		dd = t2c.DensityFile(dn_dir+'{:.3f}n_all.dat'.format(zz)).cgs_density 
		dt = t2c.calc_dt(xx, dd, zz)
		dataset['{:.3f}'.format(zz)] = {'xf': xx[:,:,100], 'dt': dt[:,:,100]}
pickle.dump(dataset,open(filename, 'wb'))

dataset_zs = np.array([ii for ii in dataset.keys()]).astype(float)
dataset_xs = np.array([dataset['{:.3f}'.format(zz)]['xf'].mean() for zz in dataset_zs])
dataset_zs = dataset_zs[np.argsort(-dataset_zs)][12:-12]


animated_gif = AnimatedGif(figsize=(5,5), fps=3, cmap='Greys', text_color='black')
animated_gif.add(dataset['{:.3f}'.format(dataset_zs[0])]['xf'], label='$x_\mathrm{{HII}}={:.2f}$'.format(dataset['{:.3f}'.format(dataset_zs[0])]['xf'].mean()))
for i,zz in enumerate(dataset_zs):
	animated_gif.add(dataset['{:.3f}'.format(zz)]['xf'], label='$x_\mathrm{{HII}}={:.2f}$'.format(dataset['{:.3f}'.format(zz)]['xf'].mean()))#label='z={:.2f}'.format(zz))
animated_gif.save('ReionSim_xf.gif')

animated_gif = AnimatedGif(figsize=(5,5), fps=3, cmap=None, text_color='black')
animated_gif.add(dataset['{:.3f}'.format(dataset_zs[0])]['dt'], label='z={:.2f}'.format(dataset_zs[0]))
for i,zz in enumerate(dataset_zs):
	animated_gif.add(dataset['{:.3f}'.format(zz)]['dt'], label='z={:.2f}'.format(zz))
animated_gif.save('ReionSim_dt.gif')


# #### mandelbrot

class AnimatedGif:
    def __init__(self, size=(640, 480)):
        self.fig = plt.figure()
        self.fig.set_size_inches(size[0] / 100, size[1] / 100)
        ax = self.fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.set_xticks([])
        ax.set_yticks([])
        self.images = []
 
    def add(self, image, label=''):
        plt_im = plt.imshow(image, cmap='Greys', animated=True, vmin=0, vmax=1)
        plt_txt = plt.text(10, 310, label, color='red')
        self.images.append([plt_im, plt_txt])
 
    def save(self, filename):
        animation = anim.ArtistAnimation(self.fig, self.images)
        animation.save(filename, writer='imagemagick', fps=1)

# m = 480
# n = 320
# x = np.linspace(-2, 1, num=m).reshape((1, m))
# y = np.linspace(-1, 1, num=n).reshape((n, 1))
# C = np.tile(x, (n, 1)) + 1j * np.tile(y, (1, m))
# Z = np.zeros((n, m), dtype=complex)
# M = np.full((n, m), True, dtype=bool)

# animated_gif = AnimatedGif(size=(m, n))
# animated_gif.add(M, label='0')
# images = []
# for i in range(1, 151):
#     Z[M] = Z[M] * Z[M] + C[M]
#     M[np.abs(Z) > 2] = False
#     if i <= 15 or not (i % 10):
#         animated_gif.add(M, label=str(i))

# animated_gif.save('mandelbrot.gif')





