import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pickle
from skimage.color import label2rgb
from glob import glob
from tqdm import tqdm 
import tools21cm as t2c
from time import sleep

# FuncAnimation makes an animation by repeatedly calling a function func. 
# ArtistAnimation : Animation using a fixed set of Artist objects.

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
dataset_range = np.array([[dataset['{:.3f}'.format(zz)]['dt'].min(),dataset['{:.3f}'.format(zz)]['dt'].max()] for zz in dataset_zs])

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



fig, axs = plt.subplots(1,2,figsize=(11.5,5))
tl0 = axs[0].set_title('$x_\mathrm{{HII}}$', fontsize=16)
dd0 = dataset['{:.3f}'.format(dataset_zs[0])]['xf']; dd0[0,0], dd0[-1,1] = 0,1
im0 = axs[0].imshow(dd0, cmap='Greys', origin='lower')
axs[0].set_ylabel('(Mpc)', fontsize=16)
axs[0].set_xlabel('(Mpc)', fontsize=16)
axs[0].set_yticks(np.arange(100,714,200)*300/500*0.7)
axs[0].set_yticklabels(np.arange(100,714,200), fontsize=14)
axs[0].set_xticks(np.arange(100,714,200)*300/500*0.7)
axs[0].set_xticklabels(np.arange(100,714,200), fontsize=14)
divider = make_axes_locatable(axs[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical')
tl1 =  axs[1].set_title('$\delta T_\mathrm{{b}}$ (mK)', fontsize=16)
dd1 = dataset['{:.3f}'.format(dataset_zs[0])]['dt']; dd1[dd1>200] = 200; dd1[0,0], dd1[-1,1] = 0, 200
im1 = axs[1].imshow(dd1, cmap='viridis', origin='lower')
axs[1].set_ylabel('(Mpc)', fontsize=16)
axs[1].set_xlabel('(Mpc)', fontsize=16)
axs[1].set_yticks(np.arange(100,714,200)*300/500*0.7)
axs[1].set_yticklabels(np.arange(100,714,200), fontsize=14)
axs[1].set_xticks(np.arange(100,714,200)*300/500*0.7)
axs[1].set_xticklabels(np.arange(100,714,200), fontsize=14)
divider = make_axes_locatable(axs[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')
fig.subplots_adjust(left=0.06, bottom=0.12, right=0.965, top=0.94, wspace=0.2, hspace=0.2)

title0 = axs[0].text(0.05,0.05, "Start", bbox={'boxstyle':'round', 'facecolor':'wheat', 'alpha':0.9}, zorder=3, transform=axs[0].transAxes, fontsize=14)#, ha="center")
title1 = axs[1].text(0.05,0.05, "Start", bbox={'boxstyle':'round', 'facecolor':'wheat', 'alpha':0.9}, zorder=3, transform=axs[1].transAxes, fontsize=14)#, ha="center")

def animate(i):
    dd0 = dataset['{:.3f}'.format(dataset_zs[i])]['xf']; dd0[0,0], dd0[-1,1] = 0,1
    dd1 = dataset['{:.3f}'.format(dataset_zs[i])]['dt']; dd1[dd1>200] = 200; dd1[0,0], dd1[-1,1] = 0, 200
    im0.set_data(dd0)
    im1.set_data(dd1)
    title0.set_text('$\langle x_\mathrm{{HII}}\\rangle={:.2f}$'.format(dataset['{:.3f}'.format(dataset_zs[i])]['xf'].mean()))
    title1.set_text('$z={:.2f}$'.format(dataset_zs[i]))
    return im0, im1,

myAnimation = anim.FuncAnimation(fig, animate, frames=np.arange(len(dataset_zs)), interval=200, repeat=True)
myAnimation.save('ReionSim.gif')

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





