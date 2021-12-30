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
from scipy.interpolate import splev, splrep

dmo_S19 = np.array([[0.007306210160205064, 3581908355054.9644],
					[0.009326175098523331, 4440220601826.679],
					[0.015637177279548192, 6763419121021.123],
					[0.021733325857132522, 8645093207693.061],
					[0.02829464011726156, 10166963850720.59],
					[0.042050099799057786, 12546834098779.879],
					[0.06940729639353253, 15014515258091.535],
					[0.1053628483528311, 15752662577924.37],
					[0.16013719978373625, 15138792311118.457],
					[0.2680821925252788, 12922900531146.8],
					[0.41631571099693926, 10329295267156.273],
					[0.5879466476135495, 8220577351897.477],
					[0.8631231818479155, 6206692488400.007],
					[1.1185095782365746, 5116318840923.451],
					[1.5794380269866717, 4107709938440.4766],
					[2.205104842717764, 3778399877665.591],
					[3.0706678260620572, 4197035729411.267],
					[4.032960731810778, 5202653250265.885],
					[5.24294320819317, 6797871556695.32],
					[6.5629384733623155, 8766287323960.171],
					[8.21230000942983, 11606200419927.406],
					[9.89888144340735, 14706936469200.236],
					[10.969046913467757, 16848325715368.281],
					]).T 
dmb_S19 = np.array([[0.006070855587329547, 5044002606534.376],
					[0.007909359145843054, 5627735336047.46],
					[0.012218668931306801, 6588746849684.533],
					[0.01638421468967931, 7191624006335.699],
					[0.021975149231807656, 7713119259963.281],
					[0.028109774022764978, 8200412210833.421],
					[0.03595046587571364, 8833994611495.178],
					[0.05197006686443474, 10252601848187.951],
					[0.07954994461242931, 11743012437710.178],
					[0.11624915518213656, 12374792584787.938],
					[0.17495943245963916, 12156376943900.979],
					[0.26867936830160344, 10986643226734.594],
					[0.3974772130887689, 9461901033643.68],
					[0.5883357137971055, 7833292672687.588],
					[0.7992717576823822, 6600167703988.5625],
					[1.1282377952811584, 5440374002345.68],
					[1.5473782311904603, 4583917084118.806],
					[2.2879121352336713, 4106743329034.8965],
					[2.9549060805336396, 4328027756893.3687],
					[3.5346633820282607, 4786978172126.398],
					[4.387212356049117, 5729506666711.08],
					[5.703135613461793, 7519181560861.714],
					[7.410197790915887, 10220357135711.682],
					[9.016240024036565, 13064901539034.963],
					[10.083942222865968, 15232077218961.938],
					[10.865314722848847, 16848427386066.09],
					]).T

dmo_S19_tck = splrep(dmo_S19[0], dmo_S19[1])
dmb_S19_tck = splrep(dmb_S19[0], dmb_S19[1])
rs = 10**np.linspace(-2.22,1,30)


fig = plt.figure(figsize=(12,5))
plt.subplot(122)
plt.loglog(rs, splev(rs, dmo_S19_tck), lw=3, c='k', label='DMO')
plt.loglog(rs, splev(rs, dmb_S19_tck), lw=3, c='g', label='DMB')
plt.xlabel('$r$ [Mpc/h]', fontsize=15)
plt.ylabel('$r^2 \\rho(r)$ [M$_\odot$/Mpc]', fontsize=15)
plt.yticks([4e12,6e12,1e13],['10$^{12.6}$','10$^{12.8}$','10$^{13.0}$'])  #['4e12','6e12','1e13']
plt.axis([0.006,10,3.2e12,1.8e13])
# plt.tight_layout()
# plt.show()
plt.subplot(121)
plt.loglog(rs, splev(rs, dmo_S19_tck)/rs**2, lw=3, c='k', label='dark-matter-only')
plt.loglog(rs, splev(rs, dmb_S19_tck)/rs**2, lw=3, c='g', label='dark-matter-baryon')
plt.xlabel('$r$ [Mpc/h]', fontsize=15)
plt.ylabel('$\\rho(r)$ [M$_\odot$/Mpc]', fontsize=15)
plt.legend(loc=0,fontsize=16)
plt.axis([0.006,10,1.2e11,1.8e17])
plt.tight_layout()
plt.savefig('matter_density_profiles.png')
plt.show()

# x = np.linspace(-5, 5, 300)
# y = np.linspace(-5, 5, 300)
# xx, yy = np.meshgrid(x, y, sparse=True)
# rr = np.sqrt(xx**2+yy**2)

# fig = plt.figure(figsize=(12,5))
# #plt.imshow(rr, origin='lower', cmap='cubehelix')
# cmap = 'magma' #'cubehelix'
# plt.subplot(121)
# plt.pcolor(xx, yy, np.log10(splev(rr, dmo_S19_tck)/rr**2), cmap=cmap)
# plt.subplot(122)
# plt.pcolor(xx, yy, np.log10(splev(rr, dmb_S19_tck)/rr**2), cmap=cmap)
# plt.tight_layout()
# plt.show()

# def pdf_to_points_2Dspace(func, n_points=1000, mins=np.array([-5, -5]), maxs=np.array([5., 5.])):
# 	import emcee 
# 	from multiprocessing import Pool, cpu_count
# 	def log_probability(theta):
# 		x, y = theta
# 		if x<mins[0] or x>maxs[0] or y<mins[1] or y>maxs[1]: return -np.inf 
# 		r = np.sqrt(x**2+y**2)
# 		lnL = np.log(func(r)) if r>10**-2.22 else np.log(func(10**-2.22))
# 		# print(r,lnL)
# 		return lnL

# 	pos = np.random.uniform(0,1,size=(64, len(mins)))
# 	nwalkers, ndim = pos.shape
# 	# with Pool() as pool:
# 	# 	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
# 	# 	sampler.run_mcmc(pos, n_points, progress=True);
# 	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
# 	sampler.run_mcmc(pos, n_points, progress=True);

# 	flat_samples = sampler.get_chain(discard=0, flat=True) 
# 	flat_logprob = sampler.get_log_prob(discard=0, flat=True) 
# 	return flat_samples#[np.argsort(flat_logprob)[-n_points:],:]


# pdf_dmo = lambda x: (splev(x, dmo_S19_tck)/x**2)/(splev(10**-2.22, dmo_S19_tck)/(10**-2.22)**2)
# pdf_dmb = lambda x: (splev(x, dmb_S19_tck)/x**2)/(splev(10**-2.22, dmb_S19_tck)/(10**-2.22)**2)
# # rs1 = 10**np.linspace(-2.22,1)
# # plt.loglog(rs1, pdf_dmo(rs1), lw=3, c='k', label='DMO')
# # plt.loglog(rs1, pdf_dmb(rs1), lw=3, c='r', label='DMB')
# # plt.legend()
# # plt.show()

# points_dmo = pdf_to_points_2Dspace(pdf_dmo, n_points=1000)
# points_dmb = pdf_to_points_2Dspace(pdf_dmb, n_points=1000)

# plt.scatter(points_dmb[:,0], points_dmb[:,1], s=1)
# plt.scatter(points_dmo[:,0], points_dmo[:,1], s=1)
# plt.axis([-5,5,-5,5])
# plt.show()





