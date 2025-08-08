import sys
import os 
import numpy as np
import numpy.linalg as la
from scipy import stats

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
sys.path.append(dir_path)

from generalrandom import GeneralRandom

# l_logm = np.log(0.05) # lower limit on log stellar mass
# u_logm = np.log(8) # upper limit on log stellar mass

# l_age = 8  # lower limit on log stellar age in Gyr
# u_age = 10.1249 # upper limit on log stellar age in Gyr

# l_feh = -4 # lower limit on [Fe/H]
# u_feh = 1 # upper limit on [Fe/H]

# def get_near_psd(A):
#     C = (A + A.T)/2
#     eigval, eigvec = np.linalg.eig(C)
#     eigval[eigval < 0] = 1e-5
#     return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

from numpy import linalg as la

def nearestPD(A):
	"""Find the nearest positive-definite matrix to input

	A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
	credits [2].

	[1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

	[2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
	matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
	:param A: input array
	:type A: array
	"""

	B = (A + A.T) / 2
	_, s, V = la.svd(B)

	H = np.dot(V.T, np.dot(np.diag(s), V))

	A2 = (B + H) / 2

	A3 = (A2 + A2.T) / 2

	if isPD(A3):
		return A3

	spacing = np.spacing(la.norm(A))
	# The above is different from [1]. It appears that MATLAB's `chol` Cholesky
	# decomposition will accept matrixes with exactly 0-eigenvalue, whereas
	# Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
	# for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
	# will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
	# the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
	# `spacing` will, for Gaussian random matrixes of small dimension, be on
	# othe order of 1e-16. In practice, both ways converge, as the unit test
	# below suggests.
	I = np.eye(A.shape[0])
	k = 1
	while not isPD(A3):
		mineig = np.min(np.real(la.eigvals(A3)))
		A3 += I * (-mineig * k**2 + spacing)
		k += 1

	return A3

def isPD(B):
	"""Returns true when input is positive-definite, via Cholesky
	:param B: input array
	:type B: bytearray
	"""
	try:
		_ = la.cholesky(B)
		return True
	except la.LinAlgError:
		return False

class exponential_decay:
    '''
	An exponential star formation that starts at loc and decays with constant 1/scale
	p(x) = 1/scale * exp(-(loc-x)/scale) for x < loc
	Parameters
	----------	loc : float
		location parameter, the maximum age in Gyr
	scale : float
		scale parameter, inverse of the rate of the exponential decay
	Returns
	-------
	rvs : array
		N x 1 array of ages in Gyr, where N is the number of samples
	'''
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale

    def rvs(self, N):
        return - stats.expon.rvs(scale=self.scale, loc=-self.loc, size=N)

class SW_SFH:
	'''
	wraps around a scipy distribution to give it a sample() method
	'''

	def __init__(self, scipy_dist, age_range, feh_range):
		self.scipy_dist = scipy_dist
		self.l_age = age_range[0]
		self.u_age = age_range[1]
		self.l_feh = feh_range[0]
		self.u_feh = feh_range[1]

	def sample(self, N):
		sfh = self.scipy_dist.rvs(N)
		age, feh = sfh.T
		within = (age > self.l_age) * (age < self.u_age) * (feh > self.l_feh) * (feh < self.u_feh)
		age[~within] = np.nan
		feh[~within] = np.nan
		#age = np.log10(age * 1e9) # CONVERT TO LOG AGE FOR ISOCHRONE
		return np.vstack((age, feh)).T



class Emp_MDF_Sci_Age:
	'''
	wraps around GeneralRandom and scipy distribution to sample metallicities from empirical distribution and ages from a scipy distribution
	Parameters
	----------
	age_dist : scipy distribution object
		distribution of ages in Gyr
	feh_gr : GeneralRandom object
		distribution of metallicities
	Returns
	-------
	sample : array
		N x 2 array of ages and metallicities, where each row is [age, feh]
	'''
	
	def __init__(self, age_dist, feh_gr, age_range, feh_range):
		self.age_dist = age_dist
		self.feh_gr = feh_gr
		self.l_age = age_range[0]
		self.u_age = age_range[1]
		self.l_feh = feh_range[0]
		self.u_feh = feh_range[1]

	def sample(self, N):
		age = self.age_dist.rvs(N)
		feh = self.feh_gr.sample(N)
		within = (age > self.l_age) * (age < self.u_age) * (feh > self.l_feh) * (feh < self.u_feh)
		age[~within] = np.nan
		feh[~within] = np.nan
		return np.vstack((age, feh)).T

class GridSFH:
	'''
	loads and samples a grid-based SFH from a dictionary of ages, metallicities, and weights
	'''

	def __init__(self, sfh_grid):
		'''
		Assumes sfh_grid is a dictionary with the following keys:
		'mets' : array of M [Fe/H] grid points
		'ages' : array of A age (Gyr) grid points
		'probabilities' : M x A matrix with probability (or some weight) of each SFH bin
		'''

		self.sfh_grid = sfh_grid
		mets, ages, probs = sfh_grid['mets'], sfh_grid['ages'], sfh_grid['probabilities']
		MM, AA = np.meshgrid(mets[:-1], ages[:-1])

		self.mm = MM.ravel()
		self.aa = AA.ravel()
		self.pp = probs.ravel() / np.sum(probs)
		self.idxs = np.arange(len(self.pp))

		self.dm = np.diff(mets)[0]
		self.da = np.diff(ages)[0]

		self.rng = np.random.default_rng()

	def sample(self, N):
		sel_idx = self.rng.choice(self.idxs, p = self.pp, size = N)
		sampled_m = self.mm[sel_idx] + self.rng.uniform(0, self.dm, size = N)
		sampled_a = self.aa[sel_idx] + self.rng.uniform(0, self.da, size = N)

		return np.vstack((sampled_a, sampled_m)).T

def set_GR_spl(slope, mass_range):
	"""
	defines a GeneralRandom object for a single power-law (Salpeter) IMF
	Parameters
	----------
	slope : float
		IMF slope

	Returns
	-------
	GeneralRandom object
		Salpeter IMF object
	"""
	l_logm = np.log(mass_range[0])  # lower limit on log stellar mass
	u_logm = np.log(mass_range[1])  # upper limit on log stellar mass
	x = np.linspace(l_logm,u_logm,1000)
	y = np.exp(x*(slope+1))
	GR_spl = GeneralRandom(x,y,1000)
	return GR_spl

def set_GR_bpl(alow,ahigh,bm, mass_range):
	"""
	defines a GeneralRandom object for a broken power-law (Kroupa) IMF
	Parameters
	----------
	alow : float
		low-mass IMF slope
	ahigh : float
		high-mass IMF slope
	bm : float
		break mass

	Returns
	-------
	GeneralRandom object	
		Kroupa IMF object
	"""
	l_logm = np.log(mass_range[0])  # lower limit on log stellar mass
	u_logm = np.log(mass_range[1])  # upper limit on log stellar mass
	x = np.linspace(l_logm,u_logm,1000)
	lkm = np.log(bm)*(alow-ahigh)
	y = np.where(x<np.log(bm), x*(alow+1), +lkm+x*(ahigh+1))
	GR_bpl = GeneralRandom(x,np.exp(y),1000)

	return GR_bpl

def set_GR_dgdm(mu1,deltamu,sigma1,sigma2,amprat):
	"""
	defines a GeneralRandom object for a double Gaussian (for use with LOS distances, e.g.)
	Parameters:
	-------------
	mu1,deltamu: Distance MODULUS of Gaussian centers. 
                     If deltamu is positive, mu1 is NEARER Gaussian and mu2 is FARTHER Gaussian.
	sigma1,sigma2: Gaussian std deviations of each of the two Gaussians in distance MODULUS
	amprat: Amplitude ratio of the two Gaussians A1/A2.

	Returns: General Random object (in magnitudes for DM)
	---------
	"""
	a1 = amprat/(1+amprat)
	a2 = 1/(1+amprat)
	ngauss = stats.norm(loc=mu1,scale=sigma1)
	fgauss = stats.norm(loc=mu1+deltamu,scale=sigma2)
	## Get min and max as 5 sigma from either Gaussian
	gxmin = np.min([mu1-(5*sigma1),mu1+deltamu-(5*sigma2)])
	gxmax = np.max([mu1+(5*sigma1),mu1+deltamu+(5*sigma2)])
	
	# Get PDFs for each Gaussian on the same x-grid as a function of distance IN KPC:

	x = np.linspace(gxmin,gxmax,1000)
	npdfmu = ngauss.pdf(x)*a1
	fpdfmu = fgauss.pdf(x)*a2
	y = npdfmu + fpdfmu

	GR_dgdm = GeneralRandom(x,y,1000)
	return GR_dgdm

def set_GR_ln10full(mc,sm,mt,sl, mass_range):
	'''
	defines a GeneralRandom object for a log-normal (Chabrier) IMF
	Parameters
	----------
	mc : float
		mean mass
	sm : float
		sigma mass
	mt : float
		transition mass
	sl : float
		power-law slope after transition mass

	Returns
	-------
	GeneralRandom object
		Chabrier IMF object
	'''
	l_logm = np.log(mass_range[0])  # lower limit on log stellar mass
	u_logm = np.log(mass_range[1])  # upper limit on log stellar mass
	x = np.linspace(l_logm,u_logm,1000)
	BMtr = x>=np.log(mt)
	lkm = np.exp(np.log(mt)*(sl+1)) / np.exp(-0.5* ((np.log(mt)/np.log(10)-np.log10(mc))/sm)**2)
	y = np.empty_like(x)        
	y[~BMtr] = np.exp(-0.5* ((x[~BMtr]/np.log(10)-np.log10(mc))/sm)**2)           
	y[BMtr]  =  np.exp(x[BMtr]*(sl+1))/lkm
	GR_ln10full = GeneralRandom(x,y,1000)

	return GR_ln10full

def set_GR_unif(bf):
	'''
	defines a GeneralRandom object for a uniform distribution coded for the binary fraction
	Parameters
	----------
	bf : binary fraction
		fraction of systems that are in binaries

	Returns
	-------
	GeneralRandom object
		uniform GR object
	'''

	x = np.array([-1,-1e-6,0.,1])
	y = np.array([1-bf,1-bf,bf,bf])
	GR_unif = GeneralRandom(x,y,1000)

	return GR_unif
