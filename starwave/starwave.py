import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import os
import sys
import functools
from sklearn.neighbors import KDTree,NearestNeighbors

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
sys.path.append(dir_path)

from generalrandom import GeneralRandom
from distributions import *
from plot import *
from parameters import *
from getmags import *
import intNN 

import torch
import sbi
from sbi import utils as utils
from sbi.utils import user_input_checks
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn

import extinction

from joblib.externals.loky import set_loky_pickler
set_loky_pickler("dill")

import logging

class StarWave:
    """
    StarWave: fitting the stellar birth function of resolved stellar populations 
    with approximate Bayesian computation.
    This is the main class that performs the CMD fitting. The class is instantiated with
    an isochrone dataframe and artifical star database, as well as the type of IMF and
    SFH you want to fit/sample from.
    
    """

    def __init__(self, isodf, asdf, bands, band_lambdas, imf_type, sfh_type = 'gaussian',
        dm_type = 'gaussian', av_type = 'lognormal', sfh_grid = None, Rv = 3.1, trgb=-100, mass_range=None, age_range=None, feh_range=None, params_kwargs = None):
        """
        Initializes the StarWave object
        Parameters
        ----------
        isodf : pandas DataFrame
            Multi-indexed dataframe containing isochrone data for the required photometric bands.
            Should be indexed in Age, [Fe/H], and mass. # add more details
        asdf : pandas DataFrame
            Artifical star database containing input and output magnitudes for artifically injected
            stars, in all the required photometric bands
        bands : list
            list of strings containing the names of the photometric bands used. These names must be consistent
            in the isodf and the asdf
        imf_type : str
            whether to fit an 'spl', 'bpl', or 'ln' IMF parameterization
        sfh_type :  str
            whether to fit a single-burst Gaussian SFH ('gaussian') or sample from a grid-based SFH ('grid')
        dm_type: str
            if dm_type = 'dg', uses fixed double Gaussian with parameters in set_dm_dist.
            Otherwise uses mean and sigma.
        av_type: str
            if av_type = 'lognormal', uses lognormal Av distribution with params 'av_logn_sigma' and 'av_logn_mu' 
            otherwise uses Gaussian with 'av' and 'sig_av'. 
        sfh_grid : dict
            if sfh_type is 'grid', then this dictionary contains the SFH with the following keys:
            'mets' : array of M [Fe/H] grid points
		    'ages' : array of A age (Gyr) grid points
		    'probabilities' : M x A matrix with probability (or some weight) of each SFH bin
        Rv : float
            Rv value to use for the extinction law, default is 3.1
        trgb : float
            TRGB value to use for the CMD fitting, default is -100 (no TRGB)
        mass_range : tuple
            tuple of (min_mass, max_mass) to limit the mass range of the sampled stars
            if not provided, defaults to the full range of the isochrone data
        age_range : tuple
            tuple of (min_age, max_age) to limit the age range of the sampled stars
            if not provided, defaults to the full range of the isochrone data
        feh_range : tuple
            tuple of (min_feh, max_feh) to limit the metallicity range of the sampled stars
            if not provided, defaults to the full range of the isochrone data
        params_kwargs : dict
            dictionary for printing/saving prior parameters

        """

        if sfh_type == 'grid' and sfh_grid is None:
            print('please pass an sfh_grid if you want to use grid-based SFH sampling!')
            raise

        self.imf_type = imf_type
        self.sfh_type = sfh_type
        self.dm_type = dm_type
        self.av_type = av_type
        self.params_kwargs = params_kwargs
        self.params = make_params(imf_type, sfh_type, dm_type, av_type, self.params_kwargs)
        self.make_prior(self.params) ## INITIALIZE FIXED PARAMS VECTOR
        self.bands = bands
        self.iso_int = intNN.intNN(isodf, self.bands)
        self.asdf = asdf
        self.return_inputmags = False

        self.bands_in = [band + '_in' for band in bands]
        self.bands_out = [band + '_out' for band in bands]

        self.asdf_noise = self.asdf[self.bands_out].to_numpy() - self.asdf[self.bands_in].to_numpy()

        self.kdtree = KDTree(asdf[self.bands_in])
        self.trgb = trgb
        self.lim_logmass = np.log(0.1)
        self.sfh_grid = sfh_grid

        self.Rv = Rv
        self.band_lambdas = band_lambdas

        iso_ages = isodf.index.get_level_values('age').unique()
        iso_age_min = iso_ages.min()
        iso_age_max = iso_ages.max()
        iso_feh = isodf.index.get_level_values('[Fe/H]').unique()
        iso_feh_min = iso_feh.min()
        iso_feh_max = iso_feh.max()
        iso_masses = isodf.index.get_level_values('mass').unique()
        iso_mass_min = iso_masses.min()
        iso_mass_max = iso_masses.max()

        if mass_range is None or len(mass_range) != 2 or mass_range[0] < iso_mass_min or mass_range[1] > iso_mass_max or mass_range[0] >= mass_range[1]:
            self.mass_range = (iso_mass_min, iso_mass_max)
            print('mass range not provided or invalid, using full isochrone mass range: %.2f - %.2f' % (iso_mass_min, iso_mass_max))
        else:
            self.mass_range = mass_range
            print('using provided mass range: %.2f - %.2f' % (mass_range[0], mass_range[1]))

        if age_range is None or len(age_range) != 2 or age_range[0] < iso_age_min or age_range[1] > iso_age_max or age_range[0] >= age_range[1]:
            self.age_range = (iso_age_min, iso_age_max)
            print('age range not provided or invalid, using full isochrone age range: %.2f - %.2f' % (iso_age_min, iso_age_max))
        else:
            self.age_range = age_range
            print('using provided age range: %.2f - %.2f' % (age_range[0], age_range[1]))

        if feh_range is None or len(feh_range) != 2 or feh_range[0] < iso_feh_min or feh_range[1] > iso_feh_max or feh_range[0] >= feh_range[1]:
            self.feh_range = (iso_feh_min, iso_feh_max)
            print('metallicity range not provided or invalid, using full isochrone metallicity range: %.2f - %.2f' % (iso_feh_min, iso_feh_max))
        else:
            self.feh_range = feh_range
            print('using provided metallicity range: %.2f - %.2f' % (feh_range[0], feh_range[1]))

        self.debug = False
        
        print('initalized starwave with %s bands, %s IMF, and default priors' % (str(bands), imf_type))
        print('using Rv = %.1f' % (self.Rv))
        self.params.summary()

    def init_scaler(self, observed_cmd, gamma = 0.5):
        """
        initialize min-max scaling of CMD, along with the Nystroem kernel
        Parameters
        ----------
        observed_cmd : array
        gamma : float

        Returns
        -------
        array
            unit-scaled CMD
        """
        self.cmd_scaler = MinMaxScaler()
        self.cmd_scaler.fit(observed_cmd);
        scaled_observed_cmd = self.cmd_scaler.transform(observed_cmd)
        Phi_approx = Nystroem(kernel = 'rbf', n_components=50, gamma = gamma) 
        Phi_approx.fit(scaled_observed_cmd)
        self.mapping = Phi_approx.transform
        print('scaler initialized and mapping defined!')
        return scaled_observed_cmd

    def get_cmd(self, nstars, gr_dict, pdict):
        """
        get a sampled CMD for a set of input parameters and total number of stars
        Parameters
        ----------
        nstars : int
            total number of sampled stars
        gr_dict : dict
            dictionary containing the IMF parameter distributions as GeneralRandom objects
        pdict : dict
            dictionary containing the current starwave parameters

        Returns
        -------

        """

        input_mags = np.empty((nstars, len(self.bands)))
        input_mags[:] = np.nan
        exts = np.empty((nstars, len(self.bands)))  ## REC added Av from pull request
        exts[:] = np.nan      

        masses = gr_dict['logM'].sample(nstars)
        binqs = gr_dict['BinQ'].sample(nstars)
        sfhs = gr_dict['SFH'].sample(nstars)
        dms = gr_dict['DM'].sample(nstars)
        avs = gr_dict['av'].sample(nstars)

        for ii in range(nstars):

            mass = masses[ii]
            binq = binqs[ii]
            age, feh = sfhs[ii]
            dm = dms[ii]
            av = avs[ii]

            if mass < self.lim_logmass or np.isnan(age) or np.isnan(feh):
                continue

            input_mag = get_absolute_mags(mass, age, feh, binq, self.iso_int, self.bands)
            input_mags[ii, :] = input_mag + dm
            exts[ii,:] = np.array([extinction.ccm89(np.array([band_lambda]),av,self.Rv)[0] for band_lambda in self.band_lambdas])

        BM_in_good = ~((np.isnan(input_mags) + (input_mags < self.trgb)).any(axis = 1))
        input_mags = input_mags[BM_in_good]

        exts = exts[BM_in_good]

        if len(input_mags) == 0:
            return input_mags, input_mags, None

        input_mags += exts

        idxs = self.kdtree.query(input_mags)[1][:, 0]

        output_mags = input_mags + self.asdf_noise[idxs]

        output_good = ~(np.isnan(output_mags).any(axis = 1))

        output_mags = output_mags[output_good]

        BM_out_good = np.zeros(len(BM_in_good),dtype=bool)	
        output_good_true_indexes = np.nonzero(output_good)[0]
        BM_in_good_true_indexes = np.nonzero(BM_in_good)[0]
        BM_out_good[BM_in_good_true_indexes[output_good_true_indexes]] = True

        sdict = {'masses': masses, 'binqs': binqs, 'sfhs': sfhs, 'dms': dms, 'avs': avs, 'exts': exts, 'BM_in_good': BM_in_good, 'BM_out_good': BM_out_good}
 
        return input_mags, output_mags, sdict 
    
    def make_cmd(self, mags):
        """
        convert magnitudes to a cmd
        Parameters
        ----------
        mags : array

        Returns
        -------
        array
        """
        cmd = mags
        for ii in range(mags.shape[1] - 1):
            cmd[:, ii + 1] -= cmd[:, 0]

        return cmd

    def best_gamma(self, cmd, q = 0.68, fac = 1, NN = 5):
        """
        find best gamma value using Mario's heuristic
        Parameters
        ----------
        cmd : array
            input CMD
        q : float
            quantile to use in the heuristic
        fac : float
            fudge factor to scale up distances
        NN : int
            number of nearest neighbours to use

        Returns
        -------
        float
            best gamma value
        """
        nbr = NearestNeighbors(n_neighbors = NN, algorithm = 'kd_tree', metric = 'minkowski', p = 2)
        nbr.fit(cmd)
        dst, idx = nbr.kneighbors(cmd, return_distance = True)
        dst = dst[:, -1] # pick NNth distance

        best_dist = np.quantile(dst, q)
        gamma = 1 / (2 * (fac * best_dist)**2)

        return gamma


    def set_sfh_dist(self, pdict, sfh_type):
        """
        initialize and return the SFH distribution so that it can be sampled
        Parameters
        ----------
        pdict : dict
            parameter dictionary containing SFH parameters
        sfh_type : str
            type of SFH being fitted/sampled from ('gaussian' or 'grid')

        Returns
        -------
        object
            A starwave SFH object that can be sampled from
        """

        if sfh_type == 'gaussian':
            cov = pdict['age_feh_corr'] * pdict['sig_age'] * pdict['sig_feh']
            covmat = np.array([[pdict['sig_age']**2, cov], [cov, pdict['sig_feh']**2]])


            if not isPD(covmat):
                covmat = nearestPD(covmat)
                print('found nearest SFH covmat...')

            means = np.array([pdict['age'], pdict['feh']])

            return SW_SFH(stats.multivariate_normal(mean = means, cov = covmat, allow_singular = True), self.age_range, self.feh_range)

        elif sfh_type == 'grid':

            if self.sfh_grid is None:
                print('must pass an sfh_grid to use grid-based sampling!')
                raise

            else:
                return GridSFH(self.sfh_grid)

    def set_dm_dist(self, pdict, dm_type):
        """
        Initialize and return DM distribution so that it can be sampled.
        If dm_type = 'dg', uses double Gaussian LOS distance distribution.
        Otherwise, assumes a mean and sigma as before.
        Parameters:
        ------------
        dm_type: str
            If it's 'dg', uses double Gaussian, otherwise assumes a mean and sigma.
        """ 
        if dm_type == 'dg':
            return set_GR_dgdm(pdict['mu1'],pdict['deltamu'],pdict['sigma1'],pdict['sigma2'],pdict['amprat'])
        else:
           return SWDist(stats.norm(loc = pdict['dm'], scale = pdict['sig_dm'])) 

    def set_av_dist(self, pdict, av_type):
        """
        Initialize and return Av distribution.
        If av_type = 'lognorm', uses Lognormal, otherwise Gaussian with std dev.
        Parameters:
        ----------
        av_type:str
           If its 'lognormal', uses a lognormal, otherwise uses a mean and sigma.
        """
        if av_type == 'lognormal':
            #return SWDist(stats.lognorm(s=pdict['av_logn_sigma'], scale = np.exp(pdict['av_logn_mu'])))
            a = 1 + (pdict['av_logn_sigma']/pdict['av_logn_mu'])**2
            s_logn = np.sqrt(np.log(a))
            scale_logn = pdict['av_logn_mu']/np.sqrt(a)
            return SWDist(stats.lognorm(s = s_logn, scale = scale_logn)) 
        else:
            return SWDist(stats.norm(loc = pdict['av'], scale = pdict['sig_av']))

    def make_prior(self, parameters):
        """
        initialize priors for all sampled parameters
        Parameters
        ----------
        parameters : object
            starwave parameters object

        Returns
        -------
        list
            list of prior distributions in torch format
        """
    
        priors = [];
        self.fixed_params = {};
        self.param_mapper = {};
        idx = 0

        for ii,(name, param) in enumerate(parameters.dict.items()):
            
            if param.fixed:
                self.fixed_params[name] = param.value
                continue
            
            lower = param.bounds[0]
            upper = param.bounds[1]
            
            if param.distribution == 'uniform':
                distribution = torch.distributions.Uniform(lower*torch.ones(1), upper*torch.ones(1))
                priors.append(distribution)    

            elif param.distribution == 'norm':
                try:
                    mean = param.dist_kwargs['mean']
                    sigma = param.dist_kwargs['sigma']
                except:
                    raise ValueError('please pass valid distribution arguments!')
                distribution = torch.distributions.Normal(torch.tensor(mean), torch.tensor(sigma))
                priors.append(distribution)
                
            else:
                raise ValueError('invalid distribution name')

            self.param_mapper[name] = idx
            idx += 1 # IDX maps the vector of sampled parameters, leaving apart the fixed ones. 
    
        return priors

    def sample_cmd(self, params, model):
        """
        wrapper function to sample a CMD for a given set of starwave parameters
        Parameters
        ----------
        params : SWParameters object
        model : str
            'spl', 'bpl', or 'ln' IMF model

        Returns
        -------
        list
            list of two arrays, one for the noiseless CMD and one for the noisy CMD, plus dict of sampled params.
        """

        is_pdict = False

        if isinstance(params, torch.FloatTensor):
            params = params.detach().cpu().numpy()
        elif isinstance(params, (list, np.ndarray)):
            pass
        elif isinstance(params, dict):
            pdict = params
            is_pdict = True


        self.make_prior(self.params) # Re-initialize priors, check fixed parameters

        pdict = {};
        for name in self.params.keys():
            if name in self.fixed_params.keys():
                pdict[name] = self.fixed_params[name]
            else:
                if is_pdict:
                    pdict[name] = params[name] # if params are dictionary  
                else:
                    pdict[name] = params[self.param_mapper[name]] # if params are array or tensor

        # if self.debug:
        #     print('param dictionary in sample_cmd:' + str(pdict))

        if model == 'spl':
            gr_dict = {'logM':set_GR_spl(pdict['slope'], self.mass_range)}
        elif model == 'bpl':
            gr_dict = {'logM':set_GR_bpl(pdict['alow'], pdict['ahigh'], pdict['bm'], self.mass_range)}
        elif model == 'ln':
            gr_dict = {'logM':set_GR_ln10full(pdict['mean'], pdict['sigma'], pdict['bm'], pdict['slope'], self.mass_range)}
        else:
            print('Unrecognized model!')

        gr_dict['BinQ'] = set_GR_unif(pdict['bf'])
        gr_dict['SFH'] = self.set_sfh_dist(pdict, self.sfh_type)
        gr_dict['DM'] = self.set_dm_dist(pdict, self.dm_type) 
        gr_dict['av'] = self.set_av_dist(pdict, self.av_type)

        intensity = 10**pdict['log_int']
        nstars = int(stats.poisson.rvs(intensity))

        mags_in, mags_out, sdict = self.get_cmd(nstars, gr_dict, pdict)
        cmd_in = self.make_cmd(mags_in)
        cmd_out = self.make_cmd(mags_out)

        return cmd_in, cmd_out, sdict

    def sample_norm_cmd(self, params, model):
        """
        wrapper function to sample unit-normalized CMD
        Parameters
        ----------
        params : SWParameters object
        model : str
            'spl', 'bpl', or 'ln' IMF model

        Returns
        -------
        list
            list of two arrays, one for the noiseless CMD and one for the noisy CMD, unit-scaled
        """
        in_cmd, out_cmd, sdict = self.sample_cmd(params, model)
        if len(in_cmd) == 0 or len(out_cmd) == 0:
            print('empty cmd!')
            return self.dummy_cmd, self.dummy_cmd
        return self.cmd_scaler.transform(in_cmd), self.cmd_scaler.transform(out_cmd)

    def kernel_representation(self, P, mapping):
        """
        project a given array (CMD) onto the kernal space
        Parameters
        ----------
        P : array
            CMD to be projected
        mapping : array
            kernel mapping from Nystroem

        Returns
        -------
        array
            projected representation of CMD
        """
        Phi_P = mapping(P).sum(axis=0)
        return Phi_P

    def cmd_sim(self, params, imf_type):
        """
        wrapper function to simulate kernel-represented CMD given parameters
        Parameters
        ----------
        params : SWParameters object
        imf_type : str
            'spl', 'bpl', or 'ln' IMF model

        Returns
        -------
        array
            sampled CMD in kernel representation form
        """
        in_cmd, out_cmd = self.sample_norm_cmd(params, model = imf_type)
        if self.return_inputmags:
            return self.kernel_representation(in_cmd, self.mapping)
        else:
            return self.kernel_representation(out_cmd, self.mapping)

    def fit_cmd(self, observed_cmd,
                n_rounds = 5,
                n_sims = 100,
                savename = 'starwave',
                min_acceptance_rate = 0.0001,
                gamma = None,
                cores = 1, alpha = 0.5,
                statistic = 'output',
                gamma_kw = {}):

        """
        main function to fit an observed CMD using an instatiated StarWave object
        Parameters
        ----------
        observed_cmd :
        n_rounds :
        n_sims :
        savename :
        min_acceptance_rate :
        gamma :
        cores :
        alpha :
        statistic :
        gamma_kw :

        Returns
        -------

        """


        if cores == 1:
            pass # IMPLEMENT SBI MULTICORE

        scaled_observed_cmd = self.init_scaler(observed_cmd, gamma = gamma)
        obs = torch.tensor(self.kernel_representation(scaled_observed_cmd, self.mapping))
        self.obs = obs

        if gamma is None:
            print('finding optimal kernel width...')
            gamma = self.best_gamma(scaled_observed_cmd, **gamma_kw)
            print('setting gamma = %i' % gamma)

        self.dummy_cmd = np.zeros(observed_cmd.shape)
        
        def simcmd(imf_type):
            return lambda params: self.cmd_sim(params, imf_type = imf_type)

        Nobs = len(scaled_observed_cmd)

        #self.params['log_int'].set(value = np.log10(Nobs), bounds = [np.log10(Nobs/2) , np.log10(Nobs*10)])

        print_prior_summary(self.params)
        
        prior = user_input_checks.MultipleIndependent(self.make_prior(self.params))
        simulator = simcmd(self.imf_type)

        self.simulator,self.prior = prepare_for_sbi(simulator,prior)

        inference = SNPE(prior = self.prior)

        self.posteriors = [];
        proposal = self.prior

        for _ in range(n_rounds):
            print('Starting round %i of neural inference...' % (_+1))
            theta, x = simulate_for_sbi(self.simulator, proposal, num_simulations=n_sims, num_workers = cores)
            density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
            posterior = inference.build_posterior(density_estimator)
            self.posteriors.append(posterior)
            proposal = posterior.set_default_x(obs)

        return self.posteriors[-1]
    
if __name__ == '__main__':
    sw = StarWave()
    sw.params.pretty_print()
