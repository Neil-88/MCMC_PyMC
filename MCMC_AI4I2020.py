import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
#import seaborn as sns
import math as m
from scipy import stats
from statsmodels.stats.anova import anova_lm
from ucimlrepo import fetch_ucirepo 
import pymc as pm
import arviz as az
from scipy.stats.mstats import mquantiles
import seaborn as sns
import matplotlib as mpl
import matplotlib.font_manager as fm

arial = fm.FontProperties(fname=r"C:\Users\kmes9\Downloads\應用程式\arial.ttf")
fm.fontManager.addfont(r"C:\Users\kmes9\Downloads\應用程式\arial.ttf")
mpl.rcParams['font.family'] = 'Arial'
arial_bold = fm.FontProperties(fname=r"C:\Users\kmes9\Downloads\應用程式\arial.ttf", weight='bold')
plt.rcParams['font.size'] = 15

# fetch dataset 
ai4i_2020_predictive_maintenance_dataset = fetch_ucirepo(id=601) 
  
# data (as pandas dataframes) 
feature = ai4i_2020_predictive_maintenance_dataset.data.features 
target = ai4i_2020_predictive_maintenance_dataset.data.targets 

def logistic(x: np.ndarray, beta:float, alpha:float=0):
    """Logistic Function"""
    z = alpha[:, None] + beta[:, None] * x[None, :]
    return 1.0 / (1.0 + np.exp(-z))

def mcmcSimulations(ToolWear, failures, draws=2000, tune=1000, seed=1234):
    """Perform the MCMC-simulations
    
    Parameters
    ----------
    ToolWear : Tool Wear data
    failureData : corresponding failure status (TWF)
    
    Returns
    -------
    alpha_post : posterior distribution of alpha values
    beta_post :  posterior distribution of beta values
    """
    w = np.asarray(ToolWear, dtype=float)
    y = np.asarray(failures).ravel().astype(int)

        # Standardise the predictor
        # beta extremely correlated and sampling very slow/unstable otherwise.
    w_mean, w_std = w.mean(), w.std()          # scalars
    w_z = (w - w_mean) / w_std


    with pm.Model() as model:
        # PyMC2's precision 0.001 == sigma of ~31.6; PyMC 5 uses sigma
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        # Define the model-function for the temperature
        p = pm.Deterministic("p", pm.math.sigmoid(alpha + beta * w_z))
        pm.Bernoulli("failure_obs", p=p, observed=y)

        # NUTS replaces the old Metropolis MCMC; no find_MAP needed
        idata = pm.sample(draws=draws, tune=tune, chains=4, random_seed=seed, progressbar=True)

    with model:
        # Calculate failure probability
        pred = pm.sample_posterior_predictive(idata, var_names=["p"], predictions=True, random_seed=seed)
    model_pred = pred.predictions["p"].isel(chain=0).values #return the value of first chain
    #print(len(model_pred[0]))
    # Dimension: chain × draw × obs_id
    #p_samples = (pred.predictions["p"].stack(sample=("chain", "draw")).transpose("obs_id", "sample").values)
    # average failure probability   
    #p_mean = p_samples.mean(axis=1)
    #print(f'The average failure probability is: %.4f' %(p_mean))

    # Convert back to the original temperature scale:
    # alpha_z + beta_z * (t - mean)/std = (alpha_z - beta_z*mean/std) + (beta_z/std) * t
    post = idata.posterior
    beta_raw = (post["beta"] / w_std).values.ravel()
    alpha_raw = (post["alpha"] - post["beta"] * w_mean / w_std).values.ravel()

    return alpha_raw, beta_raw, idata, model_pred[0]

def calculateProbability(alpha_samples, beta_samples, ToolWaer, data):
    """Calculate the mean probability, and the CIs
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    ToolWaer = np.asarray(ToolWaer, dtype=float)
    # Calculate the probability as a function of time
    w = np.linspace(ToolWaer.min() - 5, ToolWaer.max() + 5, 200)
    p_w = logistic(w, beta_samples, alpha_samples)
    
    mean_prob_w = p_w.mean(axis=0)
    
    # --- Calculate CIs ---
    # vectorized bottom and top 2.5% quantiles for "confidence interval"
    quantiles = mquantiles(p_w, [0.025, 0.975], axis=0)
    
    return (w[:, None], mean_prob_w, p_w, quantiles)

def showAlphaBetaTraceDist(idata, alpha_s, beta_s):
    '''Show the distribution of alpha and beta
    Parameters
    ------------------------------
    idata,
    alpha_s, 
    beta_s
    '''
    # plot the sample trace of alpha and beta
    plt.figure(figsize=(12.5, 4))
    az.plot_trace(idata, var_names=["alpha", "beta"]) # there are 4 chains(4 colors of lines): represents different sampling from different starting points
    plt.tight_layout()
    plt.savefig('alpha_beta_SampleTrace', dpi=300, bbox_inches='tight', transparent=False)
    
    # plot the alpha and beta distribution
    fig, ax = plt.subplots(2,1,figsize=(12.5, 4))
    ax[0].hist(alpha_s)
    ax[0].set_title('alpha')
    ax[1].hist(beta_s)
    ax[1].set_title('beta')
    plt.subplots_adjust(hspace=0.4)
    plt.savefig('alpha_beta_Dist', dpi=300, bbox_inches='tight', transparent=False)

def plotFailProb(linearToolWear, quantiles, prob_w, ToolWear, failures):
    plt.figure(figsize=(12.5, 4))
    sns.set_style('darkgrid')
        
    plt.fill_between(linearToolWear[:, 0], *quantiles, alpha=0.7,
                         color="#7A68A6") # "*"：Unpacking Operator which represents that quantiles = (a,b,c,...) (quantiles values if they are tuples)
        
    plt.plot(linearToolWear[:, 0], quantiles[0], label="95% CI", color="#7A68A6", alpha=0.7)
    plt.plot(linearToolWear[:, 0], quantiles[1], color="#7A68A6", alpha=0.7)
        
    plt.scatter(ToolWear, prob_w, s=2, color="b",
                 label="probability of defect")
        
    plt.xlim(linearToolWear.min(), linearToolWear.max())
    plt.ylim(-0.02, 1.02)
    plt.legend(loc="lower left")
    plt.scatter(ToolWear, failures, color="k", s=50, alpha=0.5)
    plt.xlabel("Tool Wear [min]")
    plt.ylabel("Posterior Probability Estimate")
        
    outFile = 'Posterior_Failure_CIs_Prob.png'
    plt.savefig(outFile, dpi=300, bbox_inches='tight', transparent=False)

def showProbabilities(linearToolWaer, ToolWaer, failures, mean_prob_w, p_w, quantiles):
    """Show the posterior probabilities, and save the resulting figures
    
    Parameters
    ----------
    linearToolWaer :
    ToolWaer : 
    failures :
    mean_prob_w :
    p_w :
    quantiles :
    
    """

    # --- Show the probability curve ----
    plt.figure(figsize=(12.5, 4))
    
    plt.plot(linearToolWaer, mean_prob_w, lw=3, label="Average posterior probability of defect")
    plt.plot(linearToolWaer, p_w[0, :], ls="--", label="Realization from posterior")
    plt.plot(linearToolWaer, p_w[-2, :], ls="--", label="Realization from posterior")
    plt.scatter(ToolWaer, failures, color="k", s=50, alpha=0.5)
    plt.title("Posterior expected value of probability of defect, plus realizations")
    plt.legend(loc="lower left")
    plt.ylim(-0.1, 1.1)
    plt.xlim(linearToolWaer.min(), linearToolWaer.max())
    plt.ylabel("Probability")
    plt.xlabel("Tool Wear [min]")
    
    outFile = 'Failure_Probability.png'
    plt.savefig(outFile, dpi=300, bbox_inches='tight', transparent=False)
    
    # --- Draw CIs ---
    sns.set_style('darkgrid')
    
    plt.fill_between(linearToolWaer[:, 0], *quantiles, alpha=0.7,
                     color="#7A68A6") # "*"：Unpacking Operator which represents that quantiles = (a,b,c,...) (quantiles values if they are tuples)
    
    plt.plot(linearToolWaer[:, 0], quantiles[0], label="95% CI", color="#7A68A6", alpha=0.7)
    plt.plot(linearToolWaer[:, 0], quantiles[1], color="#7A68A6", alpha=0.7)
    
    plt.plot(linearToolWaer, mean_prob_w, lw=1, ls="--", color="k",
             label="average posterior \nprobability of defect")
    
    plt.xlim(linearToolWaer.min(), linearToolWaer.max())
    plt.ylim(-0.02, 1.02)
    plt.legend(loc="lower left")
    plt.scatter(ToolWaer, failures, color="k", s=50, alpha=0.5)
    plt.xlabel("Tool Wear [min]")
    plt.ylabel("Posterior Probability Estimate")
    
    outFile = 'Failure_CIs_Prob.png'
    plt.savefig(outFile, dpi=300, bbox_inches='tight', transparent=False)

if __name__=='__main__':
   alpha_s, beta_s, idata, p_f = mcmcSimulations(feature["Tool wear"], target["TWF"])
   print(az.summary(idata, var_names=["alpha", "beta"]))
   print(f"beta (per K): mean={beta_s.mean():.4f}, "
          f"94% HDI={np.percentile(beta_s, [3, 97])}")
   (linearToolwear, mean_p, p, quantiles) = calculateProbability(alpha_s, beta_s, feature["Tool wear"], target["Machine failure"])
   showAlphaBetaTraceDist(idata, alpha_s, beta_s)
   #plt.scatter(feature["Tool wear"], p_f)
   #plt.savefig('failure_prob', dpi=300, bbox_inches='tight', transparent=False)
   showProbabilities(linearToolwear, feature["Tool wear"], target["TWF"], mean_p, p, quantiles)
   plotFailProb(linearToolwear, quantiles, p_f, feature["Tool wear"], target["TWF"])