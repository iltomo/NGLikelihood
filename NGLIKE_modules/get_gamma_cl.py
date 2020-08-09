#Draw a set of 1000 hatCl from Gamma distribution of fixed fid_Cl
from numpy import empty
from scipy.stats import gamma

def gamma_draw_Cl (fid_l, fid_Cl):

  gamma_hat_cl = empty (len(fid_l))

  for ell in range (len (fid_l)):
    nu = 2.*fid_l [ell] + 1.
    a = nu/2.
    b = 2. * fid_Cl [ell] / nu
            
    gamma_hat_cl [ell] = gamma.rvs(a, scale = b)

  return gamma_hat_cl
