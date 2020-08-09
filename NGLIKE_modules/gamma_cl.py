from cosmosis.datablock import names, option_section
from get_gamma_cl import gamma_draw_Cl
#import time

def setup(options):

  ndraw = options[option_section, "ndraw"]
  setup_var = options[option_section, "setup_var"]

  return ndraw, setup_var


def execute(block, config):

  ndraw, var = config

  if var == 0:
    in_ell = block ["galaxy_cl", "ell"]

    is_auto = block ["galaxy_cl", "is_auto"]
    nbin = block ["galaxy_cl", "nbin"]
    nbin_a = block ["galaxy_cl", "nbin_a"]
    nbin_b = block ["galaxy_cl", "nbin_b"]
    sample_a = block ["galaxy_cl", "sample_a"]
    sample_b = block ["galaxy_cl", "sample_b"]
    save_name = block ["galaxy_cl", "save_name"]
    sep_name = block ["galaxy_cl", "sep_name"]

    block ["multi_galaxy_cl", "is_auto"] = is_auto
    block ["multi_galaxy_cl", "nbin"] = nbin
    block ["multi_galaxy_cl", "nbin_a"] = nbin_a
    block ["multi_galaxy_cl", "nbin_b"] = nbin_b
    block ["multi_galaxy_cl", "sample_a"] = sample_a
    block ["multi_galaxy_cl", "sample_b"] = sample_b
    block ["multi_galaxy_cl", "save_name"] = save_name
    block ["multi_galaxy_cl", "sep_name"] = sep_name

#st_time = time.time ()
    for bin_a in range (1, nbin_a+1):
      for bin_b in range (1, bin_a+1):

        in_Cl = block ["galaxy_cl", "bin_"+str(bin_a)+"_"+str(bin_b)]

        for n in range (1, ndraw+1):
          gamma_cls = gamma_draw_Cl (in_ell, in_Cl);
          block ["multi_galaxy_cl", "bin_"+str(bin_a)+"_"+str(bin_b)+"_"+str(n)] = gamma_cls

    block ["multi_galaxy_cl", "ell"] = in_ell
#end_time = time.time ()
#print (end_time - st_time, "SECONDS ARE PASSED")

  return 0

def cleanup (config):
  pass
