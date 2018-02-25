#

test         = 'test1-SSA'
model        = '../models/skymodel.fits'           # this has phasecenter with dec=-30 for ALMA sims
phasecenter  = 'J2000 180.000000deg 40.000000deg'  # so modify this for ngVLA

# pick the piece of the model to image, and at what pixel size
# natively this model is 4096 pixels at 0.05"
imsize_m     = 4096
pixel_m      = 0.01

# pick the sky imaging parameters (for tclean)
imsize_s     = 512
pixel_s      = 0.25

# pick a few niter values for tclean to check flux convergence 
niter        = 0

# -- do not change parameters below this ---
import sys
for arg in ng_argv(sys.argv):
    exec(arg)

ptg = test + '.ptg'              # use a single pointing mosaic for the ptg
    
cfg1 = '../contrib/ngvlaSA_2b'
cfg2 = '../contrib/ngvlaSA_2b_utm'
# Note: diameters in configuration file will not be used - PB for NGVLA will be used

cfg = cfg1
msfile   = '.' + cfg[cfg.rfind('/')+1:]  + '.ms'
skymodel = '.' + cfg[cfg.rfind('/')+1:]  + '.skymodel'

# report
ng_version()

# create a single pointing mosaic
ng_ptg(phasecenter,ptg)

# create a MS based on a model and antenna configuration
ng_vla(test,model,imsize_m,pixel_m,cfg=cfg,ptg=ptg, phasecenter=phasecenter)

# clean this interferometric map a bit
ng_clean1(test+'/clean1',test+'/'+test+msfile,  imsize_s, pixel_s, phasecenter=phasecenter,niter=niter)

#
print "Done!"

