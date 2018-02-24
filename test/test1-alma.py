#
#
#  like test1, but for alma, just for kicks
#  The 12m PB is 50" (FWHM)
#       7m PB is 85"
#

test         = 'test1-alma'
model        = '../models/skymodel.fits'           # this has phasecenter with dec=-30 for ALMA sims
phasecenter  = 'J2000 180.000000deg -30.000000deg'  # so modify this for ngVLA

# pick the piece of the model to image, and at what pixel size
# natively this model is 4096 pixels at 0.05"
imsize_m     = 4096
pixel_m      = 0.01

# pick the sky imaging parameters (for tclean)
imsize_s     = 256
pixel_s      = 0.5

# pick a few niter values for tclean to check flux convergence 
niter        = [0,500]

# -- do not change parameters below this ---
import sys
for arg in ng_argv(sys.argv):
    exec(arg)

ptg = test + '.ptg'              # use a single pointing mosaic for the ptg
    

# report
ng_version()

# create a single pointing mosaic
ng_ptg(phasecenter,ptg)

# create a MS based on a model and antenna configuration
ng_alma(test,model,imsize_m,pixel_m,cycle=5,cfg=1,ptg=ptg, phasecenter=phasecenter)

# find out which MS we got:
import glob
mslist = glob.glob(test+"/*.ms")
print "MS files found: ",mslist

# clean this interferometric map a bit
ng_clean1(test+'/clean1',mslist,  imsize_s, pixel_s, phasecenter=phasecenter,niter=niter)
ng_clean1(test+'/clean2',mslist,  imsize_s, pixel_s, phasecenter=phasecenter,niter=niter,startmodel=model)

# create two OTF TP maps 
ng_tp_otf(test+'/clean1',test+'/'+test+'.SWcore.skymodel', 12.0)

# combine TP + INT using feather, for all niter's
for idx in range(len(niter)):
    ng_feather(test+'/clean1',niteridx=idx)

#
print "Done!"
