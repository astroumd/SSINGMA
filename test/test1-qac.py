#
#  it is assumed you have done    execfile('ngvla.py')
# 
#  This test takes about 800 MB of disk space, and needs about 2 GB memory
#
# 667.692u 20.628s 9:45.15 117.6%	0+0k 1121096+3180192io 335pf+0w     niter=0
# 2073.348u 37.540s 30:59.81 113.4%	0+0k 2335376+3269568io 887pf+0w     niter=[0,1000,2000]

test         = 'test1'
model        = '../models/skymodel.fits'           # this has phasecenter with dec=-30 for ALMA sims
phasecenter  = 'J2000 180.000000deg 40.000000deg'  # so modify this for ngVLA

# pick the piece of the model to image, and at what pixel size
# natively this model is 4096 pixels at 0.05"
imsize_m     = 4096
pixel_m      = 0.005    # 0.01 was the bench

# pick the sky imaging parameters (for tclean)
imsize_s     = 512
pixel_s      = 0.1     # 0.25 was the bench

# pick a few niter values for tclean to check flux convergence 
niter        = [0,1000,2000]

# -- do not change parameters below this ---
import sys
for arg in qac_argv(sys.argv):
    exec(arg)

ptg = test + '.ptg'              # use a single pointing mosaic for the ptg
if type(niter) != type([]): niter = [niter]

    
# report
qac_log("TEST: %s" % test)
qac_begin(test)
qac_version()

# create a single pointing mosaic
qac_ptg(phasecenter,ptg)

# create a MS based on a model and antenna configuration
qac_log("VLA")
qac_vla(test,model,imsize_m,pixel_m,cfg=1,ptg=ptg, phasecenter=phasecenter)

# clean this interferometric map a bit
qac_log("CLEAN")
qac_clean1(test+'/clean1',test+'/'+test+'.SWcore.ms',  imsize_s, pixel_s, phasecenter=phasecenter,niter=niter)

# create two OTF TP maps
qac_log("OTF")
qac_tp_otf(test+'/clean1',test+'/'+test+'.SWcore.skymodel', 45.0, label="45")
qac_tp_otf(test+'/clean1',test+'/'+test+'.SWcore.skymodel', 18.0, label="18")

# combine TP + INT using feather, for all niter's
qac_log("FEATHER")
for idx in range(len(niter)):
    qac_feather(test+'/clean1',label="45",niteridx=idx)
    qac_feather(test+'/clean1',label="18",niteridx=idx)

#
# --------------------------------------------------------------------------------------------------------------
# regression

regress51 = [
    "0.0067413167369070152 0.010552344105428218 0.0 0.10000000149011612 113100.52701950417",
    "376.81918701712272 791.0277433970175 0.25461947454935646 20152.279939787601 0.0",
    "2.0449149476928925 22.901153529996495 -33.027679443359375 96.835914611816406 1479.648825106718",
    "6570.3070261008925 8953.8105374926636 309.4671630859375 24888.931640625 60026.507913198388",
    "42668.1077336737 44654.739711457922 13273.1376953125 67745.1171875 70071.789310851134",
    "6570.3070261008925 8953.8105374926636 309.4671630859375 24888.931640625 60026.507913198388",
    "42668.1077336737 44654.739711457922 13273.1376953125 67745.1171875 70071.789310851134",
    "14.774882361675623 23.377433067737787 -7.3271188735961914 137.7615966796875 63317.084178311772",
    "25.239792422269797 33.841414373223429 -36.166255950927734 180.94865417480469 108163.97872576566",
    "9.9542300510560686 17.099277097148892 -17.992568969726562 114.72189331054688 42658.398669071932",
    "19.474530926399115 28.794431828862734 -30.066720962524414 181.16015625 83457.213655954009",
]

r = regress51


qac_log("**** REGRESSION STATS ****")

# regression
qac_stats(model,                                 r[0])
qac_stats('test1/test1.SWcore.ms',               r[1])
qac_stats('test1/clean1/dirtymap.image',         r[2])
qac_stats('test1/clean1/otf18.image.pbcor')
qac_stats('test1/clean1/otf45.image.pbcor')
qac_stats('test1/clean1/dirtymap.image.pbcor')
qac_stats('test1/clean1/dirtymap_2.image.pbcor')
qac_stats('test1/clean1/dirtymap_3.image.pbcor')
qac_stats('test1/clean1/feather18.image.pbcor')
qac_stats('test1/clean1/feather18_2.image.pbcor')
qac_stats('test1/clean1/feather18_3.image.pbcor')
qac_stats('test1/clean1/feather45.image.pbcor')
qac_stats('test1/clean1/feather45_2.image.pbcor')
qac_stats('test1/clean1/feather45_3.image.pbcor')

# done
qac_end()
