#
#  it is assumed you have done    execfile('ngvla.py')
# 


model        = '../models/skymodel.fits'           # this as phasecenter with dec=-30 for ALMA sims
phasecenter  = 'J2000 180.000000deg 40.000000deg'
ptg          = 'test1.ptg'

imsize       = 1024
pixel        = 0.01

imsize_s     = 512
pixel_s      = 0.25


if True:
    # need a better way?
    fp = open(ptg,"w")
    fp.write("%s" % phasecenter)
    fp.close()

#
ng_vla('test1',model,imsize,pixel,cfg='../SWcore',ptg=ptg, phasecenter=phasecenter)
#
ng_clean1('test1/clean1','test1/test1.SWcore.ms',  imsize_s, pixel_s, phasecenter=phasecenter,niter=1000)
#
ng_tp_otf('test1/clean1',model, 45.0, label="45")
ng_tp_otf('test1/clean1',model, 18.0, label="18")
#
ng_feather('test1/clean1',label="45")
ng_feather('test1/clean1',label="18")
