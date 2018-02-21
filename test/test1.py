#
#  it is assumed you have done    execfile('ngvla.py')
# 


model        = '../models/skymodel.fits'           # this as phasecenter with dec=-30 for ALMA sims
phasecenter  = 'J2000 180.000000deg 40.000000deg'
ptg          = 'test1.ptg'

# pick the piece of the model to image, and at what pixel size
imsize_m     = 4096
pixel_m      = 0.01

# pick the imaging parameters 
imsize_s     = 512
pixel_s      = 0.25


if True:
    # need a better way?  an ng_()
    fp = open(ptg,"w")
    fp.write("%s" % phasecenter)
    fp.close()

# create a MS based on a model and antenna configuration
ng_vla('test1',model,imsize_m,pixel_m,cfg='../SWcore',ptg=ptg, phasecenter=phasecenter)

# clean this interferometric map a bit
ng_clean1('test1/clean1','test1/test1.SWcore.ms',  imsize_s, pixel_s, phasecenter=phasecenter,niter=1000)

# create two OTF maps 
ng_tp_otf('test1/clean1',model, 45.0, label="45")
ng_tp_otf('test1/clean1',model, 18.0, label="18")

# combine TP + INT using feather
ng_feather('test1/clean1',label="45")
ng_feather('test1/clean1',label="18")
