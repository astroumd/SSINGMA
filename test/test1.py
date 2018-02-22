#
#  it is assumed you have done    execfile('ngvla.py')
# 
#  This test takes about 800 MB of disk space, and needs about 2 GB memory
#
# 667.692u 20.628s 9:45.15 117.6%	0+0k 1121096+3180192io 335pf+0w     niter=0
# 2073.348u 37.540s 30:59.81 113.4%	0+0k 2335376+3269568io 887pf+0w     niter=[0,1000,2000]


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
ng_clean1('test1/clean1','test1/test1.SWcore.ms',  imsize_s, pixel_s, phasecenter=phasecenter,niter=[0,1000,2000])

# create two OTF maps 
ng_tp_otf('test1/clean1','test1/test1.SWcore.skymodel', 45.0, label="45")
ng_tp_otf('test1/clean1','test1/test1.SWcore.skymodel', 18.0, label="18")

# combine TP + INT using feather
ng_feather('test1/clean1',label="45")
ng_feather('test1/clean1',label="18")

#
print "Done!"

# --------------------------------------------------------------------------------------------------------------
# regression

regress51 = [
    "0.0067413167369069988 0.010552344105427177 0.0 0.10000000149011612 113100.52701950389",
    "411.08972165273946 821.42796910126435 0.070715504411804908 21357.570702738558 0.0",
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


ng_log("**** REGRESSION STATS ****")

# regression
ng_stats(model,                                 r[0])
ng_stats('test1/test1.SWcore.ms',               r[1])
ng_stats('test1/clean1/dirtymap.image',         r[2])
ng_stats('test1/clean1/otf45.image',            r[3])
ng_stats('test1/clean1/otf18.image.pbcor',      r[4])
ng_stats('test1/clean1/otf45.image',            r[5])
ng_stats('test1/clean1/otf18.image.pbcor',      r[6])
ng_stats('test1/clean1/feather45.image',        r[7])
ng_stats('test1/clean1/feather45.image.pbcor',  r[8])
ng_stats('test1/clean1/feather18.image',        r[9])
ng_stats('test1/clean1/feather18.image.pbcor',  r[10])
