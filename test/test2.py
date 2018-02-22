# this is for model 10 --> just 5 channels with first and last channels empty
# ./mkgalcube run=model10 beam=0 vrange=150 nvel=5
# 
# failed at ng_feather
#
#  it is assumed you have done    execfile('ngvla.py')
#

model        = '../models/model0.fits'           # this as phasecenter with dec=-30 for ALMA sims
phasecenter  = 'J2000 180.000000deg 40.000000deg'
ptg          = 'test2.ptg'

# pick the piece of the model to image, and at what pixel size
imsize_m     = 192
pixel_m      = 0.1

# pick the imaging parameters 
imsize_s     = 512
pixel_s      = 0.1

# decide if you want the whole cube (chans=-1) or just a specific channel
chans        = '24' # must be a string. for a range of channels --> '24~30'

if chans != -1:
    model_out = '%sa.image'%model[:model.rfind('.fits')]
    # delete any previously made models otherwise imsubimage won't run
    os.system('rm -fr %s'%model_out)
    # imsubimage to pull out the selected channel(s)
    imsubimage(model, model_out, chans=chans)
    # rewrite the model variable with our new model
    model = model_out


if True:
    # need a better way?  an ng_()
    fp = open(ptg,"w")
    fp.write("%s" % phasecenter)
    fp.close()


# create a MS based on a model and antenna configuration
ng_vla('test2',model,imsize_m,pixel_m,cfg='../SWcore',ptg=ptg, phasecenter=phasecenter)

# clean this interferometric map a bit
ng_clean1('test2/clean1','test2/test2.SWcore.ms',  imsize_s, pixel_s, phasecenter=phasecenter,niter=[0,1000])

# create two OTF maps 
ng_tp_otf('test2/clean1','test2/test2.SWcore.skymodel', 45.0, label="45")
ng_tp_otf('test2/clean1','test2/test2.SWcore.skymodel', 18.0, label="18")

# combine TP + INT using feather
ng_feather('test2/clean1',label="45")
ng_feather('test2/clean1',label="18")

ng_feather('test2/clean1', 'test2/clean1/dirtymap_2.image', label='45')
ng_feather('test2/clean1', 'test2/clean1/dirtymap_2.image', label='18')

#
print "Done!"

# --------------------------------------------------------------------------------------------------------------
# regression

# regress51 = [
    # "0.0067413167369069988 0.010552344105427177 0.0 0.10000000149011612 113100.52701950389",
    # "411.08972165273946 821.42796910126435 0.070715504411804908 21357.570702738558 0.0",
    # ]
# 
# 
# r = regress51
    # 
# 
# regression
# ng_stats(model,                                 r[0])
# ng_stats('test1/test1.SWcore.ms',               r[1])
# ng_stats('test1/clean1/dirtymap.image')
# ng_stats('test1/clean1/otf45.image')
# ng_stats('test1/clean1/otf18.image.pbcor')
# ng_stats('test1/clean1/otf45.image')
# ng_stats('test1/clean1/otf18.image.pbcor')
# ng_stats('test1/clean1/feather.image')
# ng_stats('test1/clean1/feather.image.pbcor')
# 