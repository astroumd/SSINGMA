# galaxy model, 5 channels with first and last channels empty:
# ./mkgalcube run=model10 beam=0 vrange=150 nvel=5
# fails at feather
#
# default now is just a single channel pulled out of galaxy model 0
# does not fail
#
#  it is assumed you have done    execfile('ngvla.py')
#
# @todo figure out regression for this test

model        = '../models/model0.fits'           # this as phasecenter with dec=-30 for ALMA sims
phasecenter  = 'J2000 180.000000deg 40.000000deg'
ptg          = 'test2.ptg'

# pick the piece of the model to image, and at what pixel size
imsize_m     = 192
pixel_m      = 0.1

# pick the sky imaging parameters (for tclean)
imsize_s     = 512
pixel_s      = 0.1

niter = [0,1000,2000]

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

# create a single pointing mosaic
ng_ptg(phasecenter,ptg)

# create a MS based on a model and antenna configuration
ng_vla('test2',model,imsize_m,pixel_m,cfg='../SWcore',ptg=ptg, phasecenter=phasecenter)

# clean this interferometric map a bit
ng_clean1('test2/clean1','test2/test2.SWcore.ms',  imsize_s, pixel_s, phasecenter=phasecenter,niter=niter)

# create two OTF maps 
ng_tp_otf('test2/clean1','test2/test2.SWcore.skymodel', 45.0, label="45")
ng_tp_otf('test2/clean1','test2/test2.SWcore.skymodel', 18.0, label="18")

# combine TP + INT using feather
ng_feather('test2/clean1',label="45")
ng_feather('test2/clean1',label="18")

# combine TP + INT using feather on cleaned images
ng_feather('test2/clean1', label='45', niteridx=1)
ng_feather('test2/clean1', label='18', niteridx=1)
ng_feather('test2/clean1', label='45', niteridx=2)
ng_feather('test2/clean1', label='18', niteridx=2)

# smooth out skymodel image with feather beam so we can compare feather to original all in jy/beam
ng_smooth('test2/clean1', 'test2/test2.SWcore.skymodel', label='18', niteridx=2)
ng_smooth('test2/clean1', 'test2/test2.SWcore.skymodel', label='45', niteridx=2)


#
print "Done!"

# --------------------------------------------------------------------------------------------------------------
# regression

# regress51 = [
#     "1.6544389694376587e-05 0.0002642084282218718 0.0 0.0098144030198454857 0.60989238169349846"
#     ]


# r = regress51
    

# # regression
# ng_stats(model,                                 r[0])
# # ng_stats('test2/test2.SWcore.ms',               r[1])
# ng_stats('test2/clean1/dirtymap.image')
# ng_stats('test2/clean1/dirtymap_2.image')
# ng_stats('test2/clean1/otf45.image')
# ng_stats('test2/clean1/otf18.image.pbcor')
# ng_stats('test2/clean1/otf45.image')
# ng_stats('test2/clean1/otf18.image.pbcor')
# ng_stats('test2/clean1/feather18_2.image')
# ng_stats('test2/clean1/feather18_2.image.pbcor')
# ng_stats('test2/clean1/feather45_2.image')
# ng_stats('test2/clean1/feather45_2.image.pbcor')

# ng_stats('test2/clean1/feather18.image.pbcor')
# ng_stats('test2/clean1/feather18_2.image.pbcor')
# ng_stats('test2/clean1/feather45.image.pbcor')
# ng_stats('test2/clean1/feather45_2.image.pbcor')
