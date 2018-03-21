# galaxy model, 5 channels with first and last channels empty:
# ./mkgalcube run=model10 beam=0 vrange=150 nvel=5
# fails at feather
#
# default now is just a single channel pulled out of galaxy model 0
#
#  it is assumed you have done    execfile('ngvla.py')
#
# @todo figure out regression for this test

test 		 = 'test2'
model        = '../models/model0.fits'           # this as phasecenter with dec=-30 for ALMA sims
phasecenter  = 'J2000 180.000000deg 40.000000deg'

# pick the piece of the model to image, and at what pixel size
imsize_m     = 192
pixel_m      = 0.1

# pick the sky imaging parameters (for tclean)
imsize_s     = 512
pixel_s      = 0.1

# pick a few niter values for tclean to check flux convergence 
niter = [0,1000, 2000]

# decide if you want the whole cube (chans=-1) or just a specific channel
chans        = '-1' # must be a string. for a range of channels --> '24~30'


# -- do not change parameters below this ---
import sys
for arg in ng_argv(sys.argv):
    exec(arg)

# rename model variable if single channel (or range) has been chosen so we don't overwrite models 
if chans != '-1':
    model_out = '%sa.image'%model[:model.rfind('.fits')]
    # delete any previously made models otherwise imsubimage won't run
    os.system('rm -fr %s'%model_out)
    # imsubimage to pull out the selected channel(s)
    imsubimage(model, model_out, chans=chans)
    # rewrite the model variable with our new model
    model = model_out

ptg = test + '.ptg'              # use a single pointing mosaic for the ptg
if type(niter) != type([]): niter = [niter]


# report
ng_log('TEST: %s' % test)
ng_begin(test)
ng_version()

# create a single pointing mosaic
ng_ptg(phasecenter,ptg)

# create a MS based on a model and antenna configuration
ng_log('VLA')
ng_vla(test,model,imsize_m,pixel_m,cfg='../SWcore',ptg=ptg, phasecenter=phasecenter)

# clean this interferometric map a bit
ng_log('CLEAN')
ng_clean1(test, test+'/'+test+'.SWcore.ms', imsize_s, pixel_s, phasecenter=phasecenter, niter=niter)

# create two OTF maps 
ng_log('OTF')
ng_tp_otf(test,test+'/'+test+'.SWcore.skymodel', 45.0, label='45')
ng_tp_otf(test,test+'/'+test+'.SWcore.skymodel', 18.0, label='18')

# combine TP + INT using feather, for all niters
ng_log('FEATHER')
for idx in range(len(niter)):
	ng_feather(test,label='45',niteridx=idx)
	ng_feather(test,label='18',niteridx=idx)


# # smooth out skymodel image with feather beam so we can compare feather to original all in jy/beam
# ng_log('SMOOTH')
# for idx in range(len(niter)):
#     ng_smooth(test+'/clean1', test+'/'+test+'.SWcore.skymodel', label='18', niteridx=idx)
#     ng_smooth(test+'/clean1', test+'/'+test+'.SWcore.skymodel', label='45', niteridx=idx)

ng_log('ANALYZE')
for idx in range(len(niter)):
    ng_analyze(test, 'dirtymap', niteridx=idx)
    ng_analyze(test, 'feather18', niteridx=idx)
    ng_analyze(test, 'feather45', niteridx=idx)


#
ng_end()


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
