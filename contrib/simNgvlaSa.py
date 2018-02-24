
#
# January 2018 bsm (nrao)
#  script to simulate ngvla short spacing array (this one is noise only)
#  this does a fixed specifiable integration time on a single pointing
#  (specified in the ngvlaPtgFile.txt)
#
# This works but the dirty beam looks odd to me, needs looking into.
#  Early runs I had problems with the beam being elliptical but that
#  was a coordinate system problem with the config file, now solved.
#
# updated results (9jan18 bsm) 
#  11.07"x10.15" beam (quick psf)
#  10.64"x9.83" (imhead of PSF) --> 10.23" FWHM PSF @ 100 GHz.
#

# set this - names your outputs->
myproj="ngvlaSaTest"
# provide an input file if that's how you roll (check units!) ->
skmod="trueblank.fits"
# recenter input if you want to ->
in_map_center="J2000 08h37m27.2s 17d38m44.7s"
phase_center=in_map_center
#integ time-
totaltime="1000s"
my_cfg = "ngvlaSA_2b_utm"
out_vis = myproj+'/'+myproj+'.'+my_cfg
real_cfg=my_cfg+'.cfg'

# create MS-
default("simobserve")
project            =  myproj
indirection=in_map_center
skymodel         =  skmod
incell="1.5arcsec"
#inbright         = pkpixel
inbright="unchanged"
incenter="100.0GHz"
inwidth="50MHz"
# have simobserve calculate mosaic pointing locations:
setpointings       =  False
ptgfile = "ngvlaPtgFile.txt"
hourangle = "transit"
integration        =  "10s"
mapsize            =  "1arcmin"
obsmode            =  "int"
graphics           =  "both"
thermalnoise       = ""
refdate = "2017/12/01"
antennalist        =  real_cfg
totaltime          =  totaltime
thermalnoise       = ""
simobserve()

# dirty image
default('tclean')
vis=[out_vis+'.ms']
imagename=out_vis+'_dirty'
#imagermode='mosaic'
gridder='mosaic'
niter=0
phasecenter=phase_center
interactive=False
pbcor=False
imsize             =  [250,250]
cell               =  '1.5arcsec'
tclean()

