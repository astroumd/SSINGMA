#  NG:   ngVLA modeling helper functions for CASA
#
#
#           execfile('ngvla.py')
#
#  There are a series of nv_*() functions, and one static class NG.* with pure helper functions
#
# See also : https://casaguides.nrao.edu/index.php/Analysis_Utilities
# (we currently don't assume you have installed these 'au' tools, but they are very useful)
#
# See also the qtp_() routines in TP2VIS
#
# 
#
import os, shutil, math, tempfile
import os.path
# from buildmosaic import buildmosaic
# from utils import constutils as const
import numpy as np
# import numpy.ma as ma
# import pyfits as fits
import matplotlib.pyplot as plt

# this is dangerous, creating some convenient numbers in global namespace, but here they are...
cqa  = qa.constants('c')                  # (turns out to be in m/s)
cms  = qa.convert(cqa,"m/s")['value']     # speed of light, forced in m/s (299792458.0)
apr  = 180.0 * 3600.0 / np.pi             # arcsec per radian (206264.8)
bof  = np.pi / (4*math.log(2.0))          # beam oversampling factor (1.1331) : NPPB = bof * (Beam/Pixel)**2  [cbm in tp2vis.py]
stof = 2.0*np.sqrt(2.0*np.log(2.0))       # FWHM=stof*sigma  (2.3548)

# nasty globals
#restoringbeam = 'common'                # common beam for all planes
restoringbeam = None                     # given the edge channel issue, a common beam is not a good idea

def ng_version():
    """ ng helper functions """
    print "ngvla: version 19-feb-2018"
    print "casa:",casa['version']         # there is also:   cu.version_string()
    print "data:",casa['dirs']['data']    

def ng_log(message, verbose=True):
    """ ng banner message; can be turned off
    """
    if verbose:
        print ""
        print "========= NGVLA: %s " % message
        print ""


def ng_tmp(prefix, tmpdir='.'):
    """ Create a temporary file in a tmpdir

        Parameters
        ----------
        prefix : str
           starting name of the filename in <tmpdir>/<pattern>

        tmpdir

        Returns
        -------
        Unique filename
    """
    fd = tempfile.NamedTemporaryFile(prefix=prefix,dir=tmpdir,delete='false')
    name = fd.name
    fd.close()
    return name
    


def ng_stats(image, test = None, eps=None, box=None, pb=None, pbcut=0.8, edge=False):
    """ summary of some stats in an image or measurement set
        in the latter case the flux is always reported as 0

        This routine can also be used for regression testing (see test=)

        image     image file name (CASA, FITS, MIRIAD)
        test      expected regression string
        eps       if given, it should parse the test string into numbers, each number
                  needs to be within relative error "eps", i.e. abs(v1-v2)/abs(v) < eps
        box       if used, this is the box for imstat()   box='xmin,ymin,xmax,ymax'
        pb        optional pb file, if the .image -> .pb would not work
        pbcut     only used for images, and a .pb should be parallel to the .image file
                  or else it will be skipped
        edge      take off an edge channel from either end (not implemented)

        Output should contain:   mean,rms,min,max,flux
    """
    def text2array(text):
        a = text.split()
        b = np.zeros(len(a))
        for i,ai in zip(range(len(a)),a):
            b[i] = float(ai)
        return b
    def arraydiff(a1,a2):
        delta = abs(a1-a2)
        idx = np.where(delta>0)
        return delta[idx]/a1[idx]
    def lel(name):
        """ convert filename to a safe filename for LEL expressions, e.g. in mask=
        """
        return '\'' + name + '\''
    

    NG.assertf(image)

    if NG.iscasa(image + '/ANTENNA'):                      # assume it's a MS
        tb.open(image)
        data  = np.abs(tb.getcol('DATA')[0,:,:])  # first pol ->  data[nchan,nvis]
        mean = data.mean()
        rms  = data.std()
        min  = data.min()
        max  = data.max()
        flux = 0.0
        tb.close()
        del data
    else:                                    # assume it's an IM
        maskarea = None
        if pbcut != None:
            # this requires a .pb file to be parallel to the .image file
            if pb == None:
                pb = image[:image.rindex('.')] + '.pb'
                if NG.iscasa(pb):
                    maskarea = lel(pb) + '>' + str(pbcut)      # create a LEL for the mask
            else:
                maskarea = lel(pb) + '>' + str(pbcut)
        if edge:
            nchan = imhead(image)['shape'][3]
            s0 = imstat(image,mask=maskarea,chans='1~%d' % (nchan-2),box=box)
        else:
            s0 = imstat(image,box=box,mask=maskarea)
        # mean, rms, min, max, flux
        # @TODO   this often fails
        mean = s0['mean'][0]
        rms  = s0['rms'][0]
        min  = s0['min'][0]
        max  = s0['max'][0]
        if 'flux' in s0:
            flux = s0['flux'][0]
        else:
            flux = s0['sum'][0]
    test_new = "%s %s %s %s %s" % (repr(mean),repr(rms),repr(min),repr(max),repr(flux))
    if test == None:
        test_out = ""
        report = False
    else:
        if eps == None:
            if test_new == test:
                test_out = "OK"
                report = False
            else:
                test_out = "FAILED regression"
                report = True
        else:
            v1 = text2array(test_new)
            v2 = text2array(test)
            delta = arraydiff(v1,v2)
            print delta
            if delta.max() < eps:
                test_out = "OK"
                report = False
            else:
                test_out = "FAILED regression delta=%g > %g" % (delta.max(),eps)
                report = True
    msg1 = "NG_STATS: %s" % (image)
    print "%s %s %s" % (msg1,test_new,test_out)
    if report:
        fmt1 = '%%-%ds' % (len(msg1))
        msg2 = fmt1 % ' '
        print "%s %s EXPECTED" % (msg2,test)
    
    #-end of ng_stats()
    
def ng_beam(im, normalized=False, plot=None):
    """ some properties of the PSF

    im:           image representing the beam (usually a .psf file)
    normalized:   if True, axes are arcsec and normalized flux
    plot:         if set, this is the plot created, usually a png

    @todo   have an option to just print beam, no volume info
    """
    if not NG.iscasa(im):
        print "NG_BEAM: missing %s " % im
        return

    h0 = imhead(im)
    pix2 = abs(h0['incr'][0] * h0['incr'][1] * apr * apr)      # pixel**2 (in arcsec)
    if 'perplanebeams' in h0:
        bmaj = h0['perplanebeams']['beams']['*0']['*0']['major']['value']
        bmin = h0['perplanebeams']['beams']['*0']['*0']['minor']['value']
        pix  = sqrt(pix2)
        nppb =  bof * bmaj*bmin/pix2        
    elif 'restoringbeam' in h0:
        bmaj = h0['restoringbeam']['major']['value']
        bmin = h0['restoringbeam']['minor']['value']
        pix  = sqrt(pix2)        
        nppb =  bof * bmaj*bmin/pix2        
    else:
        bmaj = 1.0
        bmin = 1.0
        pix  = 1.0        
        nppb = 1.0

    if normalized:
        factor = nppb
    else:
        factor = 1.0
        pix    = 1.0

    print "NG_BEAM: %s  %g %g %g %g %g" % (im,bmaj,bmin,pix,nppb,factor)

    xcen = h0['refpix'][0]
    ycen = h0['refpix'][1]
    nx = h0['shape'][0]
    ny = h0['shape'][1]
    nz = max(h0['shape'][2],h0['shape'][3])
    size = np.arange(nx/2-20)
    flux = 0.0 * size
    zero = flux * 0.0
    for i in size:
        box = '%d,%d,%d,%d' % (xcen-i,ycen-i,xcen+i,ycen+i)
        flux[i] = imstat(im,chans='0',box=box)['sum'][0]/factor
    print "NG_BEAM: Max/Last/PeakLoc",flux.max(),flux[-1],flux.argmax()*pix
    
    if plot != None:
        plt.figure()
        if normalized:
            plt.title("%s : Normalized cumulative flux" % im)
            plt.xlabel("size/2 (arcsec)")
            plt.ylabel("Flux")
            size = size * sqrt(pix2)
        else:
            plt.title("%s : Cumulative sum" % im)
            plt.xlabel("size/2 (pixels)")
            plt.ylabel("Sum")
        plt.plot(size,flux)
        plt.plot(size,zero)
        plt.savefig(plot)
        plt.show()
    
    #-end of ng_beam()
    
    
def ng_getuv(ms, kwave=True):
    """ return the UV coordinates, in m or kilowaves

    ms       MS file, No default
    
    kwave    boolean, if true (u,v) in klambda, else in native meter
             Default:  True

    Usage:   (u,v) = ng_getuv('msfile',True)
    """
    tb.open(ms)
    uvw  = tb.getcol('UVW')
    tb.close()
    if kwave:
        tb.open(ms + '/SPECTRAL_WINDOW')
        chan_freq = tb.getcol('CHAN_FREQ')
        ref_freq = (chan_freq[0] + chan_freq[-1])/2.0
        factor = ref_freq / cms / 1000.0
        factor = factor[0]                  # assume/ignore polarization dependent issues
        tb.close()
    else:
        factor = 1.0

    print "UVW shape",uvw.shape,uvw[:,0],factor
    u = uvw[0,:] * factor                   # uvw are in m. we want m
    v = uvw[1,:] * factor                   # or klambda
    uvd = np.sqrt(u*u+v*v)
    print "UVD npts,min/max = ",len(uvd), uvd.min(), uvd.max()

    return (u,v)

    #-end of ng_getuv()
    
def ng_getamp(ms, record=0):
    """ return the AMP for each channel for the (0,0) spacings.
    It needs to sum for all fields where uv=(0,0)

    ms       MS file, No default
    
    Usage:   amp = ng_getamp('msfile')
    """
    tb.open(ms)
    uvw  = tb.getcol('UVW')[0:2,:]               # uvw[2,nvis]
    idx = np.where( np.abs(uvw).min(axis=0) == 0 )[0]

    data  = tb.getcol('DATA')[0,:,idx]      # getcol() returns  [npol,nchan,nvis]
                                            # but with idx it returns [nvisidx,nchan]
    amp = np.abs(data.max(axis=0))          # strongest field
    amp = np.abs(data.sum(axis=0))/2.0      # sum for all fields (but they overlap, so guess 2.0)
    tb.close()
    return amp

    #-end of ng_getamp()

    
def ng_alma(project, skymodel, imsize=512, pixel=0.5, phasecenter=None, freq=None, cycle=5, cfg=0, niter=-1, ptg = None):
    """
    helper function to create an MS from a skymodel for a given ALMA configuration
    See CASA/data/alma/simmos/ for the allowed (cycle,cfg) pairs

    cycle 1:   ALMA cfg = 1..6    ACA ok
    cycle 2:   ALMA cfg = 1..7    ACA bleeh ('i' and 'ns')
    cycle 3:   ALMA cfg = 1..8    ACA ok
    cycle 4:   ALMA cfg = 1..9    ACA ok
    cycle 5:   ALMA cfg = 1..10   ACA ok [same as 4]

    if niter>=0 is chosen, tclean(imagename='dirtyimage') is used, overwriting any previous dirtyimage
    """
    
    data_dir = casa['dirs']['data']                  # data_dir + '/alma/simmos' is the default location for simobserve
    if cfg==0:
        cfg = 'aca.cycle%d' % (cycle)                # cfg=0 means ACA (7m)
    else:
        cfg = 'alma.cycle%d.%d' % (cycle,cfg)        # cfg>1 means ALMA (12m)

    print "CFG: ",cfg

    ng_vla(project,skymodel,imsize,pixel,phasecenter,freq,cfg,niter,ptg)
    
    #-end of ng_alma()

    
def ng_vla(project, skymodel, imsize=512, pixel=0.5, phasecenter=None, freq=None, cfg=None, niter=-1, ptg = None):
    """
    helper function to create an MS from a skymodel for a given ngVLA configuration
    Example ngVLA  configurations:

    SWcore.cfg
    SW214.cfg
    SWVLB.cfg
    """

    # for tclean (only used if niter>=0)
    imsize    = NG.imsize2(imsize)
    cell      = ['%garcsec' % pixel]
    outms     = '%s/%s.%s.ms' % (project,project,cfg[cfg.rfind('/')+1:])
    outim     = '%s/dirtymap' % (project)
    do_fits   = True       # always output fits when you clean


    if ptg != None:
        setpointings = False
        ptgfile      = ptg
    # obsmode     = "int"
    antennalist = "%s.cfg" % cfg     # can this be a list?

    totaltime   = "28800s"     # 4 hours  (should be multiple of 2400 ?)
    integration = "30s"        # prevent too many samples for MS

    thermalnoise= ""
    verbose     = True
    overwrite   = True
    graphics    = "file"       # "both" would do "screen" as well
    user_pwv    = 0.0
    incell      = "%garcsec" % pixel
    mapsize     = ["%garcsec" % (pixel*imsize[0])  ,"%garcsec"  % (pixel*imsize[1]) ]
    
    # we allow accumulation now ..
    # ...make sure old directory is gone
    # ...os.system("rm -rf %s" % project)

    if ptg == None:
        simobserve(project, skymodel,
               indirection=phasecenter,
               incell=incell,
               mapsize=mapsize,
               integration=integration,
               totaltime=totaltime,
               antennalist=antennalist,
               verbose=verbose, overwrite=overwrite,
               user_pwv = 0.0, thermalnoise= "")
    else:
        simobserve(project, skymodel,
               setpointings=False, ptgfile=ptgfile,
               indirection=phasecenter,                   
               incell=incell,
               mapsize=mapsize,
               integration=integration,
               totaltime=totaltime,
               antennalist=antennalist,
               verbose=verbose, overwrite=overwrite,                   
               user_pwv = 0.0, thermalnoise= "")

    if niter >= 0:
        cmd1 = 'rm -rf %s.*' % outim
        os.system(cmd1)
        tclean(vis=outms,
               imagename=outim,
               niter=niter,
               gridder='mosaic',
               imsize=imsize,
               cell=cell,
               restoringbeam  = restoringbeam,
               stokes='I',
               pbcor=True,
               phasecenter=phasecenter,
               weighting='natural',
               specmode='cube')
        ng_stats(outim + '.image')
        if do_fits:
            exportfits(outim+'.image',outim+'.fits')

    #-end of ng_vla()

def ng_tpdish(name, size):
    """
    A horrific patch to work with dishes that are not 12m (currently hardcoded in tp2vis.py)

    E.g. for GBT (a 100m dish) you would need to do:

    ng_tpdish('ALMATP',100.0)
    ng_tpdish('VIRTUAL',100.0)
    """
    old_size = t2v_arrays[name]['dish']
    old_fwhm = t2v_arrays[name]['fwhm100']
    r = size/old_size
    t2v_arrays[name]['dish']   = size
    t2v_arrays[name]['fwhm100']= old_fwhm / r
    print "NG_DISH: ",old_size, old_fwhm, ' -> ', size, old_fwhm/r

def ng_tp_otf(project, skymodel, dish, label, freq=None, template=None):
    """
    helper function to create on the fly total power map
    
    dish:       dish diameter in meters
    freq:       frequency in GHz
    template:   dirty image --> must come from tclean so there is both *.image and *.pb
    
    @todo make use of the template for regrid
    @todo come up with a good way to handle the directoy structure for the project input 
    
    E.g. for 45 m single dish configuration:

    ng_tp_otf('test10/clean1', 'skymodel.im', dish=45)
    """
    # clean up old project
    # os.system('rm -rf %s ; mkdir -p %s' % (project,project))

    # projectpath/filename for temporary otf 
    out_tmp   = '%s/temp_otf.image'%project
    # projectpath/filename for otf.image.pbcor
    out_pbcor = '%s/otf%s.image.pbcor'%(project, label)
    # projectpath/filename for otf.image (primary beam applied)
    out_image = '%s/otf%s.image'%(project, label)

    # check if a freq was specificed in the input
    if freq == None:
        # if none, then pull out frequency from skymodel header
        # @todo come up with a way to check if we are actually grabbing the frequency from the header. it's not always crval3
        h0 = imhead(skymodel,mode='list')
        freq = h0['crval4'] # hertz
    else:
        freq = freq * 1.0e9

    # calculate beam size in arcsecs
    # @todo check if alma uses 1.22*lam/D or just 1.0*lam/D
    beam = cms / (freq * dish) * apr

    # convolve skymodel with beam. assumes circular beam
    imsmooth(imagename=skymodel,
             kernel='gauss',
             major='%sarcsec'%beam,
             minor='%sarcsec'%beam,
             pa='0deg',
             outfile=out_tmp,
             overwrite=True)

    # regrid
    if template == None:
        # inherit template from dirty map if template has not be specified in the input
        # @todo need a way to grab the last dirtymap (e.g. dirtymap7.image) or grab a specified dirty map (e.g. dirtymap7 is bad so we want dirtymap6)
        template = '%s/dirtymap.image'%project

    imregrid(imagename=out_tmp,
             template=template,
             output=out_pbcor,
             overwrite=True)

    # @todo modify the template.replace to make sure it's only the last '.image' that is replaced just in case
    immath(imagename=[out_pbcor, '%s/%s'%(project, template.replace('.image', '.pb'))],
           expr='IM0*IM1',
           outfile=out_image)

    # remove the temporary otf image that was created
    os.system('rm -fr %s'%out_tmp)

def ng_tp_vis(project, imagename, ptg=None, imsize=512, pixel=1.0, niter=-1, phasecenter=None, rms=None, maxuv=10.0, nvgrp=4, fix=1, deconv=True, **line):
           
    """
      Simple frontend to call tp2vis() and an optional tclean()
    
    
      _required_keywords:
      ===================
      project:       identifying (directory) name within which all files are places
      imagename:     casa image in RA-DEC-POL-FREQ order
      ptg            Filename with pointings (ptg format) to be used
                     If none specified, it will currently return, but there may be a
                     plan to allow auto-filling the (valid) map with pointings.
    
    
      _optional_keywords:
      ===================
      imsize:        if maps are made, this is mapsize (list of 2 is allowed if you need rectangular)
      pixel:         pixel size, in arcsec
      niter:         -1 if no maps needed, 0 if just fft, no cleaning cycles
    
      phasecenter    Defaults to mapcenter (note special format)
                     e.g. 'J2000 00h48m15.849s -73d05m0.158s'
      rms            if set, this is the TP cube noise to be used to set the weights
      maxuv          maximum uv distance of TP vis distribution (in m)  [10m] 
      nvgrp          Number of visibility group (nvis = 1035*nvgrp)
      fix            Various fixes such that tclean() can handle a list of ms.
                     ** this parameter will disappear or should have default 1
                     0   no fix, you need to run mstransform()/concat() on the tp.ms
                     1   output only the CORRECTED_DATA column, remove other *DATA*
                     2   debug mode, keep all intermediate MS files
                     @todo   there is a flux difference between fix=0 and fix=1 in dirtymap
      deconv         Use the deconvolved map as model for the simulator

      line           Dictionary of tclean() parameters
    """
    # assert input files
    NG.assertf(imagename)
    NG.assertf(ptg)    
    
    # clean up old project
    os.system('rm -rf %s ; mkdir -p %s' % (project,project))

    # report phasecenter in a proper phasecenter format (tp2vis used to do that)
    if True:
        h0=imhead(imagename,mode='list')
        ra  = h0['crval1'] * 180.0 / math.pi
        dec = h0['crval2'] * 180.0 / math.pi
        ra_string  = const.sixty_string(const.hms(ra),hms=True)
        dec_string = const.sixty_string(const.dms(dec),hms=False)
        phasecenter0 = 'J2000 %s %s' % (ra_string, dec_string)
        print "MAP REFERENCE: phasecenter = '%s'" % phasecenter0
        if phasecenter == None:
            phasecenter == phasecenter0

    if ptg == None:
        print "No PTG specified, no auto-regioning yet"
        return

    outfile = '%s/tp.ms' % project
    tp2vis(imagename,outfile,ptg, maxuv=maxuv, rms=rms, nvgrp=nvgrp, deconv=deconv)

    vptable = outfile + '/TP2VISVP'    
    if NG.iscasa(vptable):                   # note: current does not have a Type/SubType
        print "Note: using TP2VISVP, and attempting to use vp from ",vptable
        use_vp = True
        vp.reset()
        vp.loadfromtable(vptable)        # Kumar says this doesn't work, you need the vptable= in tclean()
    else:
        print "Note: did not find TP2VISVP, not using vp"
        use_vp = False
        vptable = None
    vp.summarizevps()

    # remove DATA_* columns to prevent tclean with mslist crash
    # for more stability (some combinations caused tclean() to fail) use concat(copypointing=False)
    # with fix_mode
    #          0 = do nothing (will need do_concat=True)
    #          1 = one fixed tp.ms file
    #          2 = tp.mp, tp1.ms and tp2.ms for experimenting
    fix_mode = fix
    
    if fix_mode == 1:    # should be the default
        print "FIX with mstransform and concat for CORRECTED_DATA" 
        outfile1 = '%s/tp1.ms' % project    
        mstransform(outfile,outfile1)
        os.system('rm -rf %s' % outfile)
        concat(outfile1,outfile)
        os.system('rm -rf %s' % outfile1)

    if fix_mode == 2:
        print "FIX with mstransform and concat and for CORRECTED_DATA keeping backups"
        outfile1 = '%s/tp1.ms' % project    
        outfile2 = '%s/tp2.ms' % project
        outfile3 = '%s/tp3.ms' % project    
        mstransform(outfile,outfile1)
        concat(outfile1,outfile2,copypointing=False)
        concat(outfile1,outfile3)
            
    # Plot UV
    figfile = outfile + ".png"
    print "PLOTUV ",figfile                                                            
    plotuv(outfile,figfile=figfile)

    if niter < 0 or imsize < 0:
        return

    # finalize by making a tclean()

    print "Final test clean around phasecenter = '%s'" % phasecenter
    dirtymap = '%s/dirtymap' % project
    imsize    = NG.imsize2(imsize)
    cell      = ['%garcsec' % pixel]
    weighting = 'natural'

    tclean(vis = outfile,
           imagename      = dirtymap,
           niter          = niter,
           gridder        = 'mosaic',
           imsize         = imsize,
           cell           = cell,
           restoringbeam  = restoringbeam,           
           stokes         = 'I',
           pbcor          = True,
           phasecenter    = phasecenter,
           vptable        = vptable,
           weighting      = weighting,
           specmode       = 'cube',
           **line)
    
    exportfits(dirtymap + ".image", dirtymap + ".fits")

    #-end of ng_tp()


def ng_clean1(project, ms, imsize=512, pixel=0.5, niter=0, weighting="natural", phasecenter="",  **line):
    """
    Simple interface to do a tclean() on one MS
    
    project - new directory for this  (it is removed before starting)
    ms      - a single MS (or a list, but no concat() is done)

    imsize       512  (list of 2 is allowed if you need rectangular)
    pixel        0.5
    niter        0 or more ; @todo   can also be a list, in which case tclean() will be returning results for each niter
    weighting    "natural"
    phasecenter  ""     (e.g. 'J2000 03h28m58.6s +31d17m05.8s')
    **line
    """
    os.system('rm -rf %s; mkdir -p %s' % (project,project))
    #
    outim1 = '%s/dirtymap' % project
    #
    imsize    = NG.imsize2(imsize)
    cell      = ['%garcsec' % pixel]
    # weighting = 'natural'
    # weighting = 'uniform'
    #
    vis1 = ms
    #
    if True:
        try:
            tb.open(ms + '/SPECTRAL_WINDOW')
            chan_freq = tb.getcol('CHAN_FREQ')
            tb.close()
            tb.open(ms + '/SOURCE')
            ref_freq = tb.getcol('REST_FREQUENCY')
            tb.close()
            print 'FREQ:',chan_freq[0][0]/1e9,chan_freq[-1][0]/1e9,ref_freq[0][0]/1e9
        except:
            print "Bypassing some error displaying freq ranges"

    print "VIS1",vis1
    print "niter=",niter
    if type(niter) == type([]):
        niters = niter
    else:
        niters = [niter]

    if type(ms) != type([]):
        vptable = ms + '/TP2VISVP'
        if NG.iscasa(vptable):                   # note: current does not have a Type/SubType
            print "Note: using TP2VISVP, and attempting to use vp from",vptable
            use_vp = True
            vp.reset()
            vp.loadfromtable(vptable)
        else:
            print "Note: did not find TP2VISVP, not using vp"
            use_vp = False
            vptable = None
        vp.summarizevps()
    else:
        use_vp = False        
        vptable = None

    restart = True
    for niter in niters:
        print "TCLEAN(niter=%d)" % niter
        tclean(vis             = vis1,
               imagename       = outim1,
               niter           = niter,
               gridder         = 'mosaic',
               imsize          = imsize,
               cell            = cell,
               restoringbeam   = restoringbeam,           
               stokes          = 'I',
               pbcor           = True,
               phasecenter     = phasecenter,
               vptable         = vptable,
               weighting       = weighting,
               specmode        = 'cube',
               restart         = restart,
               **line)
        restart = False
    
    print "Wrote %s with %s weighting" % (outim1,weighting)

    if len(niters) == 1:
        exportfits(outim1+'.image',outim1+'.fits')
    
    #-end of ng_clean1()
    
def ng_clean(project, tp, ms, imsize=512, pixel=0.5, weighting="natural", phasecenter="", niter=0, do_concat = False, do_cleanup = True, **line):
    """
    Simple interface to do a tclean() joint deconvolution of one TP and one or more MS
    
    project - new directory for this operation (it is removed before starting)
    tp      - the TP MS (needs to be a single MS)
    ms      - the array MS (can be a list of MS)
    imsize  - (square) size of the maps (list of 2 is allowed if you need rectangular)
    pixel   - pixelsize in arcsec
    niter   - 0 or more interactions for tclean

    do_concat   - work around a bug in tclean ?  Default is true until this bug is fixed
    do_alma     - also make a map from just the ms (without tp)
    """
    os.system('rm -rf %s; mkdir -p %s' % (project,project))
    #
    outim2 = '%s/tpalma' % project
    outms  = '%s/tpalma.ms' % project       # concat MS to bypass tclean() bug
    #
    imsize    = NG.imsize2(imsize)
    cell      = ['%garcsec' % pixel]
    # weighting = 'natural'
    # weighting = 'uniform'    
    #
    vis1 = ms
    if type(ms) == type([]):
        vis2 =  ms  + [tp] 
    else:
        vis2 = [ms] + [tp] 
    # @todo    get the weights[0] and print them
    # vis2.reverse()         # for debugging; in 5.0 it seems to be sort of ok,but small diffs can still be seen
    print "niter=",niter
    print "line: ",line
    #
    if type(niter) == type([]):
        niters = niter
    else:
        niters = [niter]

    print "Creating TPALMA using vis2=",vis2
    if do_concat:
        # first report weight 
        print "Weights in ",vis2
        for v in vis2:
            tp2viswt(v)
        # due to a tclean() bug, the vis2 need to be run via concat
        # MS has a pointing table, this often complaints, but in workflow5 it actually crashes concat()
        print "Using concat to bypass tclean bug - also using copypointing=False, freqtol='10kHz'"
        #concat(vis=vis2,concatvis=outms,copypointing=False,freqtol='10kHz')
        concat(vis=vis2,concatvis=outms,copypointing=False)
        vis2 = outms

    restart = True
    for niter in niters:
        print "TCLEAN(niter=%d)" % niter        
        tclean(vis=vis2,
               imagename      = outim2,
               niter          = niter,
               gridder        = 'mosaic',
               imsize         = imsize,
               cell           = cell,
               restoringbeam  = restoringbeam,           
               stokes         = 'I',
               pbcor          = True,
               phasecenter    = phasecenter,
               vptable        = None,
               weighting      = weighting,
               specmode       = 'cube',
               restart        = restart,
               **line)
        restart = False

#          phasecenter=phasecenter,weighting='briggs',robust=-2.0,threshold='0mJy',specmode='cube')

    print "Wrote %s with %s weighting" % (outim2,weighting)

    if len(niters) == 1:
        exportfits(outim2+'.image',outim2+'.fits')

    if do_concat and do_cleanup:
        print "Removing ",outms
        shutil.rmtree(outms)
    
    #-end of ng_clean()

def ng_phasecenter(im):
    """
    return the map reference center as a phasecenter
    """
    NG.assertf(im)
    #
    h0=imhead(im,mode='list')
    ra  = h0['crval1'] * 180.0 / math.pi
    dec = h0['crval2'] * 180.0 / math.pi
    phasecenter = 'J2000 %.6fdeg %.6fdeg' % (ra,dec)
    return  phasecenter
    
def ng_summary(tp, ms=None, source=None, line=False):
    """
    Summarize what could be useful to understand how to combine a TP map with one or more MS files
    and how to call mstransform()

    tp      - one image cube (casa image, or fits file)
    ms      - MS, or a list of MS
    source  - if given, it needs to match this source name in the MS
    """

    def vrange(f,rf):
        nf = len(f)
        if rf > 0:
            v0 = (1-f[0]/rf)*cms/1000.0
            v1 = (1-f[-1]/rf)*cms/1000.0
            dv = (v1-v0)/(nf-1.0)
        else:
            v0 = 0.0
            v1 = 0.0
            dv = 0.0
        return f[0],f[-1],rf,v0,v1,dv,nf

    if type(ms) == type([]):
        ms_list = ms
    elif ms == None:
        ms_list = []
    else:
        ms_list = [ms]

    # ONE IMAGE
    h0=imhead(tp,mode='list')
    ia.open(tp)
    shape = ia.shape()
    h1 = ia.summary()
    iz = h1['axisnames'].tolist().index('Frequency')     # axis # for freq
    ia.close()
    #
    restfreq = h0['restfreq']    
    ra  = h0['crval1'] * 180.0 / math.pi
    dec = h0['crval2'] * 180.0 / math.pi
    phasecenterd = 'J2000 %.6fdeg %.6fdeg' % (ra,dec)
    ra_string  = const.sixty_string(const.hms(ra),hms=True)
    dec_string = const.sixty_string(const.dms(dec),hms=False)
    phasecenter = 'J2000 %s %s' % (ra_string, dec_string)
    nx = h0['shape'][0]
    ny = h0['shape'][1]
    nz = h0['shape'][iz]
    dx = np.abs(h0['cdelt1']) # in radians!
    dy = np.abs(h0['cdelt2'])
    du = 1.0/(nx*dx)
    dv = 1.0/(ny*dy)
    # freq_values = h0['crval4'] + (np.arange(nz) - h0['crpix4']) * h0['cdelt4']
    # freq_values.reshape(1,1,1,nz)
    freq_values = h1['refval'][iz] + (np.arange(nz) - h1['refpix'][iz]) * h1['incr'][iz]
    vmin = (1-freq_values[0]/restfreq)*cms/1000.0
    vmax = (1-freq_values[-1]/restfreq)*cms/1000.0
    dv   = (vmax[0]-vmin[0])/(nz-1)
    rft = h0['reffreqtype']

    if line:
        _line = {}
        _line['restfreq'] = '%sGHz'  % repr(restfreq[0]/1e9)
        _line['nchan']    =  nz
        _line['start']    = '%skm/s' % repr(vmin[0])
        _line['width']    = '%skm/s' % repr(dv)
        return _line

    # print the image info
    print "NG_SUMMARY:"
    print "TP:",tp
    print 'OBJECT:  ',h0['object']
    print 'SHAPE:   ',h0['shape']
    print 'CRVAL:   ',phasecenter
    print 'CRVALd:  ',phasecenterd
    print 'PIXEL:   ',dx*apr
    print 'RESTFREQ:',restfreq[0]/1e9
    print "FREQ:    ",freq_values[0]/1e9,freq_values[-1]/1e9
    print "VEL:     ",vmin[0],vmax[0],dv
    print "VELTYPE: ",rft
    print "UNITS:   ",h0['bunit']
    
    # LIST OF MS (can be empty)
    for msi in ms_list:
        print ""
        if NG.iscasa(msi):
            print "MS: ",msi
        else:
            print "MS:   -- skipping non-existent ",msi
            continue

        # first get the rest_freq per source (it may be missing)
        tb.open(msi + '/SOURCE')
        source  = tb.getcol('NAME')
        nsource = len(source)
        try:
            rest_freq = tb.getcol('REST_FREQUENCY')/1e9
        except:
            rest_freq = np.array([[0.0]])
        spw_id    = tb.getcol('SPECTRAL_WINDOW_ID')
        tb.close()
        # print "rest_freq",rest_freq.shape,rest_freq
        
        # special treatment for spw, since each spw can have a different # channels (CHAN_FREQ)
        tb.open(msi + '/SPECTRAL_WINDOW')
        ref_freq = tb.getcol('REF_FREQUENCY')/1e9
        # print "ref_freq",ref_freq.shape,ref_freq

        chan_freq = []
        nspw = len(ref_freq)
        for i in range(nspw):
            chan_freq_i = tb.getcell('CHAN_FREQ',i)/1e9
            # print "spw",i,vrange(chan_freq_i,ref_freq[i])
            chan_freq.append(chan_freq_i)
        tb.close()
        #
        for i in range(nsource):
            # print "source",i,source[i],spw_id[i],rest_freq[0][i]
            # print "source",i,source[i],trans[0][i],vrange(chan_freq[spw_id[i]],rest_freq[0][i])
            print "source",i,source[i],vrange(chan_freq[spw_id[i]],rest_freq[0][i])            

        
        #print "chan_freq",chan_freq.shape,chan_freq
        # print 'FREQ:',chan_freq[0][0]/1e9,chan_freq[-1][0]/1e9,ref_freq[0][0]/1e9

    #-end of ng_summary()

def ng_mom(imcube, chan_rms, pb=None, pbcut=0.3):
    """
    Take mom0 and mom1 of an image cube, in the style of the M100 casaguide.
    
    imcube:      image cube (flux flat, i.e. the .image file)
    chan_rms:    list of 4 integers, which denote the low and high channel range where RMS should be measured
    pb:          primary beam. If given, it can do a final pb corrected version and use it for masking
    pbcut:       if PB is used, this is the cutoff above which mask is used
    
    """
    def lel(name):
        """ convert filename to a safe filename for LEL expressions, e.g. in mask=
        """
        return '\'' + name + '\''
    chans1='%d~%d' % (chan_rms[0],chan_rms[1])
    chans2='%d~%d' % (chan_rms[2],chan_rms[3])
    chans3='%d~%d' % (chan_rms[1]+1,chan_rms[2])
    rms  = imstat(imcube,axes=[0,1])['rms']
    print rms
    rms1 = imstat(imcube,axes=[0,1],chans=chans1)['rms'].mean()
    rms2 = imstat(imcube,axes=[0,1],chans=chans2)['rms'].mean()
    print rms1,rms2
    rms = 0.5*(rms1+rms2)
    print "RMS = ",rms
    if pb==None:
        mask = None
    else:
        mask = lel(pb) + '> %g' % pbcut
        print "Using mask=",mask
    mom0 = imcube + '.mom0'
    mom1 = imcube + '.mom1'
    os.system('rm -rf %s %s' % (mom0,mom1))
    immoments(imcube, 0, chans=chans3, includepix=[rms*2.0,9999], mask=mask, outfile=mom0)
    immoments(imcube, 1, chans=chans3, includepix=[rms*5.5,9999], mask=mask, outfile=mom1)

def ng_flux(image, box=None, dv = 1.0, plot='plot5.png'):
    """ Plotting min,max,rms as function of channel
    
        box     xmin,ymin,xmax,ymax       defaults to whole area

        A useful way to check the the mean RMS at the first
        or last 10 channels is:

        imstat(image,axes=[0,1])['rms'][:10].mean()
        imstat(image,axes=[0,1])['rms'][-10:].mean()
    
    """
    plt.figure()
    _tmp = imstat(image,axes=[0,1],box=box)
    fmin = _tmp['min']
    fmax = _tmp['max']
    frms = _tmp['rms']
    chan = np.arange(len(fmin))
    f = 0.5 * (fmax - fmin) / frms
    plt.plot(chan,fmin,c='r',label='min')
    plt.plot(chan,fmax,c='g',label='max')
    plt.plot(chan,frms,c='b',label='rms')
    # plt.plot(chan,f,   c='black', label='<peak>/rms')
    zero = 0.0 * frms
    plt.plot(chan,zero,c='black')
    plt.ylabel('Flux')
    plt.xlabel('Channel')
    plt.title('%s  Min/Max/RMS' % (image))
    plt.legend()
    plt.savefig(plot)
    plt.show()
    print "Sum: %g Jy km/s (%g km/s)" % (fmax.sum() * dv, dv)


def ng_combine(project, TPdata, INTdata, **kwargs):
    """
    Wishful Function to combine total power and interferometry data.

    The current implementation requires you to use the same gridding for likewise axes.

    project : project directory within which all work will be done. See below for a
              description of names of datasets.

    TPdata :  input one (or list) of datasets representing TP data.
              These will be (CASA or FITS) images.

    INTdata : input one (or list) of datasets represenring interferometry data.
              These can either be (FITS or CASA) images, or measurement sets (but no mix).
              Depending on which type, different methods can be exposed.

    mode :    non-zero if you want to try to enforce mode of combining data.   0 = automated.
    

    **kwargs : python dictionary of {key:value} pairs which depends on the method choosen.

    If INTdata is an image, the following modes are available:
        11. CASA's feather() tool will be used. [default for mode=0]
        12. Faridani's SSC will be used.
    If INTdata is a measurement set, imaging parameters for tclean() will be needed.
        21. TP2VIS will be used. [default for mode=0]
        22. SD2VIS will be used.



    """

    print "you just wished this would work...."

    if False:
        os.system('rm -rf %s; mkdir -p %s' % (project,project))    
    
    if type(TPdata) == type([]):
        _TP_data = TPdata
    else:
        _TP_data = [TPdata]        

    if type(INTdata) == type([]):
        _INT_data = INTdata
    else:
        _INT_data = [INTdata]        
        
    

class NG(object):
    """ Static class to hide some local helper functions

        rmcasa
        iscasa
        casa2np
        imsize2
        assertf
    
    """
    @staticmethod
    def rmcasa(filename):
        if NG.iscasa(filename):
            os.system('rm -rf %s' % filename)
        else:
            print "Warning: %s is not a CASA dataset" % filename

    @staticmethod
    def iscasa(filename, casatype=None):
        """is a file a casa image
        
        casatype not implemented yet
        (why) isn't there a CASA function for this?
        
        Returns
        -------
        boolean
        """
        isdir = os.path.isdir(filename)
        if casatype == None:
            return isdir
        if not isdir:
            return False
        # ms + '/table.info' is an ascii file , first line should be
        # Type = Image
        # Type = Measurement Set

        # @todo for now
        return False

    @staticmethod    
    def casa2np(image, z=None):
        """
        convert a casa[x][y] to a numpy[y][x] such that fits writers
        will produce a fits file that looks like an RA-DEC image
        and also native matplotlib routines, such that imshow(origin='lower')
        will give the correct orientation.

        z      which plane to pick in case it's a cube (not implemented)
        """
        return np.flipud(np.rot90(image))

    @staticmethod    
    def imsize2(imsize):
        """ if scalar, convert to list, else just return the list
        """
        if type(imsize) == type([]):
            return imsize
        return [imsize,imsize]

    @staticmethod
    def iarray(array):
        """
        """
        return map(int,array.split(','))

    @staticmethod
    def farray(array):
        """
        """
        return map(float,array.split(','))
    
    @staticmethod
    def assertf(filename = None):
        """ ensure a file or directory exists, else report and and fail
        """
        if filename == None: return
        assert os.path.exists(filename),  "NG.assertf: %s does not exist" % filename
        return
        
    
#- end of ngvla.py
