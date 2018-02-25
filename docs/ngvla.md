# ngVLA simulations

The ng_ routines help you writing short python scripts that
orchestrate ngVLA simulations. In general you would do all the work
within a project directory, similar to CASA's **simobserve()**,
where the names of files inside the
directory have a fixed meaning, and only the directory name (the 'project')
has a meaning for the observer.

For example, In our test/ directory they are just
called 'test1', 'test1-alma', 'test1-SSA', 'test2', etc.

## Summary

A typical simulation script might look as follows. Explanations follow next:

    ng_begin("test123")
    ng_version()
    ng_log("
    ng_ptr(phasecenter,"test123.ptg")
    ng_vla("test123","skymodel.fits", 4096, 0.01, ptg="test123.ptg",phasecenter=phasecenter)
    ng_clean1("test123/clean1",phasecenter=phasecenter)
    ng_end()

## Logging

Standard CASA logging is using , and you can use the python **print** function
in the usual way.  You can use the function

    ng_log(msg)

to space your output with a header type message.

## Performance

To measure performance, you can of course use the unix "time" prefix to your command,
e.g.

    time casa --nogui -c  sim1.py  a=1 b=10 c=100

but this only gives the final compute times of the script, nothing internal.

Every ng_ command will tag the output with a timestamp, and optionally measure
virtual and real memory. This is added to the standard python **logging** output.
But you must start and end your code as follows:

    ng_begin("test1-alma")
    nv_version()
    ....
    ng_end()

and the output will contain things like


    INFO:root:test1-alma BEGIN [ 0.  0.]
    INFO:root:test1-alma ptg  [  3.44000000e-04   3.42130661e-04   1.13960547e+03   1.90476562e+02]
    INFO:root:test1-alma alma  [  4.00000000e-04   3.99827957e-04   1.13960547e+03   1.90476562e+02]
    INFO:root:test1-alma alma  [  143.253957     331.35863519  6093.55859375  4505.33203125]
    INFO:root:test1-alma alma  [   39.554776      39.0021348   4701.64453125  3106.890625  ]
    INFO:root:test1-alma alma  [   35.794047      35.00777411  4765.64453125  3108.046875  ]
    INFO:root:test1-alma alma  [   41.075288      39.60072398  4765.64453125  3108.12890625]
    INFO:root:test1-alma alma  [   34.188305      32.5692451   4765.64453125  3108.12890625]
    INFO:root:test1-alma clean1  [   34.755774      34.09187388  4765.64453125  3108.12890625]
    INFO:root:test1-alma clean1  [ 1000.589456     309.20978689  4765.66796875  3110.734375  ]
    INFO:root:test1-alma tpdish  [ 1026.255616     315.74331117  4765.66796875  3110.734375  ]
    INFO:root:test1-alma tpdish  [   23.116537      22.50622487  4765.66796875  3111.6015625 ]
    INFO:root:test1-alma feather  [   23.429309      22.71242213  4765.66796875  3111.6015625 ]
    INFO:root:test1-alma stats  [  4.65160000e-01   1.48370099e+00   4.76566797e+03   3.11174219e+03]
    ....
    INFO:root:test1-alma smooth  [  1.16539000e-01   6.64260387e-02   4.76566797e+03   3.11183594e+03]
    INFO:root:test1-alma stats  [    8.981304       6.08805799  4765.66796875  3111.8359375 ]
    INFO:root:test1-alma done  [  2.18567800e+00   6.15752935e-01   4.76566797e+03   3.11183594e+03]
    INFO:root:test1-alma END [ 2483.496519    1244.87865996]

whereas in this case the **time**  command-prefix would have given just the END result:


    2418.71user 68.66system 20:48.26elapsed 199%CPU (0avgtext+0avgdata 7029656maxresident)k
    18162112inputs+36370600outputs (130major+11010875minor)pagefaults 0swaps

Some interestsing observations is the CPU (first number) and WALL (second number) clock, and many
routines will have the WALL less than the CPU time, indication that a good amount of parallelism
was achieved. But not always, and the above listing clearly shows some puzzling behavior why
some do have a good ratio, others are near 1.

## Simulation routines

As mentioned before,the project directory is within which all the work occurs. Some routines will accumulate
(e.g. ng_vla()), others will remove that project directory and rebuild that directory (e.g. ng_clean1). The
user will remember to orchestrate them inside each where needed, e.g.

    ng_vla("test1", cfg=0, ...
    ng_clean1("test1/clean0", ...
    
    ng_vla("test1", cfg=1, ...
    ng_clean1("test1/clean1", ...

    mslist = glob.glob("test1/*.ms")
    ng_clean1("test1/clean2", mslist, ...)


### ng_vla(project, skymodel, imsize, pixel, phasecenter, freq, cfg, niter, ptg)

This is usually how you start a simulation, from a skymodel you create a measurement set reprenting a configuration.
Since the ngVLA can have multiple configurations, you would need call this routine multiple times, e.g.

    ng_vla("test1", cfg=0, ...)
    ng_vla("test2", cfg=0, ...)

Setting different  weights based on dish sizes will need to be implemented. See also ng_alma().

### ng_alma(project, skymodel, imsize, pixel, phasecenter, freq, cycle, cfg, niter, ptg)

Just for kicks, we have way to create ALMA observations from diffent cycle's and cfg's. We automatically add
a visweightscale to the WEIGHT column to properly account for the dish size if you do a combination tclean.

### ng_clean1(project, ms, imsize, pixel, niter, weighting, startmodel, phasecenter, **line)

This is simply a front end to CASA's **tclean()**. Interesting note is that **niter** can be a list here,
e.g. niter=[0,1000,2000], thus creating a series of dirtymaps.

### ng_clean()

Unclear if we keep this routine. Don't use for now.

### ng_tp_otf(project, skymodel, dish, label, freq, template)

Create the OTF map via a simple smooth.

NOTE: **simobserve** also has an **obsmode=sd** keyword, which creates
a Measurement Set representing the autocorrelations of a TP (sd) map. After this you would need to map
this MS into a casa image, using the  XXX task.

### ng_tp_vis(project, imagename, ptg, imsize, pixel, niter, phasecenter, rms, maxuv, nvgrp, fix, deconv, **line)

TP2VIS method. Not covering here yet.


### ng_feather(project, highres, lowres, label, niteridx)

Feather two maps that were created from ng_clean1 and ng_tp_otf

### ng_smooth(project, skymodel, label, niteridx)

Smooth a skymodel so it can be compared to a (feathered for now) map.

### ng_combine

Futuristic routine that automatically detects (via the argumnent list) what the intent is,
and then combines MS/IM things into a map.

## Helper routines

### ng_stats

### ng_beam

### ng_tpdish

### ng_phasecenter

### ng_ptg

We should re-import the ng_im_ptg routine from qtp. See below.

### qtp_im_ptg(phasecenter, imsize, pixel, grid, im=[], rect=False, outfile=None)

this should possibly become an ng_im_ptg 

### ng_summary

### ng_math

Just a super simple front-end to immath, so we can wrote code such as

    ng_math(out, in1, '+', in2)

it only allows the four basic operators combining two maps. Partially driven because immath
does not have the overwrite=True option.

### ng_mom

Compute moments 0,1 (and soon 2) in some standard way, so we can compare simulations and skymodels.

### ng_flux(image, box, dv, plot)

Create a plot showing flux as function of channel. Good to compare flux comparisons
between various simulations and skymodels.
