# examples how to make models

## tp2vis

We have a skymodel.fits, derived from procedures in tp2vis. It is beyond the scope of this project
to describe this here.

## NEMO

Quick example how to install NEMO

     wget https://teuben.github.io/nemo/install_nemo
     chmod +x install_nemo
     ./install_nemo nemo=$HOME/nemo
     source $HOME/nemo/nemo_start.sh

(or .csh if you shell is a csh flavor instead)

after which you can run a default model and view it, for example in ds9


     ./mkgalcube
     ds9 model1.fits

other examples:

      ./mkcalcube run=model4 beam=0.5 noise=0.0001 nmode=1

which does beam smearing , and adds noise after (nmode=1) the beamsmearing. This gives you uncorrelated noise
in the final cube. If you added 10 times this noise becore the beamsmearing,

      ./mkcalcube run=model3 beam=0.5 noise=0.001

the final noise is now spatially correlated and much less than 0.001, and more like 0.00017

For the ngVLA modeling we will however inject noise using CASA's simobserve, not in the skymodel.
