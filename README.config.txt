
===================================================================
 9  Dec. 2017:   ngVLA Configuration Material 
===================================================================

Update from 5 Sept. 2017:
       - Corrected error with shortest baseline being smaller than a dish diameter.  

Summary:
--------------------------------
We include a finite set of notional configurations for the ngVLA that span a limited range of phase space.  This information is provided as a simple resource to help the community carry out imaging simulations/investigations that critically explore the science capabilities of the proposed array.  We expect that such investigations will lead to suggested modifications to these notional configurations for specific science cases, and hope the community will report on such findings via ngVLA memos and/or refereed publications.  



--------------------------------
Included in this tarball:
--------------------------------
Configuration files (described in more detail below):
	      - SW214.cfg
	      - SWcore.cfg
	      - SWVLB.cfg

Supporting Documents:
	   - CONFIG.CASA.pdf: Instructions on how to use configuration files to generate mock observations using CASA.  

--------------------------------
Brief Description of Configuration Files:
--------------------------------

SW214.cfg
--------------------------------
The baseline configuration for the ngVLA as it stands in Sept, 2017, based on extensive community input and design studies. The array has a total of 214 antennas of 18m diameter.  Of these, 114 are in a 'core' of about 2km diameter, centered on the VLA site. Then there are 54 antennas extending to current VLA A array baselines (30km), oriented along the current rail lines (this may be modified in the future, if needed). The remaining antennas extend to about 600km, mostly to the South and East, into Texas and Mexico. This configuration has incorporated practical constraints, such as power, roads, access, and in some cases, fiber. The performance has been shown to be reasonable for imaging on various different scales (see the ngVLA memo series). 

SWcore.cfg
--------------------------------
This condfiguration is just the 114 core stations from the SW214 array above. For low resolution simulations, there are practical reasons not to include all the long baselines in the CASA simulator.

SWVLB.cfg
--------------------------------
Includes the SW214 configuration, plus 9 more antennas on longer scales, out to 2500km baseline lengths. 

