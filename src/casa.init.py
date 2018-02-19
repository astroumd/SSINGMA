#
#  this is a piece of code you can place in ~/.casa/init.py
#        git clone ...
#  to create your ngvla_root in any location

import os

try:
    ngvla_root = os.environ['HOME'] + '/.casa/SSINGMA'      # SET THIS TO YOUR LOCATION OF SSINGMA or use a symlink
    py_files   = ['src/ngvla']                              # Pick the ones you want
    #
    work_dir = os.getcwd()
    sys.path.append(ngvla_root)
    os.chdir(ngvla_root)
    for py in py_files:
        pfile = py + '.py'
        print "NGVLA: Loading ",pfile
        execfile(pfile)
    os.chdir(work_dir)
    print "NGVLA: ",ng_version()
except:
    print "ngvla not properly loaded, back to",work_dir
    os.chdir(work_dir)
