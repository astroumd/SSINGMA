#
#  example command line parser usage - python cut and pastable
#
#  casa -c cmd2.py a=100 'c=[100,200]'

a = 1
b = 2.0
c = [1,2,3]

import sys
for arg in ng_argv(sys.argv):
    exec(arg)

print 'a=',a
print 'b=',b
print 'c=',c

