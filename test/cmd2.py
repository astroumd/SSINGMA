#
#  example command line parser usage - functional approach with keyword database
#
#  casa -c cmd2.py a=100 'c=[100,200]'

script_keywords = {
    'a'        :   1,
    'b'        :  2.0,
    'c'        :  [1,2,3],
}

import sys
ng_initkeys(script_keywords,sys.argv)

a = ng_getkey('a')
b = ng_getkey('b')
c = ng_getkey('c')

print 'a=',a
print 'b=',b
print 'c=',c

