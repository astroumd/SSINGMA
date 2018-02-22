#
#

TESTS = test1 test2

help:	
	@echo no help

.PHONY:   $(TESTS)

all:	$(TESTS)

test1:
	rm -rf test1
	time casa --nogui -c test1.py > test1.log 2>&1
test2:
	rm -rf test2
	time casa --nogui -c test2.py > test2.log 2>&1
