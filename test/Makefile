#
#

TESTS = test1 test2 test1-alma test1-SSA

help:	
	@echo no help

.PHONY:   $(TESTS)

all:	$(TESTS)

test1:
	rm -rf test1
	time casa --nogui -c test1.py > test1.log 2>&1

test1-alma:
	rm -rf test1-alma
	time casa --nogui -c test1-alma.py > test1-alma.log 2>&1

test1-SSA:
	rm -rf test1-SSA
	time casa --nogui -c test1-SSA.py > test1-SSA.log 2>&1

test2:
	rm -rf test2
	time casa --nogui -c test2.py > test2.log 2>&1
