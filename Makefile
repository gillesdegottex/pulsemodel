all:
	cd sigproc; $(MAKE)

test:
	cd test; $(MAKE)
	
distclean:
	rm -f *.pyc
	cd test; $(MAKE) distclean
