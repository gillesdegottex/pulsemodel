.PHONY: build


all: build
	

build:
	cd sigproc; $(MAKE)
	
test: build
	cd test; $(MAKE)
	
distclean:
	rm -f *.pyc
	cd test; $(MAKE) distclean
	cd sigproc; $(MAKE) distclean
