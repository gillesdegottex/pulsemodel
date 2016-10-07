.PHONY: build


all: sigproc/sinusoidal.pyx build
	

sigproc/sinusoidal.pyx:
	git submodule update --init
	
build: sigproc/sinusoidal.pyx
	cd sigproc; $(MAKE)

test: build
	cd test; $(MAKE)
	
distclean:
	rm -f *.pyc
	cd test; $(MAKE) distclean
	cd sigproc; $(MAKE) distclean
