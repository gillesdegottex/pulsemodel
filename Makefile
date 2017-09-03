.PHONY: build build_sigproc build_pyworld


all: sigproc/sinusoidal.pyx build


build: build_sigproc build_pyworld


sigproc/sinusoidal.pyx:
	git submodule update --init

build_sigproc: sigproc/sinusoidal.pyx
	cd sigproc; $(MAKE)


external/pyworld/lib/World:
	git submodule update --init

build_pyworld: external/pyworld/lib/World
	cd external/pyworld; python setup.py build_ext --inplace

	
test: build
	cd test; $(MAKE)

distclean:
	rm -f *.pyc
	cd test; $(MAKE) distclean
	cd sigproc; $(MAKE) distclean
