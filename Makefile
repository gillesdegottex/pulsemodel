.PHONY: build submodule_init build_sigproc build_reaper build_pyworld


all: build

submodule_init:
	git submodule update --init --recursive


build: submodule_init build_sigproc build_reaper build_pyworld

build_sigproc: submodule_init
	cd sigproc; $(MAKE)

build_reaper: submodule_init
	cd external/REAPER; mkdir build; cd build; cmake ..; make

build_pyworld: submodule_init
	cd external/pyworld; python setup.py build_ext --inplace


test: build
	cd test; $(MAKE)
	python test/test_smoke.py

distclean:
	rm -f *.pyc
	cd test; $(MAKE) distclean
	cd sigproc; $(MAKE) distclean
