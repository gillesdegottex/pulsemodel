## Pulse model analysis and synthesis

It is basically the vocoder described in:
> G. Degottex, P. Lanchantin, and M. Gales, "A Pulse Model in Log-domain
>    for a Uniform Synthesizer," in Proc. 9th Speech Synthesis Workshop
>    (SSW9), 2016.

### Documentation
Please see the headers of analysis.py and synthesis.py files as well as the
functions documentation for more details.

### Testing/HowTo
In the root directory, simply run:
```make
$ make test
```

You can also have a look at the file test/Makefile to see how the PML's scripts can be used.

### Legal

Copyright(C) 2016 Engineering Department, University of Cambridge, UK.

The code in this repository is released under the Apache License, Version 2.0.
Please see LICENSE.md for more details.

Author: Gilles Degottex <gad27@cam.ac.uk>

### External tools
PML first aims at extracting a noise measure and synthesis a waveform assuming F0 curve and amplitude spectral envelopes are already given.

In order to make it a standalone vocoder, it was thus necessary to import an F0 estimator and a spectral envelope estimator.

#### For F0
For F0, a modified version of REAPER is used:
> https://github.com/gillesdegottex/REAPER

However, the modifications made to the original REAPER do NOT change the f0 estimation.
They are only made for an easier interface with PML code.
You can have a look at the modifications here:
> https://github.com/google/REAPER/compare/master...gillesdegottex:master

#### For the amplitude spectral envelope
For the amplitude spectral envelope, the estimator CheapTrick is used:

> Masanori Morise, CheapTrick, a spectral envelope estimator for high-quality speech synthesis, Speech Communication, Volume 67, 2015, Pages 1-7, ISSN 0167-6393, http://dx.doi.org/10.1016/j.specom.2014.09.003.

The python wrapper of the original implementation is used (without any modification) and can be found at:
> https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder

Note that all the published results about PML have been done using the spectral envelope of the STRAIGHT vocoder, NOT using WORLD.
Because of legal reason it is not possible to release any of STRAIGHT vocoder analysis. Thus, the use of CheapTrick instead in this repository.
It also means that, even though STRAIGHT's envelope and CheapTrick are quite similar, you might observe small differences in results between the two.
