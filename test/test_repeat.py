import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'external/pyworld/pyworld'))

import unittest

import numpy as np

class TestBase(unittest.TestCase):
    def test_base(self):

        filenames = ['slt_arctic_a0010.wav', 'bdl_arctic_a0020.wav', 'clb_arctic_a0030.wav', 'awb_arctic_a0040.wav']
        f0_min = 60
        f0_max = 600

        import pulsemodel
        import pyworld
        import sigproc as sp


        for fname in filenames:
            fname = 'test/'+fname
            lf0s_ref = None
            pwf0_ref = None
            SPEC_ref = None
            fwspec_ref = None
            fwnm_ref = None
            for _ in xrange(2):
                print('Extracting features for: '+fname)
                pulsemodel.analysisf(fname, f0_min=f0_min, f0_max=f0_max, f0_file=fname.replace('.wav','.lf0'), f0_log=True,
                spec_file=fname.replace('.wav','.fwspec'), spec_nbfwbnds=65, nm_file=fname.replace('.wav','.fwnm'), nm_nbfwbnds=33, verbose=1)


                lf0s = np.fromfile(fname.replace('.wav','.lf0'), dtype=np.float32)
                lf0s = lf0s.reshape((-1, 1))
                print('lf0: '+str(np.sum((lf0s)**2)))

                if lf0s_ref is None:
                    lf0s_ref = lf0s
                else:
                    print('lf0 diff: '+str(np.sum((lf0s_ref-lf0s)**2)))


                #_f0, ts = pyworld.dio(x, fs, frame_period=shift*1000)    # raw pitch extractor # Use REAPER instead
                wav, fs, enc = sp.wavread(fname)

                pwts = 0.005*np.arange(len(lf0s)) # TODO TODO TODO
                dftlen = 4096
                # from IPython.core.debugger import  Pdb; Pdb().set_trace()
                dlf0s = lf0s.astype(np.float64)
                pwf0 = pyworld.stonemask(wav, np.ascontiguousarray(np.exp(dlf0s[:,0])), pwts, fs)  # pitch refinement
                if pwf0_ref is None:
                    pwf0_ref = pwf0
                else:
                    print('pwf0 diff: '+str(np.sum((pwf0_ref-pwf0)**2)))


                SPEC = pyworld.cheaptrick(wav, pwf0, pwts, fs, fft_size=dftlen)  # extract smoothed spectrogram
                if SPEC_ref is None:
                    SPEC_ref = SPEC
                else:
                    print('SPEC diff: '+str(np.sum((SPEC_ref-SPEC)**2)))


                fwspec = np.fromfile(fname.replace('.wav','.fwspec'), dtype=np.float32)
                fwspec = fwspec.reshape((-1, 65))
                print('fwspec: '+str(np.sum((fwspec)**2)))

                if fwspec_ref is None:
                    fwspec_ref = fwspec
                else:
                    print('fwspec diff: '+str(np.sum((fwspec_ref-fwspec)**2)))


                fwnm = np.fromfile(fname.replace('.wav','.fwnm'), dtype=np.float32)
                fwnm = fwnm.reshape((-1, 33))
                print('fwnm: '+str(np.sum((fwnm)**2)))

                if fwnm_ref is None:
                    fwnm_ref = fwnm
                else:
                    print('fwnm diff: '+str(np.sum((fwnm_ref-fwnm)**2)))

if __name__ == '__main__':
    unittest.main()
