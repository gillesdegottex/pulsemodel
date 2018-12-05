#!/usr/bin/env python
'''

References
    [1] G. Degottex, P. Lanchantin, and M. Gales, "A Pulse Model in Log-domain
        for a Uniform Synthesizer," in Proc. 9th Speech Synthesis Workshop
        (SSW9), 2016.
    [2] G. Degottex and D. Erro, "A uniform phase representation for the
        harmonic model in speech synthesis applications," EURASIP, Journal on
        Audio, Speech, and Music Processing - Special Issue: Models of Speech -
        In Search of Better Representations, vol. 2014, iss. 1, p. 38, 2014.
    [3] G. Degottex, P. Lanchantin and M. Gales, "A Log Domain Pulse Model for
        Parametric Speech Synthesis", IEEE Transactions on Audio, Speech, and
        Language Processing, 26(1):57-70, 2018.

Copyright(C) 2016 Engineering Department, University of Cambridge, UK.

License
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Author
    Gilles Degottex <gad27@cam.ac.uk>
'''

import argparse
import sys
import os
import warnings

import numpy as np
np.random.seed(123) # Generate always the same "random" numbers, for debugging.
from scipy import signal as sig

import sigproc as sp
import sigproc.pystraight
import sigproc.interfaces

# Add the path for REAPER f0 estimator
os.environ["PATH"] += os.pathsep + os.path.join(os.path.split(os.path.realpath(__file__))[0],'external/REAPER/build')
# Add the path for WORLD vocoder's amplitude spectral envelope estimator
sys.path.insert(0, os.path.join(os.path.split(os.path.realpath(__file__))[0],'external/pyworld/pyworld'))

def analysis_f0postproc(wav, fs, f0s=None, f0_min=60, f0_max=600,
             shift=0.005,        # Usually 5ms
             f0estimator='REAPER',
             verbose=1):
    '''
    Post process the F0 estimate.
    If f0s==None, an F0 estimate is extracted using REAPER.
    '''
    if f0s is None:
        # TODO Switch f0 estimator using `f0estimator`
        f0s = sigproc.interfaces.reaper(wav, fs, shift, f0_min, f0_max)

    # If only values are given, make two column matrix [time[s], value[Hz]] (ljuvela)
    if len(f0s.shape)==1:
        ts = (shift)*np.arange(len(f0s))
        f0s = np.vstack((ts, f0s)).T

    if not (f0s[:,1]>0).any():
        warnings.warn('''\n\nWARNING: No F0 value can be estimated in this signal.
         It will be replaced by the constant f0_min value ({}Hz).
        '''.format(f0_min), RuntimeWarning)
        f0s[:,1] = f0_min


    # Build the continuous f0
    f0s[:,1] = np.interp(f0s[:,0], f0s[f0s[:,1]>0,0], f0s[f0s[:,1]>0,1])
     # Avoid erratic values outside of the given interval
    f0s[:,1] = np.clip(f0s[:,1], f0_min, f0_max)
    # Removes steps in the f0 curve (see sigproc.resampling.f0s_rmsteps(.) )
    f0s = sp.f0s_rmsteps(f0s)
    # Resample the given f0 to regular intervals
    if np.std(np.diff(f0s[:,0]))>2*np.finfo(f0s[0,0]).resolution:
        warnings.warn('''\n\nWARNING: F0 curve seems to be sampled non-uniformly (mean(F0)={}, std(F0s')={}).
         It will be resampled at {}s intervals.
        '''.format(np.std(f0s[:,0]), np.std(np.diff(f0s[:,0])), shift), RuntimeWarning)
        f0s = sp.f0s_resample_cst(f0s, shift)

    return f0s

def analysis_spec(wav, fs, f0s,
             shift=0.005,    # Usually 5ms
             dftlen=4096,    # You can adapt this one according to your pipeline
             verbose=1):
    '''
    Estimate the amplitude spectral envelope.
    '''

    if sp.pystraight.isanalysiseavailable():   # pragma: no cover
                                               # Cannot be tested since STRAIGHT
                                               # is not openly available.
        warnings.warn('''\n\nWARNING: straight_mcep is available,
            STRAIGHT vocoder will thus be used instead of WORLD.
            Note that PML-related publications present results using STRAIGHT vocoder.
        ''', RuntimeWarning)

        # Use STRAIGHT's envelope if available (as in PML's publications)
        SPEC = sigproc.pystraight.analysis_spec(wav, fs, f0s, shift, dftlen, keeplen=True)

    elif sigproc.interfaces.worldvocoder_is_available():

        # Then try WORLD vocoder
        import pyworld
        wav = np.ascontiguousarray(wav)
        #_f0, ts = pyworld.dio(x, fs, frame_period=shift*1000)    # raw pitch extractor # Use REAPER instead
        pwts = np.ascontiguousarray(f0s[:,0])
        pwf0 = pyworld.stonemask(wav, np.ascontiguousarray(f0s[:,1]), pwts, fs)  # pitch refinement
        SPEC = pyworld.cheaptrick(wav, pwf0, pwts, fs, fft_size=dftlen)  # extract smoothed spectrogram
        SPEC = 10.0*np.sqrt(SPEC) # TODO Best gain correction I could find. Hard to find the good one between PML and WORLD different syntheses

    else:   # pragma: no cover
        # This a safeguard that should never happend since WORLD is embeded in
        # pulsemodel.
        # Estimate the sinusoidal parameters at regular intervals in order
        # to build the amplitude spectral envelope
        sinsreg, _ = sp.sinusoidal.estimate_sinusoidal_params(wav, fs, f0s, nbper=3, quadraticfit=True, verbose=verbose-1)

        warnings.warn('''\n\nWARNING: Neither straight_mcep nor WORLD's cheaptrick spectral envelope estimators are available.
         Thus, a SIMPLISTIC Linear interpolation will be used for the spectral envelope.
         Do _NOT_ use this envelope for speech synthesis!
         Please use a better one (e.g. STRAIGHT's or WORLD's).
         If you use this simplistic envelope, the TTS quality will
         be lower than that in the results reported.
        ''', RuntimeWarning)

        SPEC = sp.multi_linear(sinsreg, fs, dftlen)
        SPEC = np.exp(SPEC)*np.sqrt(float(dftlen))

    return SPEC

def analysis_pdd(wav, fs, f0s,
             dftlen=4096,    # You can adapt this one according to your pipeline
             pdd_sin_nbperperiod=4, # 4 analysis instants per period [2]
             pdd_sin_winnbper=2.5,  # 2.5 is enough for phase measure
                                    # (it overestimates the amplitude but we
                                    #  don't use it anyway)
             verbose=1):
    '''
    Estimate the Phase Distortion Deviation (PDD).
    '''

    # Extract the Phase Distortion Deviation (PDD) feature
    # Will need a pitch sync analysis, so resample the f0 accordingly
    f0sps = sp.f0s_resample_pitchsync(f0s, nbperperiod=pdd_sin_nbperperiod)

    # Estimate the sinusoidal parameters
    sinsps, f0sps = sp.sinusoidal.estimate_sinusoidal_params(wav, fs, f0sps, nbper=pdd_sin_winnbper, quadraticfit=True, verbose=verbose-1)

    # Compute PDD from the sinusoidal parameters
    # We don't provide an envelope estimate so the VTF's phase will stay in the computation
    # However, the VTF's phase is ~constant wrt time, thus disapear in the variance measure.
    # (The only risk is to have the VTF's variations that adds to PDD)
    PDD = sp.sinusoidal.estimate_pdd(sinsps, f0sps, fs, pdd_sin_nbperperiod, dftlen, outFscale=True, rmPDtrend=True, extrapDC=True)

    # Resample the feature from pitch synchronous to regular intervals
    PDD = sp.featureresample(f0sps[:,0], PDD, f0s[:,0])

    return PDD

def analysis_nm(wav, fs,
             f0s,                # Has to be continuous (should use analysis_f0postproc)
             PDD,                # Phase Distortion Deviation [2]
                                 # Its length should match f0s'
             pdd_threshold=0.75, # 0.75 as in [2]
             nm_clean=True,      # Use morphological opening and closure to
                                 # clean the mask and avoid learning rubish.
             verbose=1):
    '''
    Estimate the Noise Mask (NM) from the Phase Distortion Deviation (PDD).
    '''

    if f0s.shape[0]!=PDD.shape[0]:
        raise ValueError('f0s size and PDD size do not match!') # pragma: no cover

    shift = np.mean(np.diff(f0s[:,0])) # Get the time shift from the F0 times
    dftlen = (PDD.shape[1]-1)*2 # and the DFT len from the PDD feature

    # The Noise Mask is just a thresholded version of PDD
    HARM = PDD.copy()
    HARM[PDD<=pdd_threshold] = 0
    HARM[PDD>pdd_threshold] = 1

    if nm_clean:
        # Clean the PDD mask to avoid learning rubish details
        import scipy.ndimage
        frq = 70.0 # [Hz]
        morphstruct = np.ones((int(np.round((1.0/frq)/shift)),int(np.round(frq*dftlen/float(fs)))))
        HARM = 1.0-HARM
        HARM = scipy.ndimage.binary_opening(HARM, structure=morphstruct)
        HARM = scipy.ndimage.binary_closing(HARM, structure=morphstruct)
        HARM = 1.0-HARM

    # Avoid noise in low-freqs
    for n in range(len(f0s[:,0])):
        HARM[n,:int(np.round(1.5*f0s[n,1]*dftlen/float(fs)))] = 0.0

    NM = HARM

    return NM

def analysis(wav, fs, f0s=None, f0_min=60, f0_max=600, f0estimator='REAPER',
             shift=0.005,    # Usually 5ms
             dftlen=4096,    # You can adapt this one according to your pipeline
             verbose=1):

    if verbose>0: print('PML Analysis (dur={:.3f}s, fs={}Hz, f0 in [{},{}]Hz, shift={}s, dftlen={})'.format(len(wav)/float(fs), fs, f0_min, f0_max, shift, dftlen))

    f0s = analysis_f0postproc(wav, fs, f0s, f0_min=f0_min, f0_max=f0_max, shift=shift, f0estimator=f0estimator, verbose=verbose)

    SPEC = analysis_spec(wav, fs, f0s, shift=shift, dftlen=dftlen, verbose=verbose)

    PDD = analysis_pdd(wav, fs, f0s, dftlen=dftlen, verbose=verbose)

    NM = analysis_nm(wav, fs, f0s, PDD, verbose=verbose)

    if verbose>2:
        plot_features(wav=wav, fs=fs, f0s=f0s, SPEC=SPEC, PDD=PDD, NM=NM) # pragma: no cover

    return f0s, SPEC, PDD, NM

def plot_features(wav=None, fs=None, f0s=None, SPEC=None, PDD=None, NM=None): # pragma: no cover
    # TODO Could test this by writting in a picture
    tstart = 0.0
    tend = 1.0
    nbview = 0
    if not wav is None: nbview+=1
    if not f0s is None: nbview+=1
    if not SPEC is None: nbview+=1
    if not PDD is None: nbview+=1
    if not NM is None: nbview+=1
    import matplotlib.pyplot as plt
    plt.ion()
    _, axs = plt.subplots(nbview, 1, sharex=True, sharey=False)
    if not isinstance(axs, np.ndarray): axs = np.array([axs])
    view=0
    if not wav is None:
        times = np.arange(len(wav))/float(fs)
        axs[view].plot(times, wav, 'k')
        axs[view].set_ylabel('Waveform\nAmplitude')
        axs[view].grid()
        axs[view].set_xlim((0.0, times[-1]))
        view+=1
    if not f0s is None:
        tstart = f0s[0,0]
        tend = f0s[-1,0]
        axs[view].plot(f0s[:,0], f0s[:,1], 'k')
        axs[view].set_ylabel('F0\nFrequency [Hz]')
        axs[view].grid()
        view+=1
    if not SPEC is None:
        axs[view].imshow(sp.mag2db(SPEC).T, origin='lower', aspect='auto', interpolation='none', extent=(tstart, tend, 0, 0.5*fs), cmap='jet')
        axs[view].set_ylabel('Amp. Envelope\nFrequency [Hz]')
        view+=1
    if not PDD is None:
        axs[view].imshow(PDD.T, origin='lower', aspect='auto', interpolation='none', extent=(tstart, tend, 0, 0.5*fs), cmap='jet', vmin=0.0, vmax=2.0)
        axs[view].set_ylabel('PDD\nFrequency [Hz]')
        view+=1
    if not NM is None:
        axs[view].imshow(NM.T, origin='lower', aspect='auto', interpolation='none', extent=(tstart, tend, 0, 0.5*fs), cmap='Greys', vmin=0.0, vmax=1.0)
        axs[view].set_ylabel('Noise Mask \nFrequency [Hz]')
        view+=1
    axs[-1].set_xlabel('Time [s]')
    from IPython.core.debugger import  Pdb; Pdb().set_trace()

def analysisf(fwav,
        shift=0.005,
        dftlen=4096,
        finf0txt=None, f0estimator='REAPER', f0_min=60, f0_max=600, ff0=None, f0_log=False,
        finf0bin=None, # input f0 file in binary
        fspec=None,
        spec_mceporder=None, # Mel-cepstral order for compressing the spectrogram (typically 59; None: no compression)
        spec_fwceporder=None,# Frequency warped cepstral order (very similar to above, just faster and less precise) (typically 59; None: no compression)
        spec_nbfwbnds=None,  # Number of mel-bands in the compressed half log spectrogram (None: no compression)
        spec_nblinlogbnds=None,  # Number of linear-bands in the compressed half log spectrogram (None: no compression)
        fpdd=None, pdd_mceporder=None, # Mel-cepstral order for compressing PDD spectrogram (typically 59; None: no compression)
        fnm=None, nm_nbfwbnds=None,    # Number of mel-bands in the compressed noise mask (None: no compression)
        preproc_fs=None, # Resample the waveform
        preproc_hp=None, # Cut-off of high-pass filter (e.g. 20Hz)
        verbose=1):

    wav, fs, _ = sp.wavread(fwav)

    if len(wav)==0: raise ValueError('The waveform in {} is empty.'.format(fwav))

    if verbose>0: print('PML Analysis (dur={:.3f}s, fs={}Hz, f0 in [{},{}]Hz, shift={}s, dftlen={})'.format(len(wav)/float(fs), fs, f0_min, f0_max, shift, dftlen))

    if (not preproc_fs is None) and (preproc_fs!=fs):
        if verbose>0: print('    Resampling the waveform (new fs={}Hz)'.format(preproc_fs))
        wav = sp.resample(wav, fs, preproc_fs, method=2, deterministic=True)
        fs = preproc_fs

    if not preproc_hp is None:
        if verbose>0: print('    High-pass filter the waveform (cutt-off={}Hz)'.format(preproc_hp))
        b, a = sig.butter(4, preproc_hp/(fs/0.5), btype='high')
        wav = sig.filtfilt(b, a, wav)

    f0s = None
    if finf0txt:
        f0s = np.loadtxt(finf0txt)

    # read input f0 file in float32 (ljuvela)
    if finf0bin:
        f0s = np.fromfile(finf0bin, dtype=np.float32)

    f0s = analysis_f0postproc(wav, fs, f0s, f0_min=f0_min, f0_max=f0_max, shift=shift, f0estimator=f0estimator, verbose=verbose)
    if verbose>2: f0sori=f0s.copy()

    if ff0:
        f0_values = f0s[:,1]
        if verbose>0: print('    Output F0 {} in: {}'.format(f0_values.shape, ff0))
        if f0_log: f0_values = np.log(f0_values)
        if os.path.dirname(ff0)!='' and (not os.path.isdir(os.path.dirname(ff0))): os.mkdir(os.path.dirname(ff0))
        f0_values.astype(np.float32).tofile(ff0)

    SPEC = None
    if fspec:
        SPEC = analysis_spec(wav, fs, f0s, shift=shift, dftlen=dftlen, verbose=verbose)
        if verbose>2: SPECori=SPEC.copy()
        if not spec_mceporder is None: # pragma: no cover
                                       # Cannot test this because it needs SPTK
            SPEC = sp.spec2mcep(SPEC, sp.bark_alpha(fs), order=spec_mceporder)
        if not spec_fwceporder is None:
            SPEC = sp.loghspec2fwcep(np.log(abs(SPEC)), fs, order=spec_fwceporder)
        if not spec_nbfwbnds is None:
            SPEC = sp.linbnd2fwbnd(np.log(abs(SPEC)), fs, dftlen, spec_nbfwbnds)
        if not spec_nblinlogbnds is None:
            SPEC = np.log(abs(SPEC))
        if verbose>0: print('    Output Spectrogram size={} in: {}'.format(SPEC.shape, fspec))
        if os.path.dirname(fspec)!='' and (not os.path.isdir(os.path.dirname(fspec))): os.mkdir(os.path.dirname(fspec))
        SPEC.astype(np.float32).tofile(fspec)

    PDD = None
    if fpdd or fnm:
        PDD = analysis_pdd(wav, fs, f0s, dftlen=dftlen, verbose=verbose)
        if verbose>2: PDDori=PDD.copy()

    if fpdd:
        if not pdd_mceporder is None:  # pragma: no cover
                                       # Cannot test this because it needs SPTK
            # If asked, compress PDD
            PDD[PDD<0.001] = 0.001 # From COVAREP
            PDD = sp.spec2mcep(PDD, sp.bark_alpha(fs), pdd_mceporder)
        if verbose>0: print('    Output PDD size={} in: {}'.format(PDD.shape, fpdd))
        if os.path.dirname(fpdd)!='' and (not os.path.isdir(os.path.dirname(fpdd))): os.mkdir(os.path.dirname(fpdd))
        PDD.astype(np.float32).tofile(fpdd)

    NM = None
    if verbose>2: NMori=None
    if fnm:
        NM = analysis_nm(wav, fs, f0s, PDD, verbose=verbose)
        if verbose>2: NMori=NM.copy()
        # If asked, compress NM
        if nm_nbfwbnds:
            # If asked, compress the noise mask using a number of mel bands
            NM = sp.linbnd2fwbnd(NM, fs, dftlen, nm_nbfwbnds)
        if verbose>0: print('    Output Noise Mask size={} in: {}'.format(NM.shape, fnm))
        if os.path.dirname(fnm)!='' and (not os.path.isdir(os.path.dirname(fnm))): os.mkdir(os.path.dirname(fnm))
        NM.astype(np.float32).tofile(fnm)

    if verbose>2:
        plot_features(wav=wav, fs=fs, f0s=f0sori, SPEC=SPECori, PDD=PDDori, NM=NMori) # pragma: no cover

def main(argv):
    argpar = argparse.ArgumentParser()
    argpar.add_argument("wavfile", help="Input wav file")
    argpar.add_argument("--shift", default=0.005, type=float, help="time step[s] between the input frames (def. 0.005s)")
    argpar.add_argument("--dftlen", default=4096, type=int, help="Number of bins in the DFT (def. 4096)")
    argpar.add_argument("--inf0txt", default=None, help="Given f0 file")
    argpar.add_argument("--inf0bin", default=None, help="Given f0 file (single precision float binary)")
    argpar.add_argument("--f0_min", default=60, type=float, help="Minimal possible f0[Hz] value (def. 60Hz)")
    argpar.add_argument("--f0_max", default=600, type=float, help="Maximal possible f0[Hz] value (def. 600Hz)")
    argpar.add_argument("--f0", default=None, help="Output f0 file")
    argpar.add_argument("--f0_log", action='store_true', help="Output f0 file with log Hertz values instead of linear Hertz (def. False)")
    argpar.add_argument("--spec", default=None, help="Output spectrum-related file")
    argpar.add_argument("--spec_mceporder", default=None, type=int, help="Mel-cepstral order for the spectrogram (None:uncompressed; typically 59)")
    argpar.add_argument("--spec_fwceporder", default=None, type=int, help="Frequency warped cepstral order (very similar to above, just faster and less precise) (typically 59)")
    argpar.add_argument("--spec_nbfwbnds", default=None, type=int, help="Number of mel-bands in the compressed half log spectrogram (None:uncompressed; typically 129 (should be odd size as long as full spectrum size if power of 2 (even size)")
    argpar.add_argument("--spec_nblinlogbnds", default=None, type=int, help="Number of frequency bands in the compressed half log spectrogram (None:uncompressed; typically 129 (should be odd size as long as full spectrum size if power of 2 (even size)")
    argpar.add_argument("--pdd", default=None, help="Output Phase Distortion Deviation (PDD) file")
    argpar.add_argument("--pdd_mceporder", default=None, type=int, help="Cepstral order for PDD (None:uncompressed; typically 59)")
    argpar.add_argument("--nm", default=None, help="Output noise mask")
    argpar.add_argument("--nm_nbfwbnds", default=None, type=int, help="Number of mel-bands in the compressed noise mask (None:uncompressed; typically 33)")
    argpar.add_argument("--preproc_fs", default=None, type=float, help="[Hz] Resample the waveform before analysis.")
    argpar.add_argument("--preproc_hp", default=None, type=float, help="[Hz] High-pass the waveform before analysis.")
    argpar.add_argument("--verbose", default=1, type=int, help="Output some information")
    args = argpar.parse_args(argv)

    analysisf(args.wavfile,
              shift=args.shift,
              dftlen=args.dftlen,
              finf0txt=args.inf0txt, f0_min=args.f0_min, f0_max=args.f0_max, ff0=args.f0, f0_log=args.f0_log,
              finf0bin=args.inf0bin,
              fspec=args.spec, spec_mceporder=args.spec_mceporder, spec_fwceporder=args.spec_fwceporder, spec_nbfwbnds=args.spec_nbfwbnds, spec_nblinlogbnds=args.spec_nblinlogbnds,
              fpdd=args.pdd, pdd_mceporder=args.pdd_mceporder,
              fnm=args.nm, nm_nbfwbnds=args.nm_nbfwbnds,
              preproc_fs=args.preproc_fs, preproc_hp=args.preproc_hp,
              verbose=args.verbose)

if  __name__ == "__main__" :            # pragma: no cover
    main(sys.argv[1:])
