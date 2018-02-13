#!/usr/bin/env python
'''

Description

If using files, (call by command line or from python):
    all the inputs are raw float32 vectors files that are reshaped by the number
    of f0 values in ff0.

There are three safe patches that were not described in the publication[1]:
    (These are not critical, they might remove a few artifacts here and there).
    * The noise mask is slightly low-passed (smoothed) across frequency
        (def. 9 bins freq. window), in order to avoid cliffs in frequency domain
        that end up creating Gibbs phenomenon in the time domain.
    * High-pass filtering (def. 0.5*f0 cut-off)
        This centers each synthesized segment around zero, to avoid cutting
        any DC residual component (e.g. comming from the spectral envelope).
    * Short half-window  (def. 1ms (yes, one ms)) on the left of the pulse,
        in order to avoid any pre-echos.

Reference
    [1] G. Degottex, P. Lanchantin, and M. Gales, "A Pulse Model in Log-domain
        for a Uniform Synthesizer," in Proc. 9th Speech Synthesis Workshop
        (SSW9), 2016.

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
import numpy as np
np.random.seed(0) # Generate always the same "random" numbers, for debugging.
import scipy

import sigproc as sp

def synthesize(fs, f0s, SPEC, NM=None, wavlen=None
                , ener_multT0=False
                , nm_cont=False     # If False, force binary state of the noise mask (by thresholding at 0.5)
                , nm_lowpasswinlen=9
                , hp_f0coef=0.5     # factor of f0 for the cut-off of the high-pass filter (def. 0.5*f0)
                , antipreechohwindur=0.001 # [s] Use to damp the signal at the beginning of the signal AND at the end of it
                # Following options are for post-processing the features, after the generation/transformation and thus before waveform synthesis
                , pp_f0_rmsteps=False # Removes steps in the f0 curve
                                      # (see sigproc.resampling.f0s_rmsteps(.) )
                , pp_f0_smooth=None   # Smooth the f0 curve using median and FIR filters of given window duration [s]
                , pp_atten1stharminsilences=None # Typical value is -25
                , verbose=1):

    # Copy the inputs to avoid modifying them
    f0s = f0s.copy()
    SPEC = SPEC.copy()
    if not NM is None: NM = NM.copy()
    else:              NM = np.zeros(SPEC.shape)


    # Check the size of the inputs
    if f0s.shape[0]!=SPEC.shape[0]:
        raise ValueError('F0 size {} and spectrogram size {} do not match'.format(len(f0), SPEC.shape[0]))
    if not NM is None:
        if SPEC.shape!=NM.shape:
            raise ValueError('spectrogram size {} and NM size {} do not match.'.format(SPEC.shape, NM.shape))

    if wavlen==None: wavlen = int(np.round(f0s[-1,0]*fs))
    dftlen = (SPEC.shape[1]-1)*2
    shift = np.median(np.diff(f0s[:,0]))
    if verbose>0:
        print('PM Synthesis (dur={}s, fs={}Hz, f0 in [{:.0f},{:.0f}]Hz, shift={}s, dftlen={})'.format(wavlen/float(fs), fs, np.min(f0s[:,1]), np.max(f0s[:,1]), shift, dftlen))


    # Prepare the features

    # Enforce continuous f0
    f0s[:,1] = np.interp(f0s[:,0], f0s[f0s[:,1]>0,0], f0s[f0s[:,1]>0,1])
    # If asked, removes steps in the f0 curve
    if pp_f0_rmsteps:
        f0s = sp.f0s_rmsteps(f0s)
    # If asked, smooth the f0 curve using median and FIR filters
    if not pp_f0_smooth is None:
        print('    Smoothing f0 curve using {}[s] window'.format(pp_f0_smooth))
        import scipy.signal as sig
        lf0 = np.log(f0s[:,1])
        bcoefslen = int(0.5*pp_f0_smooth/shift)*2+1
        lf0 = sig.medfilt(lf0, bcoefslen)
        bcoefs = np.hamming(bcoefslen)
        bcoefs = bcoefs/sum(bcoefs)
        lf0 = sig.filtfilt(bcoefs, [1], lf0)
        f0s[:,1] = np.exp(lf0)

    if not NM is None:
        # Remove noise below f0, as it is supposed to be already the case
        for n in range(NM.shape[0]):
            NM[n,:int((float(dftlen)/fs)*2*f0s[n,1])] = 0.0

    if not nm_cont:
        print('    Forcing binary noise mask')
        NM[NM<=0.5] = 0.0 # To be sure that voiced segments are not hoarse
        NM[NM>0.5] = 1.0  # To be sure the noise segments are fully noisy

    # Generate the pulse positions [1](2) (i.e. the synthesis instants, the GCIs in voiced segments)
    ts = [0.0]
    while ts[-1]<float(wavlen)/fs:
        cf0 = np.interp(ts[-1], f0s[:,0], f0s[:,1])
        if cf0<50.0: cf0 = 50
        ts.append(ts[-1]+(1.0/cf0))
    ts = np.array(ts)
    f0s = np.vstack((ts, np.interp(ts, f0s[:,0], f0s[:,1]))).T


    # Resample the features to the pulse positions

    # Spectral envelope uses the nearest, to avoid over-smoothing
    SPECR = np.zeros((f0s.shape[0], dftlen/2+1))
    for n, t in enumerate(f0s[:,0]): # Nearest: Way better for plosives
        idx = int(np.round(t/shift))
        idx = np.clip(idx, 0, SPEC.shape[0]-1)
        SPECR[n,:] = SPEC[idx,:]

    # Keep trace of the median energy [dB] over the whole signal
    ener = np.mean(SPECR, axis=1)
    idxacs = np.where(sp.mag2db(ener) > sp.mag2db(np.max(ener))-30)[0] # Get approx active frames # TODO Param
    enermed = sp.mag2db(np.median(ener[idxacs])) # Median energy [dB]
    ener = sp.mag2db(ener)

    # Resample the noise feature to the pulse positions
    # Smooth the frequency response of the mask in order to avoid Gibbs
    # (poor Gibbs nobody want to see him)
    nm_lowpasswin = np.hanning(nm_lowpasswinlen)
    nm_lowpasswin /= np.sum(nm_lowpasswin)
    NMR = np.zeros((f0s.shape[0], dftlen/2+1))
    for n, t in enumerate(f0s[:,0]):
        idx = int(np.round(t/shift)) # Nearest is better for plosives
        idx = np.clip(idx, 0, NM.shape[0]-1)
        NMR[n,:] = NM[idx,:]
        if nm_lowpasswinlen>1:
            NMR[n,:] = scipy.signal.filtfilt(nm_lowpasswin, [1.0], NMR[n,:])

    NMR = np.clip(NMR, 0.0, 1.0)

    # The complete waveform that we will fill with the pulses
    wav = np.zeros(wavlen)
    # Half window on the left of the synthesized segment to avoid pre-echo
    dampinhwin = np.hanning(1+2*int(np.round(antipreechohwindur*fs))) # 1ms forced dampingwindow
    dampinhwin = dampinhwin[:(len(dampinhwin)-1)/2+1]

    for n, t in enumerate(f0s[:,0]):
        f0 = f0s[n,1]

        if verbose>1: print "\rPM Synthesis (python) t={:4.3f}s f0={:3.3f}Hz               ".format(t,f0),

        # Window's length
        nbper = 4
        # TODO It should be ensured that the beggining and end of the
        #      noise is within the window. Nothing is doing this currently!
        winlen = int(np.max((0.050*fs, nbper*fs/f0))/2)*2+1 # Has to be odd
        # TODO We also assume that the VTF's decay is shorter
        #      than nbper-1 periods (dangerous with high pitched tense voice).
        if winlen>dftlen: raise ValueError('winlen({})>dftlen({})'.format(winlen, dftlen))

        # Set the rough position of the pulse in the window (the closest sample)
        # We keep a third of the window (1 period) on the left because the
        # pulse signal is minimum phase. And 2/3rd (remaining 2 periods)
        # on the right to let the VTF decay.
        pulseposinwin = int((1.0/nbper)*winlen)

        # The sample indices of the current pulse wrt. the final waveform
        winidx = int(round(fs*t)) + np.arange(winlen)-pulseposinwin


        # Build the pulse spectrum

        # Let start with a Dirac
        S = np.ones(dftlen/2+1, dtype=np.complex64)

        # Add the delay to place the Dirac at the "GCI": exp(-j*2*pi*t_i)
        delay = -pulseposinwin - fs*(t-int(round(fs*t))/float(fs))
        S *= np.exp((delay*2j*np.pi/dftlen)*np.arange(dftlen/2+1))

        # Add the spectral envelope
        # Both amplitude and phase
        E = SPECR[n,:] # Take the amplitude from the given one
        if hp_f0coef!=None:
            # High-pass it to avoid any residual DC component.
            fcut = hp_f0coef*f0
            if not pp_atten1stharminsilences is None and ener[n]-enermed<pp_atten1stharminsilences:
                fcut = 1.5*f0 # Try to cut between first and second harm
            HP = sp.butter2hspec(fcut, 4, fs, dftlen, high=True)
            E *= HP
            # Not necessarily good as it is non-causal, so make it causal...
            # ... together with the VTF response below.
        # Build the phase of the envelope from the amplitude
        E = sp.hspec2minphasehspec(E, replacezero=True) # We spend 2 FFT here!
        S *= E # Add it to the current pulse

        # Add energy correction wrt f0.
        # STRAIGHT and AHOCODER vocoders do it.
        # (why ? to equalize the energy when changing the pulse's duration ?)
        if ener_multT0:
            S *= np.sqrt(fs/f0)

        # Generate the segment of Gaussian noise
        # Use mid-points before/after pulse position
        if n>0: leftbnd=int(np.round(fs*0.5*(f0s[n-1,0]+t)))
        else:   leftbnd=int(np.round(fs*(t-0.5/f0s[n,1]))) # int(0)
        if n<f0s.shape[0]-1: rightbnd=int(np.round(fs*0.5*(t+f0s[n+1,0])))-1
        else:                rightbnd=int(np.round(fs*(t+0.5/f0s[n,1])))   #rightbnd=int(wavlen-1)
        gausswinlen = rightbnd-leftbnd # The length of the noise segment
        gaussnoise4win = np.random.normal(size=(gausswinlen)) # The noise

        GN = np.fft.rfft(gaussnoise4win, dftlen) # Move the noise to freq domain
        # Normalize it by its energy (@Yannis, That's your answer at SSW9!)
        GN /= np.sqrt(np.mean(np.abs(GN)**2))
        # Place the noise within the pulse's window
        delay = (pulseposinwin-(leftbnd-winidx[0]))
        GN *= np.exp((delay*2j*np.pi/dftlen)*np.arange(dftlen/2+1))

        # Add it to the pulse spectrum, under the condition of the mask
        S *= GN**NMR[n,:]

        # That's it! the pulse spectrum is ready!

        # Move it to time domain
        deter = np.fft.irfft(S)[0:winlen]

        # Add half window on the left of the synthesized segment
        # to avoid any possible pre-echo
        deter[:leftbnd-winidx[0]-len(dampinhwin)] = 0.0
        deter[leftbnd-winidx[0]-len(dampinhwin):leftbnd-winidx[0]] *= dampinhwin

        # Add half window on the right
        # to avoid cutting the VTF response abruptly
        deter[-len(dampinhwin):] *= dampinhwin[::-1]

        # Write the synthesized segment in the final waveform
        if winidx[0]<0 or winidx[-1]>=wavlen:
            # The window is partly outside of the waveform ...
            wav4win = np.zeros(winlen)
            # ... thus copy only the existing part
            itouse = np.logical_and(winidx>=0,winidx<wavlen)
            wav[winidx[itouse]] += deter[itouse]
        else:
            wav[winidx] += deter

    if verbose>1: print '\r                                                               \r',

    if verbose>2:
        import matplotlib.pyplot as plt
        plt.ion()
        f, axs = plt.subplots(3, 1, sharex=True, sharey=False)
        times = np.arange(len(wav))/float(fs)
        axs[0].plot(times, wav, 'k')
        axs[0].set_ylabel('Waveform\nAmplitude')
        axs[0].grid()
        axs[1].plot(f0s[:,0], f0s[:,1], 'k')
        axs[1].set_ylabel('F0\nFrequency [Hz]')
        axs[1].grid()
        axs[2].imshow(sp.mag2db(SPEC).T, origin='lower', aspect='auto', interpolation='none', extent=(f0s[0,0], f0s[-1,0], 0, 0.5*fs))
        axs[2].set_ylabel('Amp. Envelope\nFrequency [Hz]')

        from IPython.core.debugger import  Pdb; Pdb().set_trace()

    return wav



def synthesizef(fs, shift=0.005, dftlen=4096, ff0=None, flf0=None, fspec=None, fmcep=None, fpdd=None, fmpdd=None, fnm=None, fbndnm=None, nm_cont=False, fsyn=None, verbose=1):
    '''
    Call the synthesis from python using file inputs and outputs
    '''
    if ff0:
        f0 = np.fromfile(ff0, dtype=np.float32)
    if flf0:
        f0 = np.fromfile(flf0, dtype=np.float32)
        f0[f0>0] = np.exp(f0[f0>0])
    ts = (shift)*np.arange(len(f0))
    f0s = np.vstack((ts, f0)).T

    if fspec:
        SPEC = np.fromfile(fspec, dtype=np.float32)
        SPEC = SPEC.reshape((len(f0), -1))
    if fmcep:
        MCEP = np.fromfile(fmcep, dtype=np.float32)
        MCEP = MCEP.reshape((len(f0), -1))
        SPEC = sp.mcep2spec(MCEP, sp.bark_alpha(fs), dftlen)

    NM = None
    pdd_thresh = 0.75 # For this value, see:
        # G. Degottex and D. Erro, "A uniform phase representation for the harmonic model in speech synthesis applications," EURASIP, Journal on Audio, Speech, and Music Processing - Special Issue: Models of Speech - In Search of Better Representations, vol. 2014, iss. 1, p. 38, 2014.
    if fpdd:
        PDD = np.fromfile(fpdd, dtype=np.float32)
        PDD = PDD.reshape((len(f0), -1))
        NM = PDD.copy()
        NM[PDD<pdd_thresh] = 0.0
        NM[PDD>pdd_thresh] = 1.0
    if fmpdd:
        MPDD = np.fromfile(fmpdd, dtype=np.float32)
        MPDD = MPDD.reshape((len(f0), -1))
        PDD = sp.mcep2spec(MPDD, sp.bark_alpha(fs), dftlen)
        NM = PDD.copy()
        NM[PDD<pdd_thresh] = 0.0
        NM[PDD>pdd_thresh] = 1.0

    if fnm:
        NM = np.fromfile(fnm, dtype=np.float32)
        NM = NM.reshape((len(f0), -1))
    if fbndnm:
        BNDNM = np.fromfile(fbndnm, dtype=np.float32)
        BNDNM = BNDNM.reshape((len(f0), -1))
        NM = sp.fwbnd2linbnd(BNDNM, fs, dftlen)

    syn = synthesize(fs, f0s, SPEC, NM=NM, nm_cont=nm_cont, verbose=verbose)
    if fsyn:
        sp.wavwrite(fsyn, syn, fs, norm_abs=True, verbose=verbose)

    return syn

if  __name__ == "__main__" :
    '''
    Call the synthesis from the command line
    '''

    argpar = argparse.ArgumentParser()
    argpar.add_argument("synthfile", help="Output synthesis file")
    argpar.add_argument("--f0file", default=None, help="Input f0[Hz] file")
    argpar.add_argument("--logf0file", default=None, help="Input f0[log Hz] file")
    argpar.add_argument("--specfile", default=None, help="Input amplitude spectrogram [linear values]")
    argpar.add_argument("--mcepfile", default=None, help="Input amplitude spectrogram [mel-cepstrum values]")
    argpar.add_argument("--pddfile", default=None, help="Input Phase Distortion Deviation file [linear values]")
    argpar.add_argument("--mpddfile", default=None, help="Input Phase Distortion Deviation file [mel-cepstrum values]")
    argpar.add_argument("--nmfile", default=None, help="Output Noise Mask [linear values in [0,1] ]")
    argpar.add_argument("--bndnmfile", default=None, help="Output Noise Mask [compressed in bands with values still in [0,1] ]")
    argpar.add_argument("--nm_cont", action='store_true', help="Allow continuous values for the noisemask (def. False)")
    argpar.add_argument("--fs", default=16000, type=int, help="Sampling frequency[Hz]")
    argpar.add_argument("--shift", default=0.005, type=float, help="Time step[s] between the frames")
    #argpar.add_argument("--dftlen", dftlen=4096, type=float, help="Size of the DFT for extracting the features")
    argpar.add_argument("--verbose", default=1, help="Output some information")
    args = argpar.parse_args()
    args.dftlen = 4096

    synthesizef(args.fs, shift=args.shift, dftlen=args.dftlen, ff0=args.f0file, flf0=args.logf0file, fspec=args.specfile, fmcep=args.mcepfile, fnm=args.nmfile, fbndnm=args.bndnmfile, nm_cont=args.nm_cont, fpdd=args.pddfile, fmpdd=args.mpddfile, fsyn=args.synthfile, verbose=args.verbose)
