---
layout: page
author: Cem Kose
title: Speech Recognition Notes
excerpt: Study for Voice and Emotion Detection.
---

# Study for Voice and Emotion Detection

## Glosary


- Digital Signal Processing (DSP) = DSP is the use of digital processing to perform a wide variety of signal processing operations.
- Waveform = A waveform describes a wave by graphing how an air molecule is displaced, over time.
- Pulse-Code Modulation (PCM) = Pulse-code modulation (PCM) is a method used to represent sampled analog signals.
- Frequency = Frequency is the measure of how many times the waveform repeats in a given amount of time.
- Hertz (Hz) = The common unit of measurements for frequency. Hz represents the number the waveform repetitions per second.
- Ampliture = A measure of how much a molecule is displaced from its resting position.
- Sampling = In DSP, sampling is the reduction of continuous-time signal to a discrete-time signal.
- Formant = A formant is a concentration of acoustic energy around a particular frequency in the speech wave.
- Nyquist Theorem = The Nyquist Theorem states that in order to adequately reproduce a signal, it should be periodically sampled at a rate that is 2X the highest frequency you wish to record.
- Windowing = Windowing is the process of taking a small subset of a larger dataset for processing or analysis.
- Spectrogram = A spectrogram is a visual representation of a signal as it varies with time.
- Fourier transform (FT) = FT is a mathematical transform which decomposes a function ( a signal) into its constituent frequencies.
- Short-time Fourier transform (STFT) = STFT is a Fourier-related transform used to determine the sinusoidal frequency and phase content of local sections of a signal as it changes over time.
- Voltage root-mean square (VRMS) = VRMS is defined as square root of the mean of the squares of the values for the one time period of the sine wave.

## Helpful links

- [Interactive introduction to waveforms](https://pudding.cool/2018/02/waveforms/)
- [What are formants?](https://person2.sol.lu.se/SidneyWood/praate/whatform.html)
- [Nyquist Theorem](http://microscopy.berkeley.edu/courses/dib/sections/02Images/sampling.html)
- [VRMS](http://www.referencedesigner.com/rfcal/cal_04.php)

## Waveform

Speech signals are sound signals, defined as pressure variations travelling through the air.


A speech signal is then represented by a sequence of numbers $ x_n $, which represent the relative air pressure at time-instant n∈ℕ.


This representation is known as pulse code modulation often abbreviated as PCM.


The accuracy of this representation is then specified by two factors;


1. the sampling frequency (the step in time between $ n $ and $ n+1 $).
2. the accuracy and distribution of amplitudes of $ x_n $ .

### Sampling Rate

In DSP Sampling is the reduction of a continuous-time signal to a discrete-time signal.

A common example is the conversion of a sound wave (a continuous signal) to a sequence of samples (a discrete-time signal).

An important aspect of Sampling is the Nyquist Theorem. The Nyquist Theorem states that in order to adequately reproduce a continuous-time signal it should sampled at a rate that is 2X the highest frequency you wish to record.
