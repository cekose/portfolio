---
layout: page
author: Cem Kose
title: Test Code Snipoet Post
excerpt: Testing adding code snippets.
---

Just testing css for code snippets.


```python

# Reading in a new .wav file
sample_rate, signal = wavfile.read('test.wav')

# Splicing the signal for 0 to 3.3 secs
signal = signal[0:int(3.3 * sample_rate)]

# Using np.linspace to create evenly space
# sequence of samples.
# np.linspace(start = , stop= , number of
# items to generate within the range= )
# in this case starts at 0s, stops at 4.2375625s,
# generates 67801 items.
time = np.linspace(0., samples.shape[0] / samplerate, signal.shape[0])

# plots time variable in x axis and left
# channel amplitude on y axis
plt.plot(time, signal[:, 0], label="Left channel")
# plots time variable in x axis and right channel
# amplitude on y axis
plt.plot(time, signal[:, 1], label="Right channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

```
