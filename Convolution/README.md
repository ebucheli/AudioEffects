# Convolution Reverb

Uses an FFT Convolution to combine some input file and an impulse response.
In our examples these are applied for reverb.

## Usage

Needs a reference to the input file, the impulse response and a name for the generated file.

Usage: `Convolution.py INFILE IR OUTFILE`

Example:

To generate convolved result using the file `voice.wav`, impulse response `ir1.wav` and save as `ex_response1.wav`.

`python Convolution.py voice.wav ir1.wav ex_response1.wav`
