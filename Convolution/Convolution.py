#Edoardo Bucheli Susarrey 2016

#This implementation of the convolution takes advantage of Fourier's Theorem.
#So it uses an FFT to make a faster convolution.
#This is done with (scipy's fftconvolve formula
#It applies an FFT to both signals and multiplies them

import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

import click

@click.command()
@click.argument('infile')
@click.argument('ir')
@click.argument('outfile')

def main(infile, ir,outfile):

    # Load Files

    y, sr_y = librosa.load(infile,sr = 44100)
    x, sr_x = librosa.load(ir,sr = 44100)

    print('The length of the IR is {:.2f} seconds'.format(y.shape[0]/sr_y))
    print('The length of the signal {:.2f} seconds'.format(x.shape[0]/sr_x))

    plt.figure()
    plt.subplot(3,1,1)
    librosa.display.waveplot(x,sr=sr_x)
    plt.title('Signal')

    plt.subplot(3,1,2)
    librosa.display.waveplot(y,sr=sr_y)
    plt.title('IR')

    from scipy.signal import fftconvolve

    #Apply scipy's fftconvolve function
    z = fftconvolve(x,y,mode='full')

    #Normalize Signal
    z_norm = z/np.amax(np.abs(z))

    from scipy.io.wavfile import write

    #Make New wav file
    write(outfile,sr_x,z_norm)

    plt.subplot(3,1,3)
    librosa.display.waveplot(z_norm,sr=sr_x)
    plt.title('Wet Signal')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
