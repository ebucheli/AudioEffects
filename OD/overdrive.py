import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io.wavfile import read


def main():

    # load file
    fs, x = read('gtr_jazz.wav')
    channels = len(x[0])
    print('Sample Rate: ',fs)
    print('Channels: ', channels)

    x_left = np.array(x[:,0],dtype=np.float16)
    x_right = np.array(x[:,1],dtype=np.float16)

    x_left /= np.max(np.abs(x_left))
    x_right /= np.max(np.abs(x_right))

    print("The file is {:.2f} seconds".format(len(x_left)/fs))
    plot_wave(x_left,x_right,fs)

    # Upsampling

    x_left_up = np.zeros((len(x_left)*4,))
    x_right_up = np.zeros((len(x_right)*4,))

    x_left_up[::4] = x_left
    x_right_up[::4] = x_right

    from scipy.signal import butter, lfilter, freqz

    cutoff = (len(x_left)/len(x_left_up))/2

    b, a = butter(5, cutoff, btype='low', analog=False)
    y_left = lfilter(b, a, x_left_up)
    y_right = lfilter(b, a, x_right_up)

    #Apply Nonlinearity

    def nonlinearity(x):
        return x/((1+np.abs(x)**2)**(1/2))

    y_left = nonlinearity(15*y_left)
    y_right = nonlinearity(15*y_right)

    b, a = butter(5, cutoff/2, btype='low', analog=False)
    y_left = lfilter(b, a, y_left)
    y_right = lfilter(b, a, y_right)

    #Downsample

    y_left = np.array(y_left[::4])
    y_left /= np.max(np.absolute(y_left))

    y_right = np.array(y_right[::4])
    y_right /= np.max(np.absolute(y_right))

    #Sum with clean signal

    drive_lvl = 0.7
    clean_lvl = 0.7

    y_sum_left = drive_lvl*y_left + clean_lvl*x_left
    y_sum_left /= np.max(np.abs(y_sum_left))

    y_sum_right = drive_lvl*y_right + clean_lvl*x_right
    y_sum_right /= np.max(np.abs(y_sum_right))

    plt.figure()

    plt.subplot(3,2,1)
    librosa.display.waveplot(x_left,fs)
    plt.title('Original Left')

    plt.subplot(3,2,2)
    librosa.display.waveplot(x_right,fs)
    plt.title('Original Right')

    plt.subplot(3,2,3)
    librosa.display.waveplot(y_left,fs)
    plt.title('Overdrive Left')

    plt.subplot(3,2,4)
    librosa.display.waveplot(y_right,fs)
    plt.title('Overdrive Right')

    plt.subplot(3,2,5)
    librosa.display.waveplot(y_sum_left,fs)
    plt.title('Mix Left')

    plt.subplot(3,2,6)
    librosa.display.waveplot(y_sum_right,fs)
    plt.title('Mix Right')

    plt.tight_layout()
    plt.show()

    #Write Wav File

    y_stereo = np.zeros((len(y_left),channels))

    y_stereo[:,0] = y_left
    y_stereo[:,1] = y_right

    from scipy.io.wavfile import write

    write('gtr_jazz_TS_noFilter.wav', fs, y_stereo)

    y_stereo_sum = np.zeros((len(y_sum_left),channels))

    y_stereo_sum[:,0] = y_sum_left
    y_stereo_sum[:,1] = y_sum_right

    from scipy.io.wavfile import write

    write('gtr_jazz_TS_SUM.wav',fs,y_stereo_sum)

def plot_wave(x_left, x_right, fs):

    plt.figure()
    plt.subplot(2,1,1)
    librosa.display.waveplot(x_left,fs)
    plt.xlabel('Seconds')
    plt.title('Left Channel Waveform')

    plt.subplot(2,1,2)
    librosa.display.waveplot(x_right,fs)
    plt.xlabel('Seconds')
    plt.title('Right Channel Waveform')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
