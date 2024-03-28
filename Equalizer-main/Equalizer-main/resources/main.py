# Main Program Libraries
# ----------------------
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QTimer
import pyqtspecgram
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from scipy.signal import spectrogram
import numpy as np
import threading
import pandas as pd
from math import ceil
import json








class Equalizer(QMainWindow):
# ---------------------------
    def __init__(self):
        # Ui importing and setting window title
        super(Equalizer, self).__init__()
        uic.loadUi("./equalizer.ui", self)
        self.setWindowTitle("Equalizer")
        self.show()


        # (InputPlot, OutputPlot) plots for time domain, (InputFFTPlot, OutputFFTPlot) plots for frequency domain, 
        # (InputSpectrogramPlot, OutputSpectrogramPlot) spectrograms
        self.plot_widgets = [self.InputPlot, self.OutputPlot,
                             self.InputFFTPlot, self.OutputFFTPlot,
                             self.InputSpectrogramPlot, self.OutputSpectrogramPlot]
        
        # No of Sliders enabled in each mode
        self.enabled_sliders = {'Uniform Range Mode': 10, 'Musical Instruments Mode': 4, 'Animal Sounds Mode': 4, 'ECG Abnormalities Mode': 3}
        
        
        self.sliders = [getattr(self, f"Slider{i+1}") for i in range(10)]                    # List for all sliders [slider1, slider2, etc]


        self.sliders_lineEdits = [getattr(self, f"lineEdit{i+1}") for i in range(10)]        # List for all sliders line edits [lineEdit1, lineEdit2, etc]
    

        self.sliders_lables = [getattr(self, f"label{i+1}") for i in range(10)]              # List for all sliders labels [label1, label2, etc]
        
        # Input and Output data (Data, Time, FFT, FFTFREQ)
        self.input, self.output = [np.array([]) for _ in range(4)], [np.array([]) for _ in range(4)]

        # Data Ranges
        self.data_ranges = []


        # Set up sliders
        for i in range(10):
            self.sliders[i].setMinimum(-30)
            self.sliders[i].setMaximum(30)
            self.sliders[i].setValue(0)
            self.sliders_lineEdits[i].setText("0")
        

        with open('./data_ranges.json', 'r') as file:
            freq_ranges = json.load(file)

        self.modes_freq_ranges = [freq_ranges['animal_freq_ranges'], freq_ranges['music_instrument_ranges'], freq_ranges['ecg_arrythmia_ranges']]


        self.PlayButton.clicked.connect(lambda: self.play_signal(1))
        self.PlayButton2.clicked.connect(lambda: self.play_signal(2))

        self.UploadButton.clicked.connect(self.open_dialog_box)
        
        self.ZoomInButton.clicked.connect(self.zoom_in)
        self.ZoomOutButton.clicked.connect(self.zoom_out)
        
        self.ZoomInButton2.clicked.connect(self.zoom_in)
        self.ZoomOutButton2.clicked.connect(self.zoom_out)
        
        self.ResetButton.clicked.connect(self.reset_handler)
        
        self.SaveButton.clicked.connect(self.save_handler)
        
        self.RewindButton.clicked.connect(self.rewind_handler)
        
        self.ChannelsReset.clicked.connect(self.reset_all_channels)
        
        self.SpectCheckbox.stateChanged.connect(self.toggle_spect)

        self.InputSpectrogramPlot.setVisible(self.SpectCheckbox.isChecked())
        self.OutputSpectrogramPlot.setVisible(self.SpectCheckbox.isChecked())

        self.ModeCombobox.currentIndexChanged.connect(lambda: self.check_mode(self.ModeCombobox.currentText()))

        self.MultModeCombobox.currentIndexChanged.connect(lambda: self.check_mult_mode(self.MultModeCombobox.currentText()))
        self.StdSlider.sliderReleased.connect(lambda: self.check_mult_mode(self.MultModeCombobox.currentText()))


        [self.connect_sliders(i) for i in range(10)]


        self.audible = True                               # FLag for audio processing

        self.timer1 = QTimer(self)
        self.timer1.timeout.connect(self.update)
        self.timer2 = QTimer(self)
        self.timer2.timeout.connect(self.update)
        self.timers = [self.timer1, self.timer2]
        
        self.playing = False
        self.mult_mode = "Rectangle"
        self.update_rate = 4100                         # Update rate for signals in cine mode
        self.play_buttons = [self.PlayButton, self.PlayButton2]

        self.check_mode(self.ModeCombobox.currentText())        
        self.link_plots()










    def open_dialog_box(self):
        # Get the path of the file of formats (mp3 / wav / csv)
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", "MP3 Files (*.mp3);;WAV Files (*.wav);;CSV Files (*.csv)", options=options)

        if path:
            # load_data returns Samples (Data Points), Time (Time Domain Points)
            self.input[0], self.input[1] = self.load_data(path)
            # Two curves initialization for previewing the signal in cine mode by setting data to those curves
            self.curves = [self.plot_widgets[0].plot(pen = pg.mkPen('#d12020')), self.plot_widgets[1].plot(pen = pg.mkPen('#d12020'))]
            # Current Pointers for the update function to keep on track where we are in the signal
            self.ptrs = [0, 0]
            # Computation and plotting of FFT, FFT and FFTFreq will be assigned to self.input[2] , self.input[3] respectively
            self.compute_fft()
            # Initiate the output by copying the same data of the input to it to edit on it soon
            self.output = self.input.copy()
            # Computation and plotting of Spectrogram
            self.compute_spectrogram()
            # Sliders Data Ranges of FFT gets set to each slider based on the entered data
            self.sliders_adjustment()
            # Plot Multiply Mode in its plot (Rectangle, hamming, hanning or gaussian)
            self.check_mult_mode(self.MultModeCombobox.currentText())









    def load_data(self, path):
        if path.endswith('.csv'):
            self.input_file = pd.read_csv(path)
            samples = np.array(pd.to_numeric(self.input_file.iloc[:, 1].values, downcast="float"))
            time = np.array(pd.to_numeric(self.input_file.iloc[:, 0].values, downcast="float"))
            self.sampling_rate = 1 / (time[1] - time[0])
            # Lesser update_rate for ecg data
            self.update_rate = 10
            # No Trials will be there to output audio from this data
            self.audible = False
        else:
            self.input_file = AudioSegment.from_file(file = path, format = 'wav') if path.endswith('.wav') else AudioSegment.from_mp3(path)
            samples = np.array(self.input_file.get_array_of_samples())
            self.sampling_rate = self.input_file.frame_rate
            time = np.arange(0, len(samples)) / float(self.sampling_rate)

        return samples, time
    








    def connect_sliders(self, index):
        # When the Slider gets released after editing it, it preforms self.eq_slider which preform the equalizer effect
        self.sliders[index].sliderReleased.connect(lambda: self.eq_slider(index))

        









    def ouput_audio_reconstruction(self):
        # Returns the audio reconstructed from the samples data points after editing present in self.output[0]
        return AudioSegment(data=np.int16(self.output[0]).tobytes(),
                            frame_rate=self.input_file.frame_rate,
                            sample_width=self.input_file.sample_width,
                            channels=self.input_file.channels)









    def save_handler(self):
        reconstructed_audio = self.ouput_audio_reconstruction()
        reconstructed_audio.export('./outEqualizer.wav', format="wav")


    






    def check_mode(self, combobox_text):
        # reset the visiblity of all the sliders
        self.reset_sliders()
        # Hide the required sliders and just leave the enabled sliders for each mode
        for i in range (self.enabled_sliders[combobox_text], 10):
            self.slider_visiblity(False, i)







    
    def reset_sliders(self):
        for i in range (0, 10):
            self.slider_visiblity(True, i)







    

    def slider_visiblity(self, visible_bool, i):
        # set sliders and its correspondants (line edits, labels) visibilty to either true or false
        self.sliders[i].setVisible(visible_bool)
        self.sliders_lineEdits[i].setVisible(visible_bool)
        self.sliders_lables[i].setVisible(visible_bool)

        






    def toggle_spect(self):
        # Hide and show the two plots of the spectrogram which are present in self.plot_widgets[4] and self.plot_widgets[5]
        [plot.setVisible(self.SpectCheckbox.isChecked()) for plot in self.plot_widgets[4:6]]








    
    def rewind_handler(self):
        # reassign the pointers to 0 to start from the begining of the signal
        self.ptrs = [0, 0]
        self.pause_audio()
        self.play_audio(self.channel)








    def reset_handler(self):
        # Stop and reset playback
        if self.playing and self.audible:
            self.pause_audio()

        # Reset all data
        self.input, self.output = [np.array([]) for _ in range(4)], [np.array([]) for _ in range(4)]

        self.data_ranges = []

        self.ptrs = [0, 0]
        self.curves = [None, None]

        # Reset UI elements
        for i in range(10):
            self.sliders[i].setValue(0)
            self.sliders_lables[i].setText('-')
            self.sliders_lineEdits[i].setText('0')

        self.SpectCheckbox.setChecked(False)
        self.ModeCombobox.setCurrentIndex(0)
        self.MultModeCombobox.setCurrentIndex(0)

        # Clear all plots
        [plot.clear() for plot in self.plot_widgets]

        # Reset the input file
        self.input_file = None
    









    def link_plots(self):
        # link the time plots together, the two fft plots together and the two spectrogram plots together
        for i in range(0, 6, 2):
            self.plot_widgets[i].setXLink(self.plot_widgets[i + 1])
            self.plot_widgets[i].setYLink(self.plot_widgets[i + 1])








    def play_signal(self, channel):
        if self.sampling_rate:
            # channel refers to the input(1) or the ouput(2)
            self.channel = channel 

            if self.playing == False:
                self.play_audio(channel)
                    
            else:
                self.pause_audio()





    
    
    def play_audio(self, channel):
        self.playing = True
        if self.audible:
            if channel == 1:
                # if channel = 1 just play the input audio file and fade the color of the output plot
                curves_order = [0, 1]
                self.audio = self.input_file
                    
            elif channel == 2:
                # if channel = 2 then reconstruct the audio from the output and fade the color of the input plot
                curves_order = [1, 0]
                self.audio = self.ouput_audio_reconstruction()

            # for audio syncing with the plot    
            self.den = len(self.input[0]) / len(self.input_file)
            self.playback = _play_with_simpleaudio(self.audio[ceil(self.ptrs[0] / self.den):])
            # fading of unplayed audio plot
            self.curves[curves_order[0]].setPen(pg.mkPen('#d12020'))
            self.curves[curves_order[1]].setPen(pg.mkPen('#201253'))
        
        for i in range(2):
            self.play_buttons[i].setText('Pause')
            self.timers[i].start(80)
                    





                

    def pause_audio(self):
        self.playing = False
        for i in range(2):
            self.play_buttons[i].setText('Play')
            self.timers[i].stop()
        if self.audible:
            self.playback.stop()







    def update(self):
        for i in range(2):
            if self.sampling_rate:
                if self.ptrs[i] <= len(self.input[1]):
                    # curve 1 is the input curve and curve 2 is the output curve so first we set the data to either input data or output data
                    x_data, y_data = (self.input[1], self.input[0]) if i == 0 else (self.output[1], self.output[0])
                    x_max = x_data[self.ptrs[i]]
                    x_min = max(0, x_max - 10)
                    # setting the curve data to the last point of current pointer
                    self.curves[i].setData(x_data[:self.ptrs[i] + 1], y_data[:self.ptrs[i] + 1])
                    # setting the x range
                    self.plot_widgets[i].setXRange(x_min, x_max + 10, padding=0)
                    # updating the pointer by 4100 for mp3/wav signals and 10 for csv signals
                    self.ptrs[i] += self.update_rate
                else:
                    self.rewind_handler()







    def zoom_in(self):
        if self.sampling_rate:
            self.zoom_helper(0.8)






    def zoom_out(self):
        if self.sampling_rate:
            self.zoom_helper(1.2)





            


    def zoom_helper(self, factor):
        x_range = self.plot_widgets[0].plotItem.getViewBox().state['viewRange'][0]
        y_range = self.plot_widgets[0].plotItem.getViewBox().state['viewRange'][1]
        new_x_range = [x_range[0] * factor, x_range[1] * factor]
        new_y_range = [y_range[0] * factor, y_range[1] * factor]
        self.plot_widgets[0].setXRange(*new_x_range)
        self.plot_widgets[0].setYRange(*new_y_range)
    




    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------







    # DISPLAY & MODES
    # ---------------

    def check_mult_mode(self, combobox_text):
            self.MultModePlot.clear()
            window_size = len(self.input[3])
            visible_bool = False
            if combobox_text == 'Rectangle':
                window = np.ones(window_size)
            elif combobox_text == 'Hamming':
                window = np.hamming(window_size)
            elif combobox_text == 'Hanning':
                window = np.hanning(window_size)
            elif combobox_text == 'Gaussian':
                window = np.exp(-0.5 * ((np.arange(window_size) - (window_size) / 2) / ((self.StdSlider.value()) ** 2) + 0.0000000001))
                visible_bool = True
            self.MultModePlot.plot(window)
            self.StdSlider.setVisible(visible_bool)








    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------





    # SLIDERS & EQUALIZER FUNCTIONS
    # -----------------------------

    

    def reset_all_channels(self):
        # reset the position of all the sliders and their corresponding labels to 0 as well
        for i in range(10):
            self.sliders[i].setValue(0)
            self.sliders_lineEdits[i].setText("0")

        # Revert the output data to the same input as we resetting all the changes that was done
        self.output = [np.copy(self.input[i]) for i in range(len(self.input))]

        # Update the FFT and Spectrogram for the output
        self.compute_fft(STATE='both')
        self.compute_spectrogram(STATE='both')








    def update_output_plot(self):
        # Clear the existing plot
        self.plot_widgets[1].clear()

        x_data, y_data = (self.output[1], self.output[0])
        x_max = x_data[self.ptrs[1]]
        x_min = max(0, x_max - 10)
        self.plot_widgets[1].plot(x_data[:self.ptrs[1] + 1], y_data[:self.ptrs[1] + 1])
        self.plot_widgets[1].setXRange(x_min, x_max + 10, padding=0)
    

    





    def eq_slider(self, index):
        # Converting Value of the slider
        gain = 10**(self.sliders[index].value() / 20)

        self.data_modified_fft = self.multiply_fft( 
            self.output[2], self.data_ranges[index], gain, self.StdSlider.value(), self.MultModeCombobox.currentText())
        # Changing the slider line edit to the value of the corresponding slider
        self.sliders_lineEdits[index].setText(str(self.sliders[index].value()))

        # Inverse fourier transform to get the signal in the time domain so we can hear the changes that was done
        self.modified_output = np.real(np.fft.ifft(self.data_modified_fft))

        # Put the output of the fft in self.output[0] and the ouput of the new fft in self.ouput[2]
        self.output[0] = np.copy(self.modified_output)
        self.output[2] = np.copy(self.data_modified_fft)

        # Updating the output fft and spectrogram plots
        self.compute_fft(STATE="solo")
        self.compute_spectrogram(STATE="solo")
        






    def multiply_fft(self, data, indices, gain, std_gaussian, mult_mode_window):
        modified_data = data.copy()
        window_size = len(modified_data[indices])

        if mult_mode_window == 'Rectangle':
            gain_factor = gain
        elif mult_mode_window == 'Hamming':
            gain_factor = np.hamming(window_size) * gain
        elif mult_mode_window == 'Hanning':
            gain_factor = np.hanning(window_size) * gain
        elif mult_mode_window == 'Gaussian':
            gain_factor = np.exp(-0.5 * ((np.arange(window_size) - (window_size) / 2) / std_gaussian) ** 2) * gain

        if gain_factor <= 0.04:
            gain_factor = 0


        # Multiplying the specific indices of the output data that corresponds to the slider moved by the gain 
        modified_data[indices] = self.input[2][indices] * gain_factor
        
        print(gain_factor)

        return modified_data









    def sliders_adjustment(self):
        # In unifrom mode we divide the whole band frequencies by 10 and assign each part to a slider
        self.band_length = self.input[3][np.argmax(self.input[3])] // 10
        if self.ModeCombobox.currentText() == "Uniform Range Mode":
            for n in range(10):
                low_band = self.band_length * n
                high_band = self.band_length * (n + 1)
                self.set_indices(n, [low_band, high_band], str(low_band))
        else:
        # In other modes we just get the ranges from the json file and assign these ranges to each visible slider
            for i, mode in enumerate(['Animal Sounds Mode', 'Musical Instruments Mode', 'ECG Abnormalities Mode']):
                if self.ModeCombobox.currentText() == mode:
                    for n, (title, ranges) in enumerate(self.modes_freq_ranges[i].items()):
                        self.set_indices(n, ranges, title)
                    break







    def set_indices(self, n, ranges, title):
        indices = np.where((np.abs(self.input[3]) >= ranges[0]) & (np.abs(self.input[3]) <= ranges[1]))[0]
        self.data_ranges.append(indices)
        self.sliders_lables[n].setText(title)








                



    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------





    # SPECTOGRAM & SMOOTH FEATURES
    # ----------------------------




    def compute_fft(self, STATE = 'both'):
        if STATE == 'both':
            # we plot the same input plot in both input and output, for initial settings and after resetting
            self.input[2] = np.fft.fft(self.input[0])
            self.input[3] = np.fft.fftfreq(len(self.input[2]), d=self.input[1][1] - self.input[1][0])

            for plot in self.plot_widgets[2:4]:
                plot.clear()
                plot.plot(np.abs(self.input[3]), np.abs(self.input[2]))
            
        else:
            # just updating the ouput FFT plot
            self.plot_widgets[3].clear()
            self.plot_widgets[3].plot(np.abs(self.output[3]), np.abs(self.output[2]))








    def compute_spectrogram(self, STATE = 'both'):
        if STATE == 'both':
            # we plot the same input spectrogram in both input and output, for initial settings and after resetting
            for plot in self.plot_widgets[4:6]:
                plot.clear()
                self.plot_spectrogram(plot, self.input[0], self.sampling_rate)
        else:
            # just updating the spectrogram of the ouput
            self.plot_widgets[5].clear()
            self.plot_spectrogram(self.plot_widgets[5], self.output[0], self.sampling_rate)




    def plot_spectrogram(self, plot, data, sample_rate):
        f, t, sxx = spectrogram(data, fs=sample_rate)
        img = pg.ImageItem()
        img.setImage(np.log(sxx + 1))
        plot.addItem(img)
        plot.setLabel('left', 'Frequency', units='Hz')
        plot.setLabel('bottom', 'Time', units='s')
        colormap = pg.colormap.get('inferno')
        img.setColorMap(colormap)













# Main Code
#----------
def main():
    app = QApplication([])
    window = Equalizer()
    app.exec_()

if __name__ == "__main__":
    main()