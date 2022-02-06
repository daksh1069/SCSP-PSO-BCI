
# SCSP-PSO-BCI

Brain-computer interfaces (BCls) and translate acquired brain signals into commands and
communicate to an output device to execute the intended action. Motor Imagery BCI (MI-BCI)
provides a non-muscular method to communicate for those who are suffering from some neurological
disease/ disorder. Common Spatial pattern (CSP) is well-known feature extraction method used in
literature for MI-BCl. But CSP is highly sensitive to noise and non-stationary EEG signals. In
literature, Stationary CSP (SCSP) is suggested to overcome these problems. Also. CSP computes
covariance matrices. whose space complexity is 0(n^2). In general, the number of available samples
for MI-BCI is small. Hence CSP and its variants suffer from the problem of overfitting. Also. it will
be difficult to find the electrodes which help in distinguishing motor imagery tasks. 

# Objective -
To propose MI-BCI decision model with the selection of minimal subset of relevant
electrodes to avoid over-fitting in the MI-BCI using PSO and SCSP.

# Outcome - 
Outcome:
• The costs of the proposed MI-BCI will be less with the use of the minimal relevant subset of
electrodes.
• The computational complexity of the decision model will reduce.
• The propose study will help us to determine brain region participating to carry out motor
imagery tasks.

# More Information
We will use evolutionary algorithm to find optimal relevant electrodes and spectral bands to
distinguish motor imagery tasks using EEG signal. SCSP will be used to represent EEG signals. We
will investigate both linear and non-linear classifiers to build decision model with publicly available
motor imagery tasks dataset.!

# Run
Run the program using - PSO on BCI .ipynb 
Necesaary, Important Modules are in the File ImportantModulesforPSO.py { Newbies may go through this thouroughly to get a proper understanding of how things work }

![Proposed Model](https://user-images.githubusercontent.com/42321349/152700647-0ba0fd7e-9665-4de7-8c88-caaff92a6f9f.png)

   Proposed Model Architecture - 
       
       
DataSet Used - BCI Competition iii, iv-a ==> https://www.bbci.de/competition/iii/#data_set_iva

Thank you ! <3

CSP, Stationary CSP ( SCSP ) is used.

References - https://github.com/wmvanvliet/neuroscience_tutorials/blob/master/eeg-bci/3.%20Imagined%20movement.ipynb https://github.com/breuderink/eegtools

![BCI Fig1](https://user-images.githubusercontent.com/42321349/152700702-1f14e2d6-cbe5-45a7-b9ec-9f8fc5b01a5b.png)
