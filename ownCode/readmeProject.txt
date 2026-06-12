Realistic Power System State Estimator

Group Members:
1. RANA MUHAMMAD ADNAN KHALIL
2. EREN DENİZ EMİNAĞAOĞLU

Course: EE574
Instructor: MURAT GÖL

Objective:
Understand the working of Power System State Estimator using WLS and observe the behaviour of estimator with different types of dataset, by developing python code.

Scope:
Parsing of IEEE format network data file.
Parsing of measurements file.
Determination of:
i) Bus admittance matrix (Ybus).
ii) Measurement Vector
iii) Jacobian matrix
iv) Gain matrix
v) State estimator using WLS
vi) Observability analysis
vii) Bad data detection and identification

Pros and cons of the project
Pros:
a) The code can handle 8 type of measurements in following sequence blocks:
1. Voltage magnitude
2. Voltage angle
3. Active Power Injection
4. Reactive Power Injection
5. Active Power Flow
6. Reactive Power Flow
7. Current Magnitude
8. Current Angle
b) The code can handle missing measurements.
c) The code can handle noisy measurements.
d) The code has self-explanatory output.
e) The code can filter measurements which are not in the network.
f) The code is not dependent on fixed files, it asks user to enter names of the IEEE CDF and the measurement file respectively.

Cons:
a)The code assumes that the cdf file is correct.
b)The code assumes that the measurements are synchronized and there is no time skew.

Project Learning Outcome:
a) Gain matrix should be non-singular for network observability.
b) If the system is unobservable, the state estimation and also bad data detection cannot be done.
c) If the number of independent measurements are less then number of states then the system is unobservable.
d) Number of independent measurements should be greater than number of states.
e) If system is observable, then largest normalized residual test is performed and bad data identification is done.
f) If there is any bad data, then it should be removed is the measurement data is redundant, otherwise it cannot be removed and declared as critical data.
g) Bad data removal is necessary because it can effect other residuals.
h) After removing bad data, the measurement vector is updated and state estimation is done again.
i) If there is still bad data then it should also be removed and state estimation is repeated again.
j) It can be seen from the graphs tha initially the residuals are quite large but after removing the amplitude of residuals decreased significantly.
k) It is also seen that higher the number of degree of freedom (higher the redundant number of measurements), the quickly estimator converges and smaller the residuals.

How to Run The Project:
1)Install the PyCharm Community Edition.
This project is built on open source PyCharm Community Edition, specifically version 2025.2.6.1.

2)After installing PyCharm Community Edition, install the following modules that the project depends on:
Go to settings, click interpreter, click '+' sign, and add the following modules and install:
numpy
scipy
tabulate
matplotlib
re

3)Open the project (double-click), and run the project by pressing Shift + F10.
On a successful run, the output window will ask for the CDF file name.

4)Enter the correct CDF file name with its extension. The file should be in the same folder as the .py file.
After that, the output window will ask for the measurement file name.

5)Enter the correct measurement file name with its extension. The file should be in the same folder as the .py file.
If the network is observable and gain matrix is non-singular then ac-wls algorithm will run smoothly and bad data detection and identification will follow and if there is any bad data it will be removed and state estimation will be run again until there is no more bad data or system becomes unobservable.
Otherwise, ac-wls will stop, and program will show observable islands and unobservable branches.
Read the output window carefully to see the step-by-step output.
The output window is self-explanatory.
Close the figures (plots) to run the project again with another dataset.

Notes:
1. If the absolute value of the first measurement in block 2 is greater than 0.35, it is considered as the power injection measurement block (not as voltage angle measurement).
2. The parsing of measurement file is very important and tricky, all calculations depend on this parsing.
3. The program assumes the measurements are synchronized and there is no time skew.
4. The program is tested with 7 different types of test cases and it works fine.

Graph:
a)If there is bad data then residuals with bad data plotted initially
b)Final residual plot is also shown if the system is observable

AI Reference:
The code skeleton is created with the help of AI tool Claude.

Future
An Android version of this state estimator is in beta, so that users can get states, observability analysis and bad data detection in their pocket. Beta version is available on Play Store. More features will be added in the near future.

Complete files related to project are available on GitHub:
https://github.com/adnanmetu/Realistic-State-Estimator-for-EMS
