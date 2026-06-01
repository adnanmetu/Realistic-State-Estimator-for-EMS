Realistic Power System State Estimator

Group Members:
1. RANA MUHAMMAD ADNAN KHALIL
2. EREN DENİZ EMİNAĞAOĞLU

Course: EE574
Instructor: MURAT GÖL

Objective:
Understand the working of Power System State Estimator using WLS and observe the behavior of estimator with different types of data set, by developing python code.

Scope:
Parsing of IEEE format network data file.
Parsing of measurements file.
Determination of:
i) Bus admittance matrix (ybus).
ii) Measurement Vector
iii) Jacobian matrix
iv) Gain matrix
v) State estimator using WLS
vi) Observability analysis
vii) Bad data detection and identification

Pros and Cons of Project
Pros:
This code can handle 8 type of measurements in following sequence blocks:
1. Voltage magnitude
2. Voltage angle
3. Active Power Injection
4. Reactive Power Injection
5. Active Power Flow
6. Reactive Power Flow
7. Current Magnitude
8. Current Angle
This code can handle missing measurements
This code can handle noisy measurements
Self explanatory output
This code can filter measurements which are not in the network.
The code is not dependent on fixed files, it asks user to enter file name of ieee cdf and measurement file respectively.

Cons:
1. It assumes the cdf file should be correct.
2. The program assumes the measurements are synchronized and there is no time skew.

AI Reference:
The code skeleton is created with the help of AI tool Claude.

How to Run The Project:
First install the PyCharm Community Edition.
This project is build with open source PyCharm Community Edition, the detail of version is given below
PyCharm 2025.2.6.1 (Community Edition)
Build #PC-252.28539.58, built on April 21, 2026
Source revision: af37552c2b565
Runtime version: 21.0.9+1-b1038.78 amd64 (JCEF 122.1.9)
VM: OpenJDK 64-Bit Server VM by JetBrains s.r.o.
Toolkit: sun.awt.windows.WToolkit
Windows 10.0
GC: G1 Young Generation, G1 Concurrent GC, G1 Old Generation
Memory: 1500M
Cores: 4
Registry:
  ide.experimental.ui=true
  llm.show.ai.promotion.window.on.start=false

Project Dependencies:
After installing PyCharm Community Edition, open the project and install the following modules.
Go go settings then interpretor click '+' sign and add the following modules and install.
numpy
scipy
tabulate
matplotlib
networkx (for future use to create network diagram using python code)
pip
pyparsing

Run the project by pressing shift + F10 key.
If there is any other dependent module then the error message will appear with that name and install that module as well.
On successful run, the output window ask the cdf file name, please enter the correct file name with extension and the file should be in project folder.
After that the output window will ask the name of measurement file, please enter the correct file name and the file should be in project folder.

If the network is observable and gain matrix is non-singular then ac-wls algorithm will run smoothly.
Otherwise ac-wls will stop but the program will try to run bad data analysis.
Read the output window carefully to see the step by step output.
The output window is self explanatory.
Close the figures (plots) to run the project again on different type of data set.

Notes:
1. If the absolute value of the first measurement in block 2 is greater than 0.1, it will be considered as power injection measurement block and not as voltage angle measurement..
2. The parsing of measurement file is very important and tricky, all calculations depend on this parsing.
3. The program assumes the measurements are synchronized and there is no time skew.
4. The program is tested with 6 different types of test cases and it works fine.

Graph
Figure 1 is residuals graph.
Figure 2 shows largest normalized residuals.

Future
An Android version of this state estimator is in beta, so that users can get states, observablity analysis and bad data detection in their pocket.
Beta version is available on google play store.
Further features will be added in near future.

Complete files related to project are available on github.
https://github.com/adnanmetu/Realistic-State-Estimator-for-EMS
