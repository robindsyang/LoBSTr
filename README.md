# Paper
https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.142631

# Abstract
With the popularization of games and VR/AR devices, there is a growing need for capturing human motion with a sparse set of tracking data.
In this paper, we introduce a deep neural network (DNN) based method for real-time prediction of the lower-body pose only from the tracking signals of the upper-body joints.
Specifically, our Gated Recurrent Unit (GRU)-based recurrent architecture predicts the lower-body pose and feet contact states from a past sequence of tracking signals of the head, hands, and pelvis.
A major feature of our method is that the input signal is represented by the velocity of tracking signals.
We show that the velocity representation better models the correlation between the upper-body and lower-body motions and increases the robustness against the diverse scales and proportions of the user body than position-orientation representations.
In addition, to remove foot-skating and floating artifacts, our network predicts feet contact state, which is used to post-process the lower-body pose with inverse kinematics to preserve the contact.
Our network is lightweight so as to run in real-time applications. 
We show the effectiveness of our method through several quantitative evaluations against other architectures and input representations with respect to wild tracking data obtained from commercial VR devices.

# Bibtex
@article {10.1111:cgf.142631,
 journal = {Computer Graphics Forum},
 title = {{LoBSTr: Real-time Lower-body Pose Prediction from Sparse Upper-body Tracking Signals}},
 author = {Yang, Dongseok and Kim, Doyeon and Lee, Sung-Hee},
 year = {2021},
 publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
 ISSN = {1467-8659},
 DOI = {10.1111/cgf.142631}
}
