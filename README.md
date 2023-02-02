# EdgeRL
Edge Reinforcement Learning for Real-World Control Applications

[![Coffee machine vs. machine learning: who is quicker?](https://img.youtube.com/vi/hQ49Mc6LV78/0.jpg)](https://www.youtube.com/watch?v=hQ49Mc6LV78)

The toolchain featured in this repository is to be used for reinforcement learning (RL) applications within real-world experiments. 
State transition samples that have been measured on an available plant system are sent from the test bench computer to an edge computing workstation via TCP/IP.
On the workstation, the applied training algorithm processes the acquired samples in order to determine a new policy for controlling the plant system.
The weighting parameters describing the new policy are then sent back to the test bench and can be applied within the operating rapid-control prototyping interface.
This enables asynchronous RL for real-world applications without the necessity to shut down the plant system during training downtimes.

Rapid-control prototyping interface: dSPACE MicroLabBox

Software
  - Test bench: 
    - MATLAB / Simulink
    - dSPACE ControlDesk
    - Python
  - Edge Workstation (Linux):
    - Python
    - that's it
