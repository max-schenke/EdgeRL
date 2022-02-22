"""
File:        06_FetchingDataWhileCapturing.py

Description: This sample demonstrates how fetch data while a capture is in process using the 
             dSPACE XIL API server.

             This program uses the turn lamp simulation application from your demo directory
             MAPort\Common\SimulationApplications\<platform>. 

             Adapt lines 61-71 of this file according to your dSPACE platform.
 
             Make sure that the dSPACE platform that is used for this demo
             is registered with ControlDesk, AutomationDesk or the Platform Management API.

             Also note in the call to the method Configure of the MAPort, the second
             parameter is set to 'false'. This means that the specified simulation application
             will not be downloaded unless there is no application loaded on the platform. 
             If the specified application is already running, no further action will be taken. 
             If any other application is running on the platform, an exception will be thrown.

Tip/Remarks: Objects of some XIL API types (e.g., MAPort, Capture) must be disposed at the end
             of the function. We strongly recommend to use exception handling for this purpose
             to make sure that Dispose is called even in the case of an error.

Version:     4.0

Date:        May 2021

             dSPACE GmbH shall not be liable for errors contained herein or
             direct, indirect, special, incidental, or consequential damages
             in connection with the furnishing, performance, or use of this
             file.
             Brand names or product names are trademarks or registered
             trademarks of their respective companies or organizations.

Copyright 2021, dSPACE GmbH. All rights reserved.
"""

# Provide print function with parameter end
from __future__ import print_function

import select

import clr
import os, sys, time

# for TCP/IP connection
import socket
import numpy as np
import struct
from interface_functions import NeuralNetworkDecoder
import threading

TCP_IP = '131.234.124.79'  # host ip Address or "local host"
TCP_PORT_DATA = 1030  #
TCP_PORT_WEIGHTS = 1031  #
BUFFER_SIZE = 1008 # = floor(1024 // (measurement_size * 4)) * (measurement_size * 4) # <- 4 = nb of bits per float
neurons_per_layer = 127
b = bytes()  # byte container for dSPACE XIL API lists
# create socket and connect
data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # TCP
weights_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # TCP

clr.AddReference("System.Collections")
from System import Array
from System.Collections.Generic import Dictionary

# Load ASAM assemblies from the global assembly cache (GAC)
clr.AddReference(
    "ASAM.XIL.Implementation.TestbenchFactory, Version=2.1.0.0, Culture=neutral, PublicKeyToken=fc9d65855b27d387")
clr.AddReference("ASAM.XIL.Interfaces, Version=2.1.0.0, Culture=neutral, PublicKeyToken=bf471dff114ae984")

# Import XIL API .NET classes from the .NET assemblies
from ASAM.XIL.Implementation.TestbenchFactory.Testbench import TestbenchFactory
from ASAM.XIL.Interfaces.Testbench.Common.Error import TestbenchPortException
from ASAM.XIL.Interfaces.Testbench.Common.Capturing.Enum import CaptureState
from ASAM.XIL.Interfaces.Testbench.MAPort.Enum import MAPortState

# Import DemoHelpers for Python 3.9
from DemoHelpers import *

# The following lines must be adapted to the dSPACE platform used
# ------------------------------------------------------------------------------------------------
# Set IsMPApplication to true if you are using a multiprocessor platform
IsMPSystem = False
# Use an MAPort configuration file that is suitable for your platform and simulation application
# See the folder Common\PortConfigurations for some predefined configuration files
MAPortConfigFile = r"MAPortConfigDS1202.xml"

# Set the name of the task here (specified in the application's TRC file)
# Note: the default task name is "HostService" for PHS bus systems, "Periodic Task 1" for VEOS systems
Task = "HostService"
# ------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------
# For multiprocessor platforms different tasknames and variable names have to be used.
# Some variables are part of the subappliaction "masterAppl", some belong to the 
# subapplication "slaveAppl"
# ------------------------------------------------------------------------------------------------
if IsMPSystem:
    masterTaskPrefix = "masterAppl/"
    slaveTaskPrefix = "slaveappl/"
    masterVariablesPrefix = "masterappl/Model Root/master/CentralLightEcu/"
    slaveVariablesPrefix = "slaveappl/Model Root/slave/FrontRearLightEcu/"

else:
    masterTaskPrefix = ""
    slaveTaskPrefix = ""
    masterVariablesPrefix = "ds1202()://Model Root/"
    slaveVariablesPrefix = "ds1202()://Model Root/"

masterTask = masterTaskPrefix + Task
slaveTask = slaveTaskPrefix + Task

# --------------------------------------------------------------------------
# Set the working directory for this demo script
# --------------------------------------------------------------------------
WorkingDir = os.path.dirname(sys.argv[0])
if not os.path.isdir(WorkingDir):
    WorkingDir = os.getcwd()
if not os.path.isdir(WorkingDir):
    os.mkdir(WorkingDir)

MAPortConfigFile = os.path.join(WorkingDir, MAPortConfigFile)

if __name__ == "__main__":

    DemoCapture = None
    DemoMAPort = None

    try:

        # --------------------------------------------------------------------------
        # Create a TestbenchFactory object; the TestbenchFactory is needed to 
        # create the vendor-specific Testbench
        # --------------------------------------------------------------------------
        MyTestbenchFactory = TestbenchFactory()

        # --------------------------------------------------------------------------
        # Create a dSPACE Testbench object; the Testbench object is the central object to access
        # factory objects for the creation of all kinds of Testbench-specific objects
        # --------------------------------------------------------------------------
        MyTestbench = MyTestbenchFactory.CreateVendorSpecificTestbench("dSPACE GmbH", "XIL API", "2021-A")

        # --------------------------------------------------------------------------
        # We need an MAPortFactory to create an MAPort, a ValueFactory to create ValueContainer 
        # objects and also a CapturingFactory to create a CaptureResultMemoryWriter
        # The WatcherFactory is used to create a DurationWatcher and a ConditionWatcher,
        # the DurationFactory provides TimeSpanDuration objects.
        # --------------------------------------------------------------------------
        MyMAPortFactory = MyTestbench.MAPortFactory
        MyValueFactory = MyTestbench.ValueFactory
        MyCapturingFactory = MyTestbench.CapturingFactory
        MyWatcherFactory = MyTestbench.WatcherFactory
        MyDurationFactory = MyTestbench.DurationFactory

        # --------------------------------------------------------------------------
        # Create and configure an MAPort object and start the simulation
        # --------------------------------------------------------------------------
        print("Creating MAPort instance...")
        # Create an MAPort object using the MAPortFactory
        DemoMAPort = MyMAPortFactory.CreateMAPort("DemoMAPort")
        print("...done.\n")
        # Load the MAPort configuration
        print("Configuring MAPort...")
        DemoMAPortConfig = DemoMAPort.LoadConfiguration(MAPortConfigFile)
        # Apply the MAPort configuration
        DemoMAPort.Configure(DemoMAPortConfig, False)
        print("...done.\n")
        if DemoMAPort.State != MAPortState.eSIMULATION_RUNNING:
            # Start the simulation
            print("Starting simulation...")
            DemoMAPort.StartSimulation()
            print("...done.\n")

        # ----------------------------------------------------------------------
        # Define the variables to be captured
        # ----------------------------------------------------------------------
        # housekeeping
        manualCaptureTrigger = masterVariablesPrefix + "Manual_Trigger/Value"
        learningRate = masterVariablesPrefix + "Learning_Rate/Value"

        # measurement
        stateOmega = masterVariablesPrefix + "Controller/Featurizer/omega_me_scaling/Out1"
        stateCurrentId = masterVariablesPrefix + "Controller/Featurizer/current_scaling/Out1[0]"
        stateCurrentIq = masterVariablesPrefix + "Controller/Featurizer/current_scaling/Out1[1]"
        stateVoltageUd = masterVariablesPrefix + "Controller/Featurizer/u_dq_scaling/Out1[0]"
        stateVoltageUq = masterVariablesPrefix + "Controller/Featurizer/u_dq_scaling/Out1[1]"
        statePositionScaledCos = masterVariablesPrefix + "Controller/Featurizer/position_scaling/Out1[0]"
        statePositionScaledSin = masterVariablesPrefix + "Controller/Featurizer/position_scaling/Out1[1]"
        stateCurrentStator = masterVariablesPrefix + "Controller/Featurizer/stator_current_scaling/i_s_norm"
        stateTorqueRef = masterVariablesPrefix + "Controller/Featurizer/T_ref_scaling/Out1"
        action = masterVariablesPrefix + "Controller/Action_Processing/EpsilonSafetyActionSelection/a_policy"
        reward = masterVariablesPrefix + "Reward function/reward"
        doneFlag = masterVariablesPrefix + "Reward function/done flag"

        # --------------------------------------------------------------------------
        # Create and initialize Capture object
        # --------------------------------------------------------------------------
        print("Creating Capture...")
        DemoCapture = DemoMAPort.CreateCapture(masterTask)
        # create a list containing the names of the variables to be captured
        DemoVariablesList = [learningRate, stateOmega, stateCurrentId, stateCurrentIq, stateVoltageUd, stateVoltageUq]
        DemoVariablesList.extend([statePositionScaledCos, statePositionScaledSin, stateCurrentStator, stateTorqueRef])
        DemoVariablesList.extend([action, reward, doneFlag])
        # The Python list hast to be converted to an .net Array
        DemoCapture.Variables = Array[str](DemoVariablesList)

        # In this demo a higher downsampling is used to reduce the number of captured data samples
        # Only every 20th measured sample is captured
        DemoCapture.Downsampling = 1
        print("...done.\n")

        # --------------------------------------------------------------------------
        # Create one ConditionWatcher and one DurationWatcher and set start- and stop triggers
        # --------------------------------------------------------------------------
        # Create Defines for ConditionWatchers
        DemoDefines = Dictionary[str, str]()
        DemoDefines.Add('CaptureTrigger', manualCaptureTrigger)

        # Negative Delay: Start Capturing 0.1s before StartTriggerCondition is met
        StartDelay = MyDurationFactory.CreateTimeSpanDuration(0.0)
        print("Creating ConditionWatcher...")
        DemoStartWatcher = MyWatcherFactory.CreateConditionWatcher("posedge(CaptureTrigger,0.5)", DemoDefines)
        DemoCapture.SetStartTrigger(DemoStartWatcher, StartDelay)
        print("...done.\n")

        print("Creating DurationWatcher...")
        StopDelay = MyDurationFactory.CreateTimeSpanDuration(0)
        captureTime = 1
        DemoStopWatcher = MyWatcherFactory.CreateConditionWatcher("negedge(CaptureTrigger,0.5)", DemoDefines)
        #DemoStopWatcher = MyWatcherFactory.CreateDurationWatcherByTimeSpan(captureTime)
        DemoCapture.SetStopTrigger(DemoStopWatcher, StopDelay)
        print("...done.\n")

        # --------------------------------------------------------------------------
        # Create CaptureResultMemoryWriter object
        # --------------------------------------------------------------------------
        print("Creating CaptureResultMemoryWriter...")
        DemoCaptureWriter = MyCapturingFactory.CreateCaptureResultMemoryWriter()
        print("...done.\n")

        # --------------------------------------------------------------------------
        # Declare a CaptureResult 
        # --------------------------------------------------------------------------
        DemoCaptureResult = MyCapturingFactory.CreateCaptureResult()

        # --------------------------------------------------------------------------
        # Capturing process
        # --------------------------------------------------------------------------
        print("\nStart capturing.")
        DemoCapture.Start(DemoCaptureWriter)

        # establish socket for TCP/IP
        data_socket.connect((TCP_IP, TCP_PORT_DATA))
        weights_socket.connect((TCP_IP, TCP_PORT_WEIGHTS))

        time.sleep(2.0)

        print("Receiving architecture from remote RL server")
        binary_architecture = weights_socket.recv(1024)
        architecture = np.frombuffer(binary_architecture, dtype=np.float32)
        nn_decoder = NeuralNetworkDecoder(architecture)
        nn_parameter_paths = []
        for _i in range(nn_decoder.nb_dense_layers):
            nn_parameter_paths.append(masterVariablesPrefix + "Controller/w" + str(_i) + "/Value")

        print("Receiving parameters from remote RL server")
        weights = nn_decoder.recv_first_network(weights_socket)

        time.sleep(2.0)

        print("Receiving initialized learning_rate")
        learning_rate = np.frombuffer(weights_socket.recv(4), dtype=np.float32)[0]

        print("Initializing network on MicroLabBox")
        for _i in range(nn_decoder.nb_dense_layers):
            w = np.transpose(np.append(weights[_i * 2], [weights[_i * 2 + 1]], axis=0))
            w_shape = np.shape(w)
            if _i == nn_decoder.nb_dense_layers-1:
                w = np.append(w, np.zeros([neurons_per_layer-w_shape[0], w_shape[1]]), axis=0)
            else:
                w = np.append(w, np.zeros([w_shape[0], neurons_per_layer+1-w_shape[1]]), axis=1)
            DemoMAPort.Write(
                nn_parameter_paths[_i],
                MyValueFactory.CreateFloatMatrixValue(
                Array[Array[float]](w.tolist())
                )
            )
        DemoMAPort.Write(
            learningRate,
            MyValueFactory.CreateFloatValue(learning_rate)
        )

        print("Waiting until Capture is running...")
        while DemoCapture.State != CaptureState.eRUNNING:
            time.sleep(0.02)
        print("Starting to fetch data...\n")

        # While the Capture is running, data is fetched in intervalls.
        # this data can be worked with while the capturing continues.
        # In case of this demo it is just printed into the console
        capture_count = 0
        print("STARTING")
        weights_socket.send(bytes(1))
        threading.Thread(target=nn_decoder.network_acquisition,
                         args=(weights_socket, DemoMAPort, nn_parameter_paths, MyValueFactory,)).start()

        while DemoCapture.State != CaptureState.eFINISHED:
            # time.sleep(0.00004) # sleep is for the weak
            DemoCaptureResult = DemoCapture.Fetch(False)
            nn_decoder.pipeline_active = True

            # --------------------------------------------------------------------------
            # Extract measured data from CaptureResult
            # --------------------------------------------------------------------------

            omegaSignalValue = DemoCaptureResult.ExtractSignalValue(masterTask, stateOmega)
            XAxisValues = convertIBaseValue(omegaSignalValue.XVector).Value
            omegaValues = convertIBaseValue(omegaSignalValue.FcnValues).Value

            currentIdSignalValue = DemoCaptureResult.ExtractSignalValue(slaveTask, stateCurrentId)
            currentIdValues = convertIBaseValue(currentIdSignalValue.FcnValues).Value

            currentIqSignalValue = DemoCaptureResult.ExtractSignalValue(slaveTask, stateCurrentIq)
            currentIqValues = convertIBaseValue(currentIqSignalValue.FcnValues).Value

            voltageUdSignalValue = DemoCaptureResult.ExtractSignalValue(slaveTask, stateVoltageUd)
            voltageUdValues = convertIBaseValue(voltageUdSignalValue.FcnValues).Value

            voltageUqSignalValue = DemoCaptureResult.ExtractSignalValue(slaveTask, stateVoltageUq)
            voltageUqValues = convertIBaseValue(voltageUqSignalValue.FcnValues).Value

            positionCosSignalValue = DemoCaptureResult.ExtractSignalValue(slaveTask, statePositionScaledCos)
            positionCosValues = convertIBaseValue(positionCosSignalValue.FcnValues).Value

            positionSinSignalValue = DemoCaptureResult.ExtractSignalValue(slaveTask, statePositionScaledSin)
            positionSinValues = convertIBaseValue(positionSinSignalValue.FcnValues).Value

            statorCurrentSignalValue = DemoCaptureResult.ExtractSignalValue(slaveTask, stateCurrentStator)
            statorCurrentValues = convertIBaseValue(statorCurrentSignalValue.FcnValues).Value

            torqueRefSignalValue = DemoCaptureResult.ExtractSignalValue(slaveTask, stateTorqueRef)
            torqueRefValues = convertIBaseValue(torqueRefSignalValue.FcnValues).Value

            # action, reward, done
            actionSignalValue = DemoCaptureResult.ExtractSignalValue(slaveTask, action)
            actionValues = convertIBaseValue(actionSignalValue.FcnValues).Value

            rewardSignalValue = DemoCaptureResult.ExtractSignalValue(slaveTask, reward)
            rewardValues = convertIBaseValue(rewardSignalValue.FcnValues).Value

            doneFlagSignalValue = DemoCaptureResult.ExtractSignalValue(slaveTask, doneFlag)
            doneFlagValues = convertIBaseValue(doneFlagSignalValue.FcnValues).Value

            # housekeeping
            newLearningRateSignalValue = DemoCaptureResult.ExtractSignalValue(slaveTask, learningRate)
            newLearningRateValue = convertIBaseValue(newLearningRateSignalValue.FcnValues).Value


            # --------------------------------------------------------------------------
            # Write the fetched data samples into the console window
            # --------------------------------------------------------------------------

            # For MP applications, the number of samples fetched by the masterApplication and the slaveApplication may be different.
            # To avoid IndexOutOfBounds errors, the lower value for NSamples is used. For single processor platforms, both values are the same.
            for date in zip(XAxisValues, newLearningRateValue, omegaValues, currentIdValues, currentIqValues,
                            voltageUdValues, voltageUqValues, positionCosValues, positionSinValues,
                            statorCurrentValues, torqueRefValues, actionValues, rewardValues, doneFlagValues):

                #print(date)

                # convert float data list to bytes
                b = bytes()
                b = b.join((struct.pack('f', val) for val in date))

                # send bytes
                if (len(b) <= BUFFER_SIZE):
                    data_socket.send(b)
                else:
                    for i in range(0, len(b) // BUFFER_SIZE):
                        data_socket.send(b[i * BUFFER_SIZE:(i + 1) * BUFFER_SIZE])

                    if len(b) > ((i + 1) * BUFFER_SIZE):
                        data_socket.send(b[(i + 1) * BUFFER_SIZE:])

                #remove data from byte stream for next data acquisition
                b = bytes()

            capture_count += 1

        data_socket.close()
        weights_socket.close()
        print("Capturing finished.\n")
        nn_decoder.pipeline_active = False

        print("Setting Trigger to 0.0 (off)\n")
        DemoMAPort.Write(manualCaptureTrigger, MyValueFactory.CreateFloatValue(0.0))

        print("")
        print("Demo successfully finished!\n")

    except TestbenchPortException as ex:
        # -----------------------------------------------------------------------
        # Display the vendor code description to get the cause of an error
        # -----------------------------------------------------------------------
        print("A TestbenchPortException occurred:")
        print("CodeDescription: %s" % ex.CodeDescription)
        print("VendorCodeDescription: %s" % ex.VendorCodeDescription)
        raise
    finally:
        # -----------------------------------------------------------------------
        # Attention: make sure to dispose the Capture object and the MAPort object in any case to free
        # system resources like allocated memory and also resources and services on the platform
        # -----------------------------------------------------------------------
        if DemoCapture != None:
            DemoCapture.Dispose()
            DemoCapture = None
        if DemoMAPort != None:
            DemoMAPort.Dispose()
            DemoMAPort = None
