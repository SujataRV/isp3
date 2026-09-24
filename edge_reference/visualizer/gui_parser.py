# ----- Imports -------------------------------------------------------

# Standard Imports
import struct
import serial
import time
import numpy as np
import math
import datetime
import json
import os

# json_fix allows json.dumps to handle numpy arrays automatically
# Install with: pip install json-fix
try:
    import json_fix
    json.fallback_table[np.ndarray] = lambda array: array.tolist()
except ImportError:
    print("Warning: json_fix not installed. Install it with: pip install json-fix")

# Local Imports
from parseFrame import *


def _detach(obj):
    """
    Deep-copy a parsed frame so later in-place edits by the GUI can't reach it.
    numpy arrays are copied; dicts/lists are rebuilt; scalars pass through.
    """
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, dict):
        return {k: _detach(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_detach(v) for v in obj]
    return obj

# Initialize this Class to create a UART Parser. Initialization takes one argument:
# 1: String Lab_Type - These can be:
#   a. 3D People Counting
#   b. SDK Out of Box Demo
#   c. Long Range People Detection
#   d. Indoor False Detection Mitigation
#   e. (Legacy): Overhead People Counting
#   f. (Legacy) 2D People Counting
# Default is (f). Once initialized, call connectComPorts(self, cliComPort, DataComPort) to connect to device com ports.
# Then call readAndParseUart() to read one frame of data from the device. The gui this is packaged with calls this every frame period.
# readAndParseUart() will return all radar detection and tracking information.
class uartParser():
    def __init__(self, type='SDK Out of Box Demo'):
        self.replay = 0

        if (type == DEMO_NAME_OOB):
            self.parserType = "Standard"
        elif (type == DEMO_NAME_LRPD):
            self.parserType = "Standard"
        elif (type == DEMO_NAME_3DPC):
            self.parserType = "Standard"
        elif (type == DEMO_NAME_SOD):
            self.parserType = "Standard"
        elif (type == DEMO_NAME_VITALS):
            self.parserType = "Standard"
        elif (type == DEMO_NAME_MT):
            self.parserType = "Standard"
        # TODO Implement these
        elif (type == "Replay"):
            self.replay = 1
        else:
            print("ERROR, unsupported demo type selected!")

        # Data storage (legacy)
        self.now_time = datetime.datetime.now().strftime('%Y%m%d-%H%M')

        # ---- JSON Save state ----
        self.saveBinary    = 0       # Set to 1 via setSaveBinary(True) to enable saving
        self.frames        = []      # Buffer of parsed frame dicts
        self.uartCounter   = 0       # Total frames received since saving started
        self.framesPerFile = 100     # How many frames per JSON file
        self.first_file    = True    # Whether the output folder has been created yet
        # Unique folder name per session based on timestamp
        self.filepath = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.cfg  = []               # Stores cfg lines so they can be embedded in JSON
        self.demo = type             # Demo name embedded in JSON


    # ------------------------------------------------------------------
    # Enable / disable JSON saving at runtime
    # ------------------------------------------------------------------
    def setSaveBinary(self, saveBinary):
        self.saveBinary = saveBinary
        if saveBinary:
            print("JSON saving ENABLED  ->  ./binData/" + self.filepath + "/")
        else:
            print("JSON saving DISABLED")


    # ------------------------------------------------------------------
    # Legacy binary writer — kept for reference, not called by default
    # ------------------------------------------------------------------
    def WriteFile(self, data):
        filepath = self.now_time + '.bin'
        objStruct = '6144B'
        objSize = struct.calcsize(objStruct)
        binfile = open(filepath, 'ab+')   # open binary file for append
        binfile.write(bytes(data))
        binfile.close()


    # ------------------------------------------------------------------
    # Main parse + save function
    # ------------------------------------------------------------------
    def readAndParseUart(self):
        magicWord = bytearray(b'\x02\x01\x04\x03\x06\x05\x08\x07')
        self.fail = 0
        if (self.replay):
            return self.replayHist()

        # Find magic word, and therefore the start of the frame
        index = 0
        magicByte = self.dataCom.read(1)
        frameData = bytearray(b'')
        while (1):
            # Found matching byte
            if (magicByte[0] == magicWord[index]):
                index += 1
                frameData.append(magicByte[0])
                if (index == 8):   # Found the full magic word
                    break
                magicByte = self.dataCom.read(1)
            else:
                # When you fail, compare current byte against first byte of sequence too
                if (index == 0):
                    magicByte = self.dataCom.read(1)
                index = 0                    # Reset index
                frameData = bytearray(b'')   # Reset current frame data

        # Read in version from the header
        versionBytes = self.dataCom.read(4)
        frameData += bytearray(versionBytes)

        # Read in length from header
        lengthBytes = self.dataCom.read(4)
        frameData += bytearray(lengthBytes)
        frameLength = int.from_bytes(lengthBytes, byteorder='little')

        # Subtract bytes already read (magic word + version + length = 16 bytes)
        frameLength -= 16

        # Read in rest of the frame
        frameData += bytearray(self.dataCom.read(frameLength))

        # frameData now contains an entire frame — send it to the parser
        if (self.parserType == "Standard"):
            outputDict = parseStandardFrame(frameData)
        else:
            print('FAILURE: Bad parserType')
            return {}

        # ---- JSON Save Logic ----
        if self.saveBinary == 1:
            self.uartCounter += 1

            # Build the per-frame record.
            #
            # This MUST be a detached copy, not a reference. outputDict holds
            # numpy arrays, and the GUI's updateGraph() rotates them for
            # sensor tilt and adds sensorHeight to z **in place** on these
            # very arrays. Since the buffer is only serialised every
            # framesPerFile frames, storing a reference meant the points were
            # already transformed by the time they hit disk — every saved
            # recording came out in world coordinates (z offset by ~ the
            # sensor height) instead of the raw sensor frame it claims to be.
            # Models trained on raw captures then saw shifted z at inference,
            # which skews the activity classifier heavily toward "crouching".
            self.frames.append({
                'frameData': _detach(outputDict),
                'timestamp': time.time() * 1000   # milliseconds
            })

            # Every framesPerFile frames, flush buffer to a JSON file
            if (self.uartCounter % self.framesPerFile == 0):
                self._writeJsonFile()

        return outputDict


    # ------------------------------------------------------------------
    # Internal: flush current frame buffer to a JSON file
    # ------------------------------------------------------------------
    def _writeJsonFile(self):
        # Create the output folder on first write
        if self.first_file:
            if not os.path.exists('binData'):
                os.mkdir('binData')
            os.mkdir(os.path.join('binData', self.filepath))
            self.first_file = False

        # Full data package to save
        data = {
            'cfg':  self.cfg,
            'demo': self.demo,
            'data': self.frames
        }

        fileIndex = math.floor(self.uartCounter / self.framesPerFile)
        filePath  = os.path.join(
            'binData', self.filepath,
            'replay_' + str(fileIndex) + '.json'
        )

        try:
            with open(filePath, 'w') as fp:
                json_object = json.dumps(data, indent=4)
                fp.write(json_object)
            print("Saved " + str(len(self.frames)) + " frames -> " + filePath)
        except Exception as e:
            print("ERROR: Failed to write JSON file: " + str(e))

        # Reset buffer so next file starts fresh
        self.frames = []


    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------

    # Connect to CLI and Data COM ports
    def connectComPorts(self, cliCom, dataCom):
        self.cliCom  = serial.Serial(cliCom,  115200, parity=serial.PARITY_NONE,
                                     stopbits=serial.STOPBITS_ONE, timeout=0.3)
        self.dataCom = serial.Serial(dataCom, 921600, parity=serial.PARITY_NONE,
                                     stopbits=serial.STOPBITS_ONE, timeout=0.3)
        self.dataCom.reset_output_buffer()
        print('Connected')

    # Send cfg over UART
    def sendCfg(self, cfg):
        # Store cfg lines so they are embedded in saved JSON files
        self.cfg = cfg

        for line in cfg:
            time.sleep(.03)
            self.cliCom.write(line.encode())
            ack = self.cliCom.readline()
            print(ack)
            ack = self.cliCom.readline()
            print(ack)
        time.sleep(3)
        self.cliCom.reset_input_buffer()
        self.cliCom.close()

    # Send single command to device over UART
    def sendLine(self, line):
        self.cliCom.write(line.encode())
        ack = self.cliCom.readline()
        print(ack)
        ack = self.cliCom.readline()
        print(ack)


def getBit(byte, bitNum):
    mask = 1 << bitNum
    if (byte & mask):
        return 1
    else:
        return 0