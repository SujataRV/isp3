# ----- Imports -------------------------------------------------------
# Standard imports
import random
import numpy as np
import time

# PyQT imports
from PyQt5.QtCore import QDateTime, Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QFileDialog)
from PyQt5.QtGui import QPainter, QColor, QFont
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# Local Imports
from gui_parser import uartParser
from gui_common import *
from graphUtilities import *

# ----- Defines -------------------------------------------------------
# Configurables
SNR_EXPECTED_MIN = 10 # Sets the min SNR we expect so that we can color points 
SNR_EXPECTED_MAX = 30 # Sets the max SNR we expect so that we can color points 
SNR_EXPECTED_RANGE = SNR_EXPECTED_MAX - SNR_EXPECTED_MIN # Sets the range we expect so that we can color points 
DOPPLER_EXPECTED_MIN = -30 # Sets the min Doppler we expect so that we can color points
DOPPLER_EXPECTED_MAX = 30 # Sets the max Doppler we expect so that we can color points 
DOPPLER_EXPECTED_RANGE = DOPPLER_EXPECTED_MAX - DOPPLER_EXPECTED_MIN

class parseUartThread(QThread):
        fin = pyqtSignal('PyQt_PyObject')

        def __init__(self, uParser):
                QThread.__init__(self)
                self.parser = uParser

        def run(self):
                outputDict = self.parser.readAndParseUart()
                self.fin.emit(outputDict)

class sendCommandThread(QThread):
        done = pyqtSignal()
        def __init__(self, uParser, command):
                QThread.__init__(self)
                self.parser = uParser
                self.command = command

        def run(self):
            self.parser.sendLine(self.command)
            self.done.emit()

class updateQTTargetThread3D(QThread):
    done = pyqtSignal()

    def __init__(self, pointCloud, targets, scatter, pcplot, numTargets, ellipsoids, coords, classifierOut=[], zRange=[-3, 3], colorGradient=[], pointColorMode="", drawTracks=True, trackColorMap=None):
        QThread.__init__(self)
        self.pointCloud = pointCloud
        self.targets = targets
        self.scatter = scatter
        self.pcplot = pcplot
        self.colorArray = ('r','g','b','w')
        self.numTargets = numTargets
        self.ellipsoids = ellipsoids
        self.coordStr = coords
        self.classifierOut = classifierOut
        self.zRange = zRange
        self.colorGradient = colorGradient
        self.pointColorMode = pointColorMode
        self.drawTracks = drawTracks
        self.trackColorMap = trackColorMap
        # This ignores divide by 0 errors when calculating the log2
        np.seterr(divide = 'ignore')

    def drawTrack(self, track, trackColor):
        # Get necessary track data
        tid = int(track[0])
        x = track[1]
        y = track[2]
        z = track[3]

        track = self.ellipsoids[tid]
        mesh = getBoxLinesCoords(x,y,z)
        track.setData(pos=mesh,color=trackColor,width=2,antialias=True,mode='lines')
        track.setVisible(True)
        

    def run(self):
        # Clear all previous targets
        for e in self.ellipsoids:
            if (e.visible()):
                e.hide()

        # Create a list of just X, Y, Z values to be plotted
        toPlot = self.pointCloud[:, 0:3]

        # Determine the size of each point based on its SNR
        with np.errstate(divide='ignore'):
            size = np.log2(self.pointCloud[:, 4])
        
        # Each color is an array of 4 values, so we need an numPoints*4 size 2d array to hold these values
        pointColors = np.zeros((self.pointCloud.shape[0], 4))
       
       # Color the points by their SNR
        if (self.pointColorMode == COLOR_MODE_SNR):
            for i in range(self.pointCloud.shape[0]):
                snr = self.pointCloud[i,4]
                # SNR value is out of expected bounds, make it white
                if (snr < SNR_EXPECTED_MIN) or (snr > SNR_EXPECTED_MAX):
                    pointColors[i] = pg.glColor('w')
                else:
                    pointColors[i] = pg.glColor(self.colorGradient.getColor((snr-SNR_EXPECTED_MIN)/SNR_EXPECTED_RANGE))

        # Color the points by their Height
        elif (self.pointColorMode == COLOR_MODE_HEIGHT):
            for i in range(self.pointCloud.shape[0]):
                #zs = self.zRange + (self.pointCloud[2,i] - self.zRange/2)
                zs = self.pointCloud[i, 2]

                # Points outside expected z range, make it white
                if (zs < self.zRange[0]) or (zs > self.zRange[1]):
                    pointColors[i] = pg.glColor('w')
                else:
                    colorRange = self.zRange[1]+abs(self.zRange[0]) 
                    #zs = colorRange/2 + zs 
                    #zs = self.zRange[0]-zs
                    zs = self.zRange[1] - zs 
                    #print(zs)
                    #print(self.zRange[1]+abs(self.zRange[0]))
                    #print(zs/colorRange)
                    pointColors[i]=pg.glColor(self.colorGradient.getColor(abs(zs/colorRange)))
        # Color Points by their doppler
        elif(self.pointColorMode == COLOR_MODE_DOPPLER):
            for i in range(self.pointCloud.shape[0]):
                doppler = self.pointCloud[i,3]
                # Doppler value is out of expected bounds, make it white
                if (doppler < DOPPLER_EXPECTED_MIN) or (doppler > DOPPLER_EXPECTED_MAX):
                    pointColors[i] = pg.glColor('w')
                else:
                    pointColors[i] = pg.glColor(self.colorGradient.getColor((doppler-DOPPLER_EXPECTED_MIN)/DOPPLER_EXPECTED_RANGE))
        elif (self.pointColorMode == COLOR_MODE_TRACK):
            for i in range(self.pointCloud.shape[0]):
                trackIndex = int(self.pointCloud[i, 6])
                # 253, 254, and 255 indicate that the point is not associated with a track
                if (trackIndex == 253 or trackIndex == 254 or trackIndex == 255 ):
                    pointColors[i] = pg.glColor('w')
                else:
                    pointColors[i] = self.trackColorMap[trackIndex]
        # Unknown Color Option, make all points green
        else:
            for i in range(self.pointCloud.shape[0]):
                pointColors[i]= pg.glColor('g')
            
        self.scatter.setData(pos=toPlot, color=pointColors, size=size)
        # Graph the targets
        if (self.drawTracks):
            if (self.targets is not None):
                for track in self.targets:
                    trackID = int(track[0])
                    trackColor = self.trackColorMap[trackID]
                    self.drawTrack(track,trackColor)
        self.done.emit()


class updateHeightGraphs(QThread):
    done = pyqtSignal('PyQt_PyObject')

    def __init__(self, targetSize, plots, frameNum, tids):
        QThread.__init__(self)
        self.targetSize = targetSize
        self.plots = plots
        self.frameNum = frameNum
        self.tids = tids

    def run(self):
        out ={'success':0, 'height':[],'mH':[],'dH':[],'x':[]}
        #start by plotting height data, mean height, and delta height of first TID only
        if (len(self.tids) > 0):
            tid = int(self.tids[0])
            age = int(self.targetSize[4,tid,0])
            height = self.targetSize[0,tid,:]
            mH = self.targetSize[5,tid,:]
            dH = self.targetSize[6,tid,:]
            fNum = self.frameNum%100
            shift=99-fNum
            height=np.roll(height,shift)
            mH=np.roll(mH,shift)
            dH=np.roll(dH,shift)
            if age<100:
                height[:int(100-age)]=0
                mH[:int(100-age)]=0
                dH[:int(100-age)]=0
            x=np.arange(self.frameNum-100,self.frameNum)
            out['success']=1
            out['height']=height
            out['mH']=mH
            out['dH']=dH
            out['x']=x
            self.done.emit(out)
        else:
            self.done.emit(out)


class updateVSHeightGraphs(QThread):
    done = pyqtSignal('PyQt_PyObject')

    def __init__(self, vs_in, plots, frameNum,ADCRAW_IN):
        QThread.__init__(self)
        self.plots = plots
        self.frameNum = frameNum
        self.vs = vs_in
        self.ADCRAW = ADCRAW_IN

    def run(self):
        #print("Target detected running updateHeightGraphs")
        #print(self.vs)
        out ={'wave0':[], 'wave1':[],'heart0':[],'heart1':[],'breath0':[],'breath1':[],\
                'heart_rate0':[], 'heart_rate1':[],'breathing_rate0':[],'breathing_rate1':[], \
                'x0':[],'x1':[],'y0':[],'y1':[],'z0':[],'z1':[],  \
                'id0':[],'id1':[],'range0':[],'range1':[],'angle0':[],'angle1':[], \
                'rangeidx0':[],'rangeidx1':[],'angleidx0':[],'angleidx1':[],'test':[]}
        #start by plotting height data, mean height, and delta height of first TID only
        #print(str(out))
        
        out['wave0'] = self.vs[0,0]
        out['wave1'] = self.vs[0,1]
        out['heart0'] = self.vs[1,0]
        out['heart1'] = self.vs[1,1]
        out['breath0'] = self.vs[2,0]
        out['breath1'] = self.vs[2,1]
        out['heart_rate0'] = self.vs[3,0]
        out['heart_rate1'] = self.vs[3,1]
        out['breathing_rate0'] = self.vs[4,0]
        out['breathing_rate1'] = self.vs[4,1]
        out['x0'] = self.vs[5,0]
        out['x1'] = self.vs[5,1]
        # out['x0'] = self.vs[13,0]
        # out['x1'] = self.vs[13,1]
        out['y0'] = self.vs[6,0]
        out['y1'] = self.vs[6,1]
        out['z0'] = self.vs[7,0]
        out['z1'] = self.vs[7,1]
        out['id0'] = self.vs[8,0]
        out['id1'] = self.vs[8,1]
        out['range0'] = self.vs[9,0]
        out['range1'] = self.vs[9,1]
        out['angle0'] = self.vs[10,0]
        out['angle1'] = self.vs[10,1]
        out['rangeidx0'] = self.vs[11,0]
        out['rangeidx1'] = self.vs[11,1]
        out['angleidx0'] = self.vs[12,0]
        out['angleidx1'] = self.vs[12,1]
        out['test'] = self.vs[13:674+13,0]
        out['ADCRAW'] = self.ADCRAW
        ###print(str(out))
        self.done.emit(out)

class zeroHeightGraphs(QThread):
    done = pyqtSignal('PyQt_PyObject')

    def __init__(self, vs_in, plots, frameNum,ADCRAW_IN):
        QThread.__init__(self)
        self.plots = plots
        self.frameNum = frameNum
        self.vs = vs_in
        self.ADCRAW = ADCRAW_IN

    def run(self):
        #print("No target detected running zeroHeightGraphs")
        out ={'wave0':[], 'wave1':[],'heart0':[],'heart1':[],'breath0':[],'breath1':[],\
                'heart_rate0':[], 'heart_rate1':[],'breathing_rate0':[],'breathing_rate1':[], \
                'x0':[],'x1':[],'y0':[],'y1':[],'z0':[],'z1':[],  \
                'id0':[],'id1':[],'range0':[],'range1':[],'angle0':[],'angle1':[], \
                'rangeidx0':[],'rangeidx1':[],'angleidx0':[],'angleidx1':[],'test':[]}
        #start by plotting height data, mean height, and delta height of first TID only
        #print(str(out))
        
        out['wave0'] = 0
        out['wave1'] = 0
        out['heart0'] = 0
        out['heart1'] = 0
        out['breath0'] = 0
        out['breath1'] = 0
        out['heart_rate0'] = 0
        out['heart_rate1'] = 0
        out['breathing_rate0'] = 0
        out['breathing_rate1'] = 0
        out['x0'] = 0
        out['x1'] = 0
        # out['x0'] = self.vs[13,0]
        # out['x1'] = self.vs[13,1]
        out['y0'] = 0
        out['y1'] = 0
        out['z0'] = 0
        out['z1'] = 0
        out['id0'] = 0
        out['id1'] = 0
        out['range0'] = 0
        out['range1'] = 0
        out['angle0'] = 0
        out['angle1'] = 0
        out['rangeidx0'] = 0
        out['rangeidx1'] = 0
        out['angleidx0'] = 0
        out['angleidx1'] = 0
        out['test'] = 0
        out['ADCRAW'] = 0
        self.done.emit(out)
