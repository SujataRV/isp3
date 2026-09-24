# ---------------------------------------------------------------------------
# gui_activity.py — mmWave 3D People Counting + real-time Activity Detection
# ---------------------------------------------------------------------------
# A trimmed-down build of gui_main.py that:
#   * auto-loads the .cfg sitting next to this file (AOP_6m_default.cfg)
#   * auto-selects the "3D People Counting" demo / parser
#   * shows a live HUMAN COUNT and a per-person ACTIVITY card
#     (crouching / running / stationary / walk_and_stop / walking)
#   * keeps only Config / COM-connect / Start / point-colour on the left
#
# Run:   python gui_activity.py
# ---------------------------------------------------------------------------

# ----- Imports -------------------------------------------------------
import sys
import os
import time
import math
import struct
import string
import random
import numpy as np

import serial
import serial.tools.list_ports

from PyQt5.QtCore import Qt, QTimer, QThread
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
        QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
        QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
        QTabWidget, QVBoxLayout, QWidget, QFileDialog)
from PyQt5 import QtGui
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from collections import OrderedDict

# Local File Imports
from gui_threads import *
from gui_parser import uartParser
from graphUtilities import *
from gui_common import *
from activity_predictor import MultiPersonPredictor
from stream_writer import StreamWriter, DEFAULT_STREAM_DIR

# ----- Defines -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Always load the TI toolbox's own chirp config — this is the one the models
# were trained against and the only one fall detection is validated on. The
# copy next to this script is only a fallback for machines without the
# toolbox installed (it is byte-identical apart from comments).
# RADAR_CFG overrides it (the Raspberry Pi has no TI toolbox installed, so the
# edge deployment points this at the copy shipped beside this script).
TI_CFG = os.getenv("RADAR_CFG") or (
          r"C:\ti\mmwave_industrial_toolbox_4_12_1\labs\People_Counting"
          r"\3D_People_Counting\chirp_configs\AOP_6m_default.cfg")
DEFAULT_CFG = TI_CFG if os.path.exists(TI_CFG) else os.path.join(BASE_DIR, 'AOP_6m_default.cfg')

# Label every tracked person. Set True to fall back to labelling only the
# closest track (the old single-person behaviour).
SINGLE_PERSON_MODE = False

# Colour per activity label (also used for the 3D text over each track)
LABEL_COLORS = {
    'crouching':     '#e74c3c',
    'running':       '#e67e22',
    'slow_walk':     '#f1c40f',
    'stationary':    '#2ecc71',
    'walk_and_stop': '#3498db',
    'walking':       '#9b59b6',
    'uncertain':     '#95a5a6',
    'collecting...': '#7f8c8d',
    'model missing': '#c0392b',
}


# Create a list of N distinct colors, visible on the dark GUI background
def get_trackColors(n):
    modKellyColors = [
        (230,  25,  75, 255),   # Red
        ( 60, 180,  75, 255),   # Green
        (255, 225,  25, 255),   # Yellow
        ( 67,  99, 216, 255),   # Blue
        (245, 130,  49, 255),   # Orange
        (145,  30, 180, 255),   # Purple
        ( 66, 212, 244, 255),   # Cyan
        (240,  50, 230, 255),   # Magenta
        (191, 239,  69, 255),   # Lime
        (250, 190, 212, 255),   # Pink
        ( 70, 153, 144, 255),   # Teal
        (220, 190, 255, 255),   # Lavender
        (154,  99,  36, 255),   # Brown
        (255, 250, 200, 255),   # Beige
        (128,   0,   0, 255),   # Maroon
        (170, 255, 195, 255),   # Mint
        (128, 128,   0, 255),   # Olive
        (255, 216, 177, 255),   # Apricot
        (  0,   0, 117, 255)    # Navy
    ]
    modKellyColorsNorm = [tuple(ti / 255 for ti in tup) for tup in modKellyColors]

    trackColorList = []
    for i in range(n):
        if i < len(modKellyColorsNorm):
            trackColorList.append(modKellyColorsNorm[i])
        else:
            (r_2, g_2, b_2, _) = modKellyColorsNorm[random.randint(0, len(modKellyColorsNorm) - 1)]
            (r_1, g_1, b_1, _) = modKellyColorsNorm[random.randint(0, len(modKellyColorsNorm) - 1)]
            gen = ((r_2 + r_1) / 2, (g_2 + g_1) / 2, (b_2 + b_1) / 2, 1.0)
            modKellyColorsNorm.append(gen)
            trackColorList.append(gen)
    return trackColorList


def makeActivityTextItem():
    """
    A 3D text label for the activity.

    NOTE: gl_classes.GLTextItem (used by gui_main.py) relies on the legacy
    OpenGL fixed-function matrix stack (glGetDoublev(GL_MODELVIEW_MATRIX) +
    gluProject) and on GLViewWidget.renderText/qglColor. pyqtgraph >= 0.13 is
    fully shader based: those matrices come back as the identity and the two
    widget methods no longer exist, so that item silently draws nothing (or
    draws off screen). pyqtgraph's own GLTextItem computes the projection from
    the item's mvpMatrix() and works correctly, so we use that instead.
    """
    font = QtGui.QFont('Helvetica', 13)
    font.setBold(True)
    item = gl.GLTextItem(pos=np.array([0.0, 0.0, 0.0]), text='', font=font,
                         color=QtGui.QColor('#ffffff'),
                         alignment=Qt.AlignHCenter | Qt.AlignBottom)
    item.setGLOptions('translucent')
    return item


def normColorToHex(c):
    """(r,g,b,a) normalized 0-1  ->  '#rrggbb'"""
    return '#{:02x}{:02x}{:02x}'.format(
        int(max(0, min(1, c[0])) * 255),
        int(max(0, min(1, c[1])) * 255),
        int(max(0, min(1, c[2])) * 255))


class Window(QDialog):
    def __init__(self, parent=None, size=[]):
        super(Window, self).__init__(parent)
        self.setWindowFlags(
            Qt.Window |
            Qt.CustomizeWindowHint |
            Qt.WindowTitleHint |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )
        self.setWindowTitle("mmWave People Counting + Activity Detection")

        print('Python is ', struct.calcsize("P") * 8, ' bit')
        print('Python version: ', sys.version_info)

        # ----- state -----
        self.frameTime = 50
        self.plotTargets = 1
        self.frameNum = 0
        self.profile = {'startFreq': 60.25, 'numLoops': 64, 'numTx': 3,
                        'sensorHeight': 1.5, 'maxRange': 10, 'az_tilt': 0,
                        'elev_tilt': 0, 'maxTracks': 20}
        self.configSent = 0
        self.trackColorMap = None
        self.previousClouds = []
        self.cfg = []
        self.cfgPath = ''

        # Activity detection state
        self.predictor = MultiPersonPredictor()
        self.activityResults = {}       # {tid: (label, conf)}
        self.primaryTid = None          # the person we label in single-person mode

        # isp3 live-stream bridge state
        self.streamWriter = StreamWriter()

        self.Gradients = OrderedDict([
            ('bw', {'ticks': [(0.0, (0, 0, 0, 255)), (1, (255, 255, 255, 255))], 'mode': 'rgb'}),
            ('hot', {'ticks': [(0.3333, (185, 0, 0, 255)), (0.6666, (255, 220, 0, 255)), (1, (255, 255, 255, 255)), (0, (0, 0, 0, 255))], 'mode': 'rgb'}),
            ('jet', {'ticks': [(1, (166, 0, 0, 255)), (0.32247191011235954, (0, 255, 255, 255)), (0.11348314606741573, (0, 68, 255, 255)), (0.6797752808988764, (255, 255, 0, 255)), (0.902247191011236, (255, 0, 0, 255)), (0.0, (0, 0, 166, 255)), (0.5022471910112359, (0, 255, 0, 255))], 'mode': 'rgb'}),
            ('heatmap', {'ticks': [(1, (255, 0, 0, 255)), (0, (131, 238, 255, 255))], 'mode': 'hsv'}),
        ])
        self.gradientMode = self.Gradients['heatmap']
        self.zRange = [-3, 3]

        if (size):
            self.setGeometry(50, 50, math.ceil(size.width() * 0.9), math.ceil(size.height() * 0.9))

        # ----- build UI -----
        self.init3dGraph()
        self.initColorGradient()
        self.initConnectionPane()
        self.initConfigPane()
        self.initPlotControlPane()
        self.initBoundaryBoxPane()
        self.initActivityPane()
        self.initStreamPane()

        leftCol = QVBoxLayout()
        leftCol.addWidget(self.comBox)
        leftCol.addWidget(self.configBox)
        leftCol.addWidget(self.plotControlBox)
        leftCol.addWidget(self.streamBox)
        leftCol.addStretch(1)
        leftWidget = QWidget()
        leftWidget.setLayout(leftCol)
        leftWidget.setMaximumWidth(340)

        gridlay = QGridLayout()
        gridlay.addWidget(leftWidget, 0, 0)
        gridlay.addWidget(self.pcplot, 0, 1)
        gridlay.addWidget(self.colorGradient, 0, 2)
        gridlay.addWidget(self.activityPane, 0, 3)
        gridlay.setColumnStretch(0, 0)
        gridlay.setColumnStretch(1, 4)
        gridlay.setColumnStretch(3, 1)
        self.setLayout(gridlay)

        # ----- auto setup: demo + config -----
        self.configType.setCurrentText(DEMO_NAME_3DPC)
        self.pointColorMode.setCurrentText(COLOR_MODE_TRACK)
        self.autoLoadCfg()

    # =====================================================================
    #  UI panes
    # =====================================================================
    def initConnectionPane(self):
        self.comBox = QGroupBox('1. Connect to COM Ports')
        self.cliCom = QLineEdit('')
        self.dataCom = QLineEdit('')
        self.connectStatus = QLabel('Not Connected')
        self.connectButton = QPushButton('Connect')
        self.connectButton.setMinimumHeight(34)
        self.connectButton.clicked.connect(self.connectCom)

        # Demo type is fixed to 3D People Counting — kept as a hidden widget so
        # the rest of the code (parser selection, persistence logic) is unchanged
        self.configType = QComboBox()
        self.configType.addItems([DEMO_NAME_3DPC])
        self.configType.setCurrentText(DEMO_NAME_3DPC)
        self.configType.setVisible(False)

        self.saveDataCheckbox = QCheckBox('Save Data to File')
        self.saveDataCheckbox.stateChanged.connect(self.toggleSaveData)

        self.comLayout = QGridLayout()
        self.comLayout.addWidget(QLabel('CLI COM:'), 0, 0)
        self.comLayout.addWidget(self.cliCom, 0, 1)
        self.comLayout.addWidget(QLabel('DATA COM:'), 1, 0)
        self.comLayout.addWidget(self.dataCom, 1, 1)
        self.comLayout.addWidget(self.connectButton, 2, 0)
        self.comLayout.addWidget(self.connectStatus, 2, 1)
        self.comLayout.addWidget(self.saveDataCheckbox, 3, 0, 1, 2)
        self.comBox.setLayout(self.comLayout)

        # Auto-detect the COM ports
        for port in list(serial.tools.list_ports.comports()):
            if (CLI_XDS_SERIAL_PORT_NAME in port.description or CLI_SIL_SERIAL_PORT_NAME in port.description):
                print(f'\tCLI COM Port found: {port.device}')
                self.cliCom.setText(port.device.replace("COM", ""))
            elif (DATA_XDS_SERIAL_PORT_NAME in port.description or DATA_SIL_SERIAL_PORT_NAME in port.description):
                print(f'\tData COM Port found: {port.device}')
                self.dataCom.setText(port.device.replace("COM", ""))

    def initConfigPane(self):
        self.configBox = QGroupBox('2. Configuration')
        self.cfgLabel = QLabel('No config loaded')
        self.cfgLabel.setWordWrap(True)
        self.cfgLabel.setStyleSheet('color: #f39c12;')

        self.selectConfig = QPushButton('Select Configuration')
        self.selectConfig.clicked.connect(self.selectCfg)

        self.sendConfig = QPushButton('START  (send config)')
        self.sendConfig.setMinimumHeight(44)
        self.sendConfig.setStyleSheet('font-weight: bold;')
        self.sendConfig.clicked.connect(self.sendCfg)

        self.start = QPushButton('Start without sending config')
        self.start.clicked.connect(self.startApp)

        self.configTable = QTableWidget(5, 2)
        self.configTable.setMaximumHeight(160)
        self.configTable.setItem(0, 0, QTableWidgetItem('Radar Parameter'))
        self.configTable.setItem(0, 1, QTableWidgetItem('Value'))
        self.configTable.setItem(1, 0, QTableWidgetItem('Max Range'))
        self.configTable.setItem(2, 0, QTableWidgetItem('Range Resolution'))
        self.configTable.setItem(3, 0, QTableWidgetItem('Max Velocity'))
        self.configTable.setItem(4, 0, QTableWidgetItem('Velocity Resolution'))

        self.configLayout = QVBoxLayout()
        self.configLayout.addWidget(self.cfgLabel)
        self.configLayout.addWidget(self.selectConfig)
        self.configLayout.addWidget(self.sendConfig)
        self.configLayout.addWidget(self.start)
        self.configLayout.addWidget(self.configTable)
        self.configBox.setLayout(self.configLayout)

    def initPlotControlPane(self):
        self.plotControlBox = QGroupBox('3. Display')
        self.pointColorMode = QComboBox()
        self.pointColorMode.addItems([COLOR_MODE_TRACK, COLOR_MODE_SNR,
                                      COLOR_MODE_HEIGHT, COLOR_MODE_DOPPLER])
        self.plotTracks = QCheckBox('Plot Tracks')
        self.plotTracks.setChecked(True)
        self.showActivityIn3D = QCheckBox('Activity label over person')
        self.showActivityIn3D.setChecked(True)
        self.persistentFramesInput = QComboBox()
        self.persistentFramesInput.addItems([str(i) for i in range(1, 11)])
        self.persistentFramesInput.setCurrentIndex(2)

        lay = QFormLayout()
        lay.addRow("Color Points By:", self.pointColorMode)
        lay.addRow("Persistent Frames:", self.persistentFramesInput)
        lay.addRow(self.plotTracks)
        lay.addRow(self.showActivityIn3D)
        self.plotControlBox.setLayout(lay)

    def initBoundaryBoxPane(self):
        # Boundary boxes still get drawn in 3D from the cfg, but there is no
        # editing UI in this build — the tab widget is created and kept hidden.
        self.boundaryBoxes = []
        self.boxTab = QTabWidget()
        self.boxTab.setVisible(False)

    def initStreamPane(self):
        """
        Bridge to isp3: writes a JSON file roughly every second into a fixed
        folder (radar_stream/) that isp3's rpi_pipeline/aws_watcher.py tails
        and forwards to the local fall + activity API. Independent of the
        archival 'Save Data to File' checkbox above — this one is meant to be
        started, stopped, and wiped freely, hence the delete button.
        """
        self.streamBox = QGroupBox('4. Stream to isp3')
        self.streamCheckbox = QCheckBox('Stream to isp3 (JSON / 1s)')
        self.streamCheckbox.stateChanged.connect(self.toggleStream)

        folderLabel = QLabel(DEFAULT_STREAM_DIR)
        folderLabel.setWordWrap(True)
        folderLabel.setStyleSheet('color: #888; font-size: 10px;')

        self.streamStatusLabel = QLabel('Not streaming')
        self.streamStatusLabel.setStyleSheet('color: #888; font-size: 11px;')

        self.streamDeleteButton = QPushButton('Stop && Delete Stream Files')
        self.streamDeleteButton.setStyleSheet('color: #c0392b;')
        self.streamDeleteButton.clicked.connect(self.stopAndDeleteStream)

        lay = QVBoxLayout()
        lay.addWidget(self.streamCheckbox)
        lay.addWidget(folderLabel)
        lay.addWidget(self.streamStatusLabel)
        lay.addWidget(self.streamDeleteButton)
        self.streamBox.setLayout(lay)

    def initActivityPane(self):
        self.activityPane = QGroupBox('People & Activity')
        self.activityPane.setMinimumWidth(190)
        self.activityPane.setMaximumWidth(230)
        lay = QVBoxLayout()

        # ---- big human counter ----
        countTitle = QLabel('HUMANS DETECTED')
        countTitle.setAlignment(Qt.AlignCenter)
        countTitle.setStyleSheet('color: #bdc3c7; font-size: 13px; font-weight: bold;')

        self.humanCountLabel = QLabel('0')
        self.humanCountLabel.setAlignment(Qt.AlignCenter)
        self.humanCountLabel.setStyleSheet(
            'color: #2ecc71; font-size: 64px; font-weight: bold;')

        lay.addWidget(countTitle)
        lay.addWidget(self.humanCountLabel)
        lay.addStretch(1)

        # ---- compact stats footer ----
        self.statsLabel = QLabel('Frame: 0   Points: 0')
        self.statsLabel.setStyleSheet('color: #95a5a6; font-size: 11px;')
        self.modelLabel = QLabel('')
        self.modelLabel.setWordWrap(True)
        self.modelLabel.setStyleSheet('color: #95a5a6; font-size: 11px;')
        if self.predictor.model_ready:
            self.modelLabel.setText('Model: OK  ({})'.format(', '.join(self.predictor.classes)))
        else:
            self.modelLabel.setStyleSheet('color: #e74c3c; font-size: 11px;')
            self.modelLabel.setText('Model NOT loaded — run:  pip install xgboost scikit-learn joblib')
        lay.addWidget(self.statsLabel)
        lay.addWidget(self.modelLabel)

        self.activityPane.setLayout(lay)

    def initColorGradient(self):
        self.colorGradient = pg.GradientWidget(orientation='right')
        self.colorGradient.restoreState(self.gradientMode)
        self.colorGradient.setVisible(False)

    def init3dGraph(self):
        self.pcplot = gl.GLViewWidget()
        self.pcplot.setBackgroundColor(70, 72, 79)
        self.gz = gl.GLGridItem()
        self.pcplot.addItem(self.gz)

        self.scatter = gl.GLScatterPlotItem(size=5)
        self.scatter.setData(pos=np.zeros((1, 3)))
        self.pcplot.addItem(self.scatter)

        # Box representing the EVM
        evmSizeX = 0.0625
        evmSizeZ = 0.125
        verts = np.empty((2, 3, 3))
        verts[0, 0, :] = [-evmSizeX, 0, evmSizeZ]
        verts[0, 1, :] = [-evmSizeX, 0, -evmSizeZ]
        verts[0, 2, :] = [evmSizeX, 0, -evmSizeZ]
        verts[1, 0, :] = [-evmSizeX, 0, evmSizeZ]
        verts[1, 1, :] = [evmSizeX, 0, evmSizeZ]
        verts[1, 2, :] = [evmSizeX, 0, -evmSizeZ]
        self.evmBox = gl.GLMeshItem(vertexes=verts, smooth=False, drawEdges=True,
                                    edgeColor=pg.glColor('r'), drawFaces=False)
        self.pcplot.addItem(self.evmBox)

        self.boundaryBoxViz = []
        self.coordStr = []
        self.ellipsoids = []

    # =====================================================================
    #  Sensor position (driven by the cfg, no UI)
    # =====================================================================
    def applySensorPosition(self):
        self.evmBox.resetTransform()
        self.evmBox.rotate(-1 * self.profile['elev_tilt'], 1, 0, 0)
        self.evmBox.rotate(-1 * self.profile['az_tilt'], 0, 0, 1)
        self.evmBox.translate(0, 0, self.profile['sensorHeight'])

    # =====================================================================
    #  Config handling
    # =====================================================================
    def autoLoadCfg(self):
        """Load the TI toolbox chirp config automatically at startup."""
        path = DEFAULT_CFG
        if path == TI_CFG:
            print('Using TI toolbox config: ' + TI_CFG)
        elif os.path.exists(path):
            print('TI toolbox config not found, using local copy: ' + path)
        if not os.path.exists(path):
            # fall back to any .cfg in this folder
            cfgs = [f for f in os.listdir(BASE_DIR) if f.lower().endswith('.cfg')]
            if not cfgs:
                self.cfgLabel.setText('No .cfg found — click "Select Configuration"')
                return
            path = os.path.join(BASE_DIR, cfgs[0])
        try:
            self.parseCfg(path)
            print('Auto-loaded config: ' + path)
        except Exception as e:
            print('Failed to auto-load config: ' + str(e))
            self.cfgLabel.setText('Config auto-load failed — select one manually')

    def selectCfg(self):
        try:
            fname = self.selectFile()
            if fname:
                self.parseCfg(fname)
        except Exception as e:
            print(e)
            print('No cfg file selected!')

    def selectFile(self):
        fd = QFileDialog()
        filename = fd.getOpenFileName(directory=BASE_DIR, filter="cfg(*.cfg)")
        return filename[0]

    def addBoundBox(self, name, minX=0, maxX=0, minY=0, maxY=0, minZ=0, maxZ=0):
        """Add a boundary box visualization (drawn from the cfg)."""
        boxLines = getBoxLines(minX, minY, minZ, maxX, maxY, maxZ)
        viz = gl.GLLinePlotItem()
        viz.setData(pos=boxLines, color=pg.glColor('b'), width=2,
                    antialias=True, mode='lines')
        viz.setVisible('trackerBounds' in name or 'occZone' in name)
        self.pcplot.addItem(viz)
        self.boundaryBoxViz.append(viz)
        self.boundaryBoxes.append({'name': name,
                                   'bounds': [minX, maxX, minY, maxY, minZ, maxZ]})

    def parseCfg(self, fname):
        with open(fname, 'r') as cfg_file:
            self.cfg = cfg_file.readlines()
        self.cfgPath = fname
        self.cfgLabel.setText('Config: ' + os.path.basename(fname))
        self.cfgLabel.setStyleSheet('color: #2ecc71;')

        # clear any previously created track objects / boxes
        for item in self.ellipsoids + self.coordStr + self.boundaryBoxViz:
            self.pcplot.removeItem(item)
        self.ellipsoids = []
        self.coordStr = []
        self.boundaryBoxViz = []
        self.boundaryBoxes = []

        counter = 0
        chirpCount = 0
        for line in self.cfg:
            args = line.split()
            if (len(args) > 0):
                if (args[0] == 'trackingCfg'):
                    if (len(args) < 5):
                        print("Error: trackingCfg had fewer arguments than expected")
                        continue
                    self.profile['maxTracks'] = int(args[4])
                    self.trackColorMap = get_trackColors(self.profile['maxTracks'])
                    for m in range(self.profile['maxTracks']):
                        mesh = gl.GLLinePlotItem()
                        mesh.setVisible(False)
                        self.pcplot.addItem(mesh)
                        self.ellipsoids.append(mesh)
                        text = makeActivityTextItem()
                        text.setVisible(False)
                        self.pcplot.addItem(text)
                        self.coordStr.append(text)
                elif (args[0] == 'SceneryParam' or args[0] == 'boundaryBox'):
                    if (len(args) < 7):
                        print("Error: SceneryParam/boundaryBox had fewer arguments than expected")
                        continue
                    self.addBoundBox('trackerBounds', float(args[1]), float(args[2]),
                                     float(args[3]), float(args[4]),
                                     float(args[5]), float(args[6]))
                elif (args[0] == 'profileCfg'):
                    if (len(args) < 12):
                        print("Error: profileCfg had fewer arguments than expected")
                        continue
                    self.profile['startFreq'] = float(args[2])
                    self.profile['idle'] = float(args[3])
                    self.profile['adcStart'] = float(args[4])
                    self.profile['rampEnd'] = float(args[5])
                    self.profile['slope'] = float(args[8])
                    self.profile['samples'] = float(args[10])
                    self.profile['sampleRate'] = float(args[11])
                elif (args[0] == 'frameCfg'):
                    if (len(args) < 4):
                        print("Error: frameCfg had fewer arguments than expected")
                        continue
                    self.profile['numLoops'] = float(args[3])
                    self.profile['numTx'] = float(args[2]) + 1
                elif (args[0] == 'chirpCfg'):
                    chirpCount += 1
                elif (args[0] == 'sensorPosition'):
                    if (len(args) < 4):
                        print("Error: sensorPosition had fewer arguments than expected")
                        continue
                    self.profile['sensorHeight'] = float(args[1])
                    self.profile['az_tilt'] = float(args[2])
                    self.profile['elev_tilt'] = float(args[3])
            counter += 1

        # Derived radar parameters for the info table
        try:
            self.profile['maxRange'] = self.profile['sampleRate'] * 1e3 * 0.9 * 3e8 / (2 * self.profile['slope'] * 1e12)
            bw = self.profile['samples'] / (self.profile['sampleRate'] * 1e3) * self.profile['slope'] * 1e12
            rangeRes = 3e8 / (2 * bw)
            Tc = (self.profile['idle'] * 1e-6 + self.profile['rampEnd'] * 1e-6) * chirpCount
            lda = 3e8 / (self.profile['startFreq'] * 1e9)
            maxVelocity = lda / (4 * Tc)
            velocityRes = lda / (2 * Tc * self.profile['numLoops'] * self.profile['numTx'])
            self.configTable.setItem(1, 1, QTableWidgetItem(str(self.profile['maxRange'])[:5]))
            self.configTable.setItem(2, 1, QTableWidgetItem(str(rangeRes)[:5]))
            self.configTable.setItem(3, 1, QTableWidgetItem(str(maxVelocity)[:5]))
            self.configTable.setItem(4, 1, QTableWidgetItem(str(velocityRes)[:5]))
        except Exception as e:
            print('Could not compute radar parameters: ' + str(e))

        print('Sensor height {} m, az tilt {}, elev tilt {}'.format(
            self.profile['sensorHeight'], self.profile['az_tilt'], self.profile['elev_tilt']))
        self.applySensorPosition()

    # =====================================================================
    #  Connection / start
    # =====================================================================
    def toggleSaveData(self, state):
        if hasattr(self, 'parser'):
            self.parser.setSaveBinary(bool(state))

    # ── isp3 live stream ─────────────────────────────────────────────────────
    def toggleStream(self, state):
        if state:
            self.streamWriter.start(cfg=self.cfg, demo=self.configType.currentText())
            self.streamStatusLabel.setText('Streaming — 0 files, 0 frames')
            self.streamStatusLabel.setStyleSheet('color: #2ecc71; font-size: 11px;')
        else:
            self.streamWriter.stop()
            self.streamStatusLabel.setText('Stopped ({} files written)'.format(
                self.streamWriter.files_written))
            self.streamStatusLabel.setStyleSheet('color: #888; font-size: 11px;')

    def stopAndDeleteStream(self):
        n_before = self.streamWriter.file_count()
        if n_before == 0 and not self.streamWriter.active:
            QMessageBox.information(self, 'Stream', 'No stream files to delete.')
            return
        reply = QMessageBox.question(
            self, 'Delete Stream Files',
            'Stop streaming and permanently delete {} JSON file(s) in:\n{}\n\n'
            'This only affects the isp3 stream folder — recordings you saved '
            'with "Save Data to File" are not touched.'.format(
                max(n_before, self.streamWriter.file_count()), DEFAULT_STREAM_DIR),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        self.streamCheckbox.setChecked(False)   # triggers toggleStream(False) -> stop()
        removed = self.streamWriter.delete_all()
        self.streamStatusLabel.setText('Deleted {} file(s)'.format(removed))
        self.streamStatusLabel.setStyleSheet('color: #888; font-size: 11px;')

    def connectCom(self):
        self.parser = uartParser(type=self.configType.currentText())
        self.parser.frameTime = self.frameTime
        self.parser.setSaveBinary(self.saveDataCheckbox.isChecked())
        print('Parser type: ', self.configType.currentText())

        self.uart_thread = parseUartThread(self.parser)
        self.uart_thread.fin.connect(self.parseData)
        self.uart_thread.fin.connect(self.updateGraph)
        self.parseTimer = QTimer()
        self.parseTimer.setSingleShot(False)
        self.parseTimer.timeout.connect(self.parseData)
        try:
            uart = "COM" + self.cliCom.text()
            data = "COM" + self.dataCom.text()
            self.parser.connectComPorts(uart, data)
            self.connectStatus.setText('Connected')
            self.connectStatus.setStyleSheet('color: #2ecc71;')
            self.connectButton.setText('Connected')
        except Exception as e:
            print(e)
            self.connectStatus.setText('Unable to Connect')
            self.connectStatus.setStyleSheet('color: #e74c3c;')

    def sendCfg(self):
        try:
            if not self.cfg:
                print('No cfg loaded!')
                return
            self.parser.sendCfg(self.cfg)
            self.configSent = 1
            self.predictor.reset()
            self.parseTimer.start(self.frameTime)
        except Exception as e:
            print(e)
            print('Connect to the COM ports first, then press START.')

    def startApp(self):
        try:
            self.configSent = 1
            self.predictor.reset()
            self.parseTimer.start(self.frameTime)
        except Exception as e:
            print(e)
            print('Connect to the COM ports first.')

    def parseData(self):
        # The thread's own 'fin' signal is wired back to parseData AND the
        # parse timer fires every frameTime ms, so this can be re-entered while
        # the previous read is still in flight. Restarting a running QThread is
        # undefined behaviour — skip instead.
        if (self.uart_thread.isRunning()):
            return
        self.uart_thread.start(priority=QThread.HighestPriority)

    # =====================================================================
    #  Frame handling
    # =====================================================================
    def selectPrimaryTrack(self, tracks):
        """Single-person mode: stick with the current person while they are
        still tracked, otherwise take the one closest to the radar."""
        if tracks is None or len(tracks) == 0:
            self.primaryTid = None
            return None
        ids = [int(t[0]) for t in tracks]
        if self.primaryTid in ids:
            return self.primaryTid
        # closest track = smallest horizontal range from the sensor
        closest = min(tracks, key=lambda t: float(t[1]) ** 2 + float(t[2]) ** 2)
        self.primaryTid = int(closest[0])
        return self.primaryTid

    def updateHumanCount(self, numTracks):
        self.humanCountLabel.setText(str(numTracks))
        if numTracks == 0:
            self.humanCountLabel.setStyleSheet('color: #7f8c8d; font-size: 64px; font-weight: bold;')
        else:
            self.humanCountLabel.setStyleSheet('color: #2ecc71; font-size: 64px; font-weight: bold;')

    def updateGraph(self, outputDict):
        pointCloud = outputDict.get('pointCloud', None)
        numPoints = outputDict.get('numDetectedPoints', 0)
        tracks = outputDict.get('trackData', None)
        trackIndexs = outputDict.get('trackIndexes', None)
        numTracks = outputDict.get('numDetectedTracks', 0)
        self.frameNum = outputDict.get('frameNum', 0)
        error = outputDict.get('error', 0)
        heights = outputDict.get('heightData', None)

        # Guard against empty / malformed clouds (nothing detected this frame)
        if (pointCloud is not None):
            pointCloud = np.asarray(pointCloud)
            if (pointCloud.ndim != 2 or pointCloud.shape[0] == 0 or pointCloud.shape[1] < 7):
                pointCloud = None
                outputDict = dict(outputDict)
                outputDict.pop('pointCloud', None)
        if (tracks is not None):
            tracks = np.asarray(tracks)
            if (tracks.ndim != 2 or tracks.shape[0] == 0):
                tracks = None
                numTracks = 0

        if (error != 0):
            print("Parsing Error on frame: %d" % (self.frameNum))
            print("\tError Number: %d" % (error))

        # ---- ACTIVITY DETECTION -------------------------------------------
        # Must run on the RAW point cloud, before tilt rotation / height shift,
        # because that is how the model was trained.
        try:
            self.activityResults = self.predictor.push(outputDict)
        except Exception as e:
            print('[activity] prediction error: ' + str(e))
            self.activityResults = {}
        self.updateHumanCount(numTracks)

        # ---- STREAM TO isp3 -------------------------------------------------
        # Same reason as above: must see the frame before tilt rotation /
        # height shift mutate it in place for on-screen display.
        if self.streamWriter.active:
            self.streamWriter.push(outputDict)
            self.streamStatusLabel.setText(
                'Streaming — {} files, {} frames'.format(
                    self.streamWriter.files_written, self.streamWriter.frames_written))

        self.statsLabel.setText('Frame: {}   Points: {}   Tracks: {}'.format(
            self.frameNum, numPoints, numTracks))

        # Rotate point cloud and tracks to account for elevation and azimuth tilt
        if (self.profile['elev_tilt'] != 0 or self.profile['az_tilt'] != 0):
            if (pointCloud is not None):
                for i in range(numPoints):
                    rotX, rotY, rotZ = eulerRot(pointCloud[i, 0], pointCloud[i, 1], pointCloud[i, 2],
                                                self.profile['elev_tilt'], self.profile['az_tilt'])
                    pointCloud[i, 0] = rotX
                    pointCloud[i, 1] = rotY
                    pointCloud[i, 2] = rotZ
            if (tracks is not None):
                for i in range(numTracks):
                    rotX, rotY, rotZ = eulerRot(tracks[i, 1], tracks[i, 2], tracks[i, 3],
                                                self.profile['elev_tilt'], self.profile['az_tilt'])
                    tracks[i, 1] = rotX
                    tracks[i, 2] = rotY
                    tracks[i, 3] = rotZ

        # Shift points to account for sensor height
        if (self.profile['sensorHeight'] != 0):
            if (pointCloud is not None):
                pointCloud[:, 2] = pointCloud[:, 2] + self.profile['sensorHeight']
            if (tracks is not None):
                tracks[:, 3] = tracks[:, 3] + self.profile['sensorHeight']

        # ---- floating text over each tracked person -----------------------
        for cstr in self.coordStr:
            cstr.setVisible(False)

        primaryTid = self.selectPrimaryTrack(tracks)

        if (tracks is not None and self.showActivityIn3D.isChecked()):
            for track in tracks:
                tid = int(track[0])
                if (tid >= len(self.coordStr)):
                    continue
                # Single-person mode: only label the primary person
                if (SINGLE_PERSON_MODE and tid != primaryTid):
                    continue

                label, conf = self.activityResults.get(tid, ('collecting...', 0.0))
                if conf > 0:
                    txt = '{}  {:.0f}%'.format(label.upper().replace('_', ' '), conf * 100)
                elif label == 'collecting...':
                    txt = 'READING… {:.0f}%'.format(self.predictor.window_fill(tid) * 100)
                else:
                    txt = label.upper().replace('_', ' ')

                item = self.coordStr[tid]
                # float the label ~0.4 m above the track centroid
                item.setData(pos=np.array([float(track[1]), float(track[2]),
                                           float(track[3]) + 0.4]),
                             text=txt,
                             color=QtGui.QColor(LABEL_COLORS.get(label, '#ffffff')))
                item.setVisible(True)

        # ---- point cloud persistence --------------------------------------
        numPersistentFrames = int(self.persistentFramesInput.currentText()) + 1

        if (trackIndexs is not None and len(self.previousClouds) > 0):
            if (self.previousClouds[-1].shape[0] == trackIndexs.shape[0]):
                self.previousClouds[-1][:, 6] = trackIndexs

        if pointCloud is not None:
            self.previousClouds.append(pointCloud)
        while (len(self.previousClouds) > numPersistentFrames):
            self.previousClouds.pop(0)
        if not self.previousClouds:
            return

        # Track indexes lag by one frame, so hold the newest frame back
        if (self.frameNum > 1 and len(self.previousClouds) > 1):
            cumulativeCloud = np.concatenate(self.previousClouds[:-1])
        else:
            cumulativeCloud = np.concatenate(self.previousClouds)

        self.drawScene(cumulativeCloud, tracks)

    def drawScene(self, cloud, tracks):
        """
        Draw the point cloud + track boxes.

        gui_main.py does this in a QThread (updateQTTargetThread3D), which
        mutates OpenGL items from a non-GUI thread while the widget may be
        painting them — a race that shows up as random hard crashes. The work
        here is a few milliseconds, so it runs inline on the GUI thread.
        """
        if cloud is None or cloud.shape[0] == 0:
            return

        with np.errstate(divide='ignore', invalid='ignore'):
            size = np.log2(np.maximum(cloud[:, 4], 1.0))

        mode = self.pointColorMode.currentText()
        n = cloud.shape[0]
        pointColors = np.zeros((n, 4))

        if (mode == COLOR_MODE_SNR):
            for i in range(n):
                snr = cloud[i, 4]
                if (snr < SNR_EXPECTED_MIN) or (snr > SNR_EXPECTED_MAX):
                    pointColors[i] = pg.glColor('w')
                else:
                    pointColors[i] = pg.glColor(self.colorGradient.getColor(
                        (snr - SNR_EXPECTED_MIN) / SNR_EXPECTED_RANGE))
        elif (mode == COLOR_MODE_HEIGHT):
            colorRange = self.zRange[1] + abs(self.zRange[0])
            for i in range(n):
                zs = cloud[i, 2]
                if (zs < self.zRange[0]) or (zs > self.zRange[1]):
                    pointColors[i] = pg.glColor('w')
                else:
                    pointColors[i] = pg.glColor(self.colorGradient.getColor(
                        abs((self.zRange[1] - zs) / colorRange)))
        elif (mode == COLOR_MODE_DOPPLER):
            for i in range(n):
                doppler = cloud[i, 3]
                if (doppler < DOPPLER_EXPECTED_MIN) or (doppler > DOPPLER_EXPECTED_MAX):
                    pointColors[i] = pg.glColor('w')
                else:
                    pointColors[i] = pg.glColor(self.colorGradient.getColor(
                        (doppler - DOPPLER_EXPECTED_MIN) / DOPPLER_EXPECTED_RANGE))
        elif (mode == COLOR_MODE_TRACK and self.trackColorMap is not None):
            for i in range(n):
                trackIndex = int(cloud[i, 6])
                if (trackIndex >= 253 or trackIndex >= len(self.trackColorMap)):
                    pointColors[i] = pg.glColor('w')
                else:
                    pointColors[i] = self.trackColorMap[trackIndex]
        else:
            pointColors[:] = pg.glColor('g')

        self.scatter.setData(pos=cloud[:, 0:3], color=pointColors, size=size)

        # Track bounding boxes
        for e in self.ellipsoids:
            if (e.visible()):
                e.hide()
        if (self.plotTracks.isChecked() and tracks is not None
                and self.trackColorMap is not None):
            for track in tracks:
                tid = int(track[0])
                if (tid >= len(self.ellipsoids) or tid >= len(self.trackColorMap)):
                    continue
                mesh = getBoxLinesCoords(track[1], track[2], track[3])
                self.ellipsoids[tid].setData(pos=mesh, color=self.trackColorMap[tid],
                                             width=2, antialias=True, mode='lines')
                self.ellipsoids[tid].setVisible(True)

    def closeEvent(self, event):
        try:
            self.predictor.close_log()
        except Exception:
            pass
        try:
            self.streamWriter.stop()   # flush any buffered frames before exit
        except Exception:
            pass
        super().closeEvent(event)


def _installCrashLogging():
    """
    Write a stack trace to crash_log.txt for both Python exceptions and hard
    interpreter crashes (segfault / access violation), so a crash during a
    live radar run can actually be diagnosed afterwards.
    """
    import faulthandler
    import traceback

    logPath = os.path.join(BASE_DIR, 'crash_log.txt')
    logFile = open(logPath, 'a', buffering=1, encoding='utf-8')
    logFile.write('\n===== session started {} =====\n'.format(
        time.strftime('%Y-%m-%d %H:%M:%S')))
    faulthandler.enable(file=logFile, all_threads=True)

    def hook(exctype, value, tb):
        text = ''.join(traceback.format_exception(exctype, value, tb))
        logFile.write(text)
        sys.stderr.write(text)

    sys.excepthook = hook
    print('Crash log: ' + logPath)


if __name__ == '__main__':
    _installCrashLogging()
    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    size = screen.size()
    main = Window(size=size)
    main.show()
    sys.exit(app.exec_())
