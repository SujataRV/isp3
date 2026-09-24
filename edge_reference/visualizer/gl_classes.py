import pyqtgraph.opengl as gl
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.Qt import QtCore, QtGui

class GLTextItem(GLGraphicsItem):
    def __init__(self, X=None, Y=None, Z=None, text=None):
        GLGraphicsItem.__init__(self)

        self.text = text
        self.X = X
        self.Y = Y
        self.Z = Z

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget

    def setText(self, text):
        self.text = text
        self.update()

    def setX(self, X):
        self.X = X
        self.update()

    def setY(self, Y):
        self.Y = Y
        self.update()

    def setZ(self, Z):
        self.Z = Z
        self.update()

    def setPosition(self, X, Y, Z):
        self.X = X + 0.25
        self.Z = Z + 0.6
        self.Y = Y
        self.text = '(' + str(X)[:4] + ', ' + str(Y)[:4] + ', ' + str(Z)[:4] + ')'
        self.update()

    def paint(self):
        # Guard: skip if any position or text is not set yet
        if self.text is None or self.X is None or self.Y is None or self.Z is None:
            return

        # --- Method 1: Legacy renderText (pyqtgraph < 0.12 / older PyQt5) ---
        try:
            self.GLViewWidget.qglColor(QtCore.Qt.white)
            self.GLViewWidget.renderText(self.X, self.Y, self.Z, self.text)
            return   # success — done
        except AttributeError:
            pass     # renderText removed in newer versions, fall through

        # --- Method 2: Modern fallback using OpenGL projection + QPainter ---
        # Projects the 3D world coordinate to 2D screen space,
        # then draws the text using QPainter on top of the GL widget.
        try:
            from OpenGL.GL import (glGetDoublev, glGetIntegerv,
                                   GL_MODELVIEW_MATRIX,
                                   GL_PROJECTION_MATRIX,
                                   GL_VIEWPORT)
            from OpenGL.GLU import gluProject

            mv   = glGetDoublev(GL_MODELVIEW_MATRIX)
            proj = glGetDoublev(GL_PROJECTION_MATRIX)
            view = glGetIntegerv(GL_VIEWPORT)

            # Project 3-D point to window coordinates
            sx, sy, sz = gluProject(self.X, self.Y, self.Z, mv, proj, view)

            # sy is measured from the bottom in OpenGL, flip for Qt
            sy_flipped = self.GLViewWidget.height() - int(sy)

            painter = QtGui.QPainter(self.GLViewWidget)
            painter.setPen(QtGui.QColor(255, 255, 255))       # white text
            painter.setFont(QtGui.QFont('Helvetica', 10))
            painter.drawText(int(sx), sy_flipped, self.text)
            painter.end()
            return   # success — done

        except Exception:
            pass     # OpenGL/GLU not available or projection failed — silently skip