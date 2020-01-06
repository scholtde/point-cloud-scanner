#! /usr/bin/python

#--------------------------------#
# Kinect v2 point cloud visualization using a Numpy based 
# real-world coordinate processing algorithm and OpenGL.
#
# https://stackoverflow.com/questions/41241236/vectorizing-the-kinect-real-world-coordinate-processing-algorithm-for-speed
#--------------------------------#

import sys
import numpy as np
import socket
import socketserver
import json
from threading import Thread
from termcolor import colored
import logging

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame, libfreenect2


# Sensor Socket
data = None
azimuth = 0
HOST, PORT = "192.168.1.183", 55000
sensor_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sensor_sock.connect((HOST, PORT))

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial)

types = 0
types |= FrameType.Color
types |= (FrameType.Ir | FrameType.Depth)
listener = SyncMultiFrameListener(types)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()

# NOTE: must be called after device.start()
registration = Registration(device.getIrCameraParams(),
                            device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)


#QT app
app = QtGui.QApplication([])
gl_widget = gl.GLViewWidget()
gl_widget.show()
gl_grid = gl.GLGridItem()
gl_widget.addItem(gl_grid)

#initialize some points data
pos = np.zeros((1,3))

sp2 = gl.GLScatterPlotItem(pos=pos)
sp2.setGLOptions('opaque') # Ensures not to allow vertexes located behinde other vertexes to be seen.

gl_widget.addItem(sp2)

# Kinects's intrinsic parameters based on v2 hardware (estimated).
CameraParams = {
  "cx":254.878,
  "cy":205.395,
  "fx":365.456,
  "fy":365.456,
  "k1":0.0905474,
  "k2":-0.26819,
  "k3":0.0950862,
  "p1":0.0,
  "p2":0.0,
}

# Kinect's physical orientation in the real world.
CameraPosition = {
    "x": 0, # actual position in meters of kinect sensor relative to the viewport's center.
    "y": 0, # actual position in meters of kinect sensor relative to the viewport's center.
    "z": 0.96, # height in meters of actual kinect sensor from the floor.
    "roll": 0, # angle in degrees of sensor's roll (used for INU input - trig function for this is commented out by default).
    "azimuth": azimuth, # sensor's yaw angle in degrees.
    "elevation": 0, # sensor's pitch angle in degrees.
}

def depthToPointCloudPos(x_d, y_d, z, scale=1000):
    # This runs in Python slowly as it is required to be called from within a loop, but it is a more intuitive example than it's vertorized alternative (Purly for example)
    # calculate the real-world xyz vertex coordinate from the raw depth data (one vertex at a time).
    x = (x_d - CameraParams['cx']) * z / CameraParams['fx']
    y = (y_d - CameraParams['cy']) * z / CameraParams['fy']

    return x / scale, y / scale, z / scale

def depthMatrixToPointCloudPos(z, scale=1000):
    # bacically this is a vectorized version of depthToPointCloudPos()
    # calculate the real-world xyz vertex coordinates from the raw depth data matrix.
    C, R = np.indices(z.shape)

    R = np.subtract(R, CameraParams['cx'])
    R = np.multiply(R, z)
    R = np.divide(R, CameraParams['fx'] * scale)

    C = np.subtract(C, CameraParams['cy'])
    C = np.multiply(C, z)
    C = np.divide(C, CameraParams['fy'] * scale)

    return np.column_stack((z.ravel() / scale, R.ravel(), -C.ravel()))

def applyCameraOrientation(pt):
    # Kinect Sensor Orientation Compensation
    # This runs slowly in Python as it is required to be called within a loop, but it is a more intuitive example than it's vertorized alternative (Purly for example)
    # use trig to rotate a vertex around a gimbal.
    def rotatePoints(ax1, ax2, deg):
        # math to rotate vertexes around a center point on a plane.
        hyp = np.sqrt(pt[ax1] ** 2 + pt[ax2] ** 2) # Get the length of the hypotenuse of the real-world coordinate from center of rotation, this is the radius!
        d_tan = np.arctan2(pt[ax2], pt[ax1]) # Calculate the vertexes current angle (returns radians that go from -180 to 180)

        cur_angle = np.degrees(d_tan) % 360 # Convert radians to degrees and use modulo to adjust range from 0 to 360.
        new_angle = np.radians((cur_angle + deg) % 360) # The new angle (in radians) of the vertexes after being rotated by the value of deg.

        pt[ax1] = hyp * np.cos(new_angle) # Calculate the rotated coordinate for this axis.
        pt[ax2] = hyp * np.sin(new_angle) # Calculate the rotated coordinate for this axis.

    #rotatePoints(0, 2, CameraPosition['roll']) #rotate on the Y&Z plane # Disabled because most tripods don't roll. If an Inertial Nav Unit is available this could be used)
    rotatePoints(1, 2, CameraPosition['elevation']) #rotate on the X&Z plane
    rotatePoints(0, 1, CameraPosition['azimuth']) #rotate on the X&Y plane

    # Apply offsets for height and linear position of the sensor (from viewport's center)
    pt[:] += np.float_([CameraPosition['x'], CameraPosition['y'], CameraPosition['z']])



    return pt

def applyCameraMatrixOrientation(pt):
    global CameraPosition
    # Kinect Sensor Orientation Compensation
    # bacically this is a vectorized version of applyCameraOrientation()
    # uses same trig to rotate a vertex around a gimbal.
    def rotatePoints(ax1, ax2, deg):
        # math to rotate vertexes around a center point on a plane.
        hyp = np.sqrt(pt[:, ax1] ** 2 + pt[:, ax2] ** 2) # Get the length of the hypotenuse of the real-world coordinate from center of rotation, this is the radius!
        d_tan = np.arctan2(pt[:, ax2], pt[:, ax1]) # Calculate the vertexes current angle (returns radians that go from -180 to 180)

        cur_angle = np.degrees(d_tan) % 360 # Convert radians to degrees and use modulo to adjust range from 0 to 360.
        new_angle = np.radians((cur_angle + deg) % 360) # The new angle (in radians) of the vertexes after being rotated by the value of deg.

        pt[:, ax1] = hyp * np.cos(new_angle) # Calculate the rotated coordinate for this axis.
        pt[:, ax2] = hyp * np.sin(new_angle) # Calculate the rotated coordinate for this axis.

    #rotatePoints(1, 2, CameraPosition['roll']) #rotate on the Y&Z plane # Disabled because most tripods don't roll. If an Inertial Nav Unit is available this could be used)
    rotatePoints(0, 2, CameraPosition['elevation']) #rotate on the X&Z plane
    rotatePoints(0, 1, CameraPosition['azimuth']) #rotate on the X&Y
    print(CameraPosition['azimuth'])

    # Apply offsets for height and linear position of the sensor (from viewport's center)
    pt[:] += np.float_([CameraPosition['x'], CameraPosition['y'], CameraPosition['z']])



    return pt


def update():
    data = sensor_sock.recv(1024)
    data = data.decode("utf-8") 
    data = data.strip('\n')
    data = eval(data)
    #print(data["locationTrueHeading"])
    azimuth = data["locationTrueHeading"]

    # Kinect's physical orientation in the real world.
    global CameraPosition
    CameraPosition = {
        "x": 0, # actual position in meters of kinect sensor relative to the viewport's center.
        "y": 0, # actual position in meters of kinect sensor relative to the viewport's center.
        "z": 0.96, # height in meters of actual kinect sensor from the floor.
        "roll": 0, # angle in degrees of sensor's roll (used for INU input - trig function for this is commented out by default).
        "azimuth": azimuth, # sensor's yaw angle in degrees.
        "elevation": 0, # sensor's pitch angle in degrees.
    }

    colors = ((1.0, 1.0, 1.0, 1.0))

    frames = listener.waitForNewFrame()

    # Get the frames from the Kinect sensor
    ir = frames["ir"]
    color = frames["color"]
    depth = frames["depth"]

    d = depth.asarray() #the depth frame as an array (Needed only with non-vectorized functions)

    registration.apply(color, depth, undistorted, registered)

    # Format the color registration map - To become the "color" input for the scatterplot's setData() function.
    colors = registered.asarray(np.uint8)
    colors = np.divide(colors, 255) # values must be between 0.0 - 1.0
    colors = colors.reshape(colors.shape[0] * colors.shape[1], 4 ) # From: Rows X Cols X RGB -to- [[r,g,b],[r,g,b]...]
    colors = colors[:, :3:]  # remove alpha (fourth index) from BGRA to BGR
    colors = colors[...,::-1] #BGR to RGB

    # Calculate a dynamic vertex size based on window dimensions and camera's position - To become the "size" input for the scatterplot's setData() function.
    v_rate = 5.0 # Rate that vertex sizes will increase as zoom level increases (adjust this to any desired value).
    v_scale = np.float32(v_rate) / gl_widget.opts['distance'] # Vertex size increases as the camera is "zoomed" towards center of view.
    v_offset = (gl_widget.geometry().width() / 1000)**2 # Vertex size is offset based on actual width of the viewport.
    v_size = v_scale + v_offset

    # Calculate 3d coordinates (Note: five optional methods are shown - only one should be un-commented at any given time)

    """
    # Method 1 (No Processing) - Format raw depth data to be displayed
    m, n = d.shape
    R, C = np.mgrid[:m, :n]
    out = np.column_stack((d.ravel() / 4500, C.ravel()/m, (-R.ravel()/n)+1))
    """

    # Method 2 (Fastest) - Format and compute the real-world 3d coordinates using a fast vectorized algorithm - To become the "pos" input for the scatterplot's setData() function.
    out = depthMatrixToPointCloudPos(undistorted.asarray(np.float32))

    """
    # Method 3 - Format undistorted depth data to real-world coordinates
    n_rows, n_columns = d.shape
    out = np.zeros((n_rows * n_columns, 3), dtype=np.float32)
    for row in range(n_rows):
        for col in range(n_columns):
            z = undistorted.asarray(np.float32)[row][col]
            X, Y, Z = depthToPointCloudPos(row, col, z)
            out[row * n_columns + col] = np.array([Z, Y, -X])
    """

    """
    # Method 4 - Format undistorted depth data to real-world coordinates
    n_rows, n_columns = d.shape
    out = np.zeros((n_rows * n_columns, 3), dtype=np.float64)
    for row in range(n_rows):
        for col in range(n_columns):
            X, Y, Z = registration.getPointXYZ(undistorted, row, col)
            out[row * n_columns + col] = np.array([Z, X, -Y])
    """

    """
    # Method 5 - Format undistorted and regisered data to real-world coordinates with mapped colors (dont forget color=colors in setData)
    n_rows, n_columns = d.shape
    out = np.zeros((n_rows * n_columns, 3), dtype=np.float64)
    colors = np.zeros((d.shape[0] * d.shape[1], 3), dtype=np.float64)
    for row in range(n_rows):
        for col in range(n_columns):
            X, Y, Z, B, G, R = registration.getPointXYZRGB(undistorted, registered, row, col)
            out[row * n_columns + col] = np.array([Z, X, -Y])
            colors[row * n_columns + col] = np.divide([R, G, B], 255)
    """


    # Kinect sensor real-world orientation compensation.
    out = applyCameraMatrixOrientation(out)

    """
    # For demonstrating the non-vectorized orientation compensation function (slow)
    for i, pt in enumerate(out):
        out[i] = applyCameraOrientation(pt)
    """


    # Show the data in a scatter plot
    sp2.setData(pos=out, color=colors, size=v_size)

    # Lastly, release frames from memory.
    listener.release(frames)


t = QtCore.QTimer()
t.timeout.connect(update)
t.start(50)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

device.stop()
device.close()

sys.exit(0)
