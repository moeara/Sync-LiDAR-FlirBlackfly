# coding=utf-8
# =============================================================================
# Copyright (c) 2001-2019 FLIR Systems, Inc. All Rights Reserved.
#
# This software is the confidential and proprietary information of FLIR
# Integrated Imaging Solutions, Inc. ("Confidential Information"). You
# shall not disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with FLIR Integrated Imaging Solutions, Inc. (FLIR).

#  Trigger_Lidar.py triggers the camera. It relies on information
#  provided in the Trigger, ImageFormatControl, Enumeration, Acquisition, and NodeMapInfo examples.
#
#  It can also be helpful to familiarize yourself with the ImageFormatControl
#  and Exposure examples. As they are somewhat shorter and simpler, either
#  provides a strong introduction to camera customization.
#
#  This script shows the process of configuring, using, and cleaning up a
#  camera for use with both a software trigger with a VLP-16.

import os
import PySpin
import time
import socket

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.qos import QoSProfile
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from velodyne_msgs.msg import VelodyneScan

rclpy.init(args=None)

node = rclpy.create_node('blackfly')

# Modify the QoS profile of the publisher
'''qos_profile = QoSProfile(depth=10)
qos_profile.reliability = QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT
qos_profile.history = QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST
qos_profile.durability = QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE'''

publisher = node.create_publisher(Image, 'blackfly', 1) #qos_profile)

br = CvBridge()
dtype, n_channels = br.encoding_to_dtype_with_channels('8UC3')


class TriggerType:
    SOFTWARE = 1
    HARDWARE = 2

CHOSEN_TRIGGER = TriggerType.SOFTWARE


class TriggerSubscriber(Node):

    def __init__(self):
        super().__init__('Trigger_Subscriber')
        self.subscription = self.create_subscription(
            VelodyneScan,
            '/velodyne_packets',
            self.listener_callback,
            1)
        self.subscription  #prevent unused variable warning

    def listener_callback(self, msg):
        print("Trigger ", time.time())

# Subscriber to listen to the lidar messages for triggering
lidar_subscriber = TriggerSubscriber()

############################################################################


def configure_trigger(cam):
    """
    This function configures the camera to use a trigger. First, trigger mode is
    set to off in order to select the trigger source. Once the trigger source
    has been selected, trigger mode is then enabled, which has the camera
    capture only a single image upon the execution of the chosen trigger.

     :param cam: Camera to configure trigger for.
     :type cam: CameraPtr
     :return: True if successful, False otherwise.
     :rtype: bool
    """
    result = True

    print('*** CONFIGURING TRIGGER ***\n')

    if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
        print('Software trigger chosen ...')
    elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
        print('Hardware trigger chose ...')

    try:
        # Ensure trigger mode off
        # The trigger must be disabled in order to configure whether the source
        # is software or hardware.
        nodemap = cam.GetNodeMap()
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
            print('Unable to disable trigger mode (node retrieval). Aborting...')
            return False

        node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
        if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
            print('Unable to disable trigger mode (enum entry retrieval). Aborting...')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

        print('Trigger mode disabled...')

        # Select trigger source
        # The trigger source must be set to hardware or software while trigger
        # mode is off.
        node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
        if not PySpin.IsAvailable(node_trigger_source) or not PySpin.IsWritable(node_trigger_source):
            print('Unable to get trigger source (node retrieval). Aborting...')
            return False

        if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
            node_trigger_source_software = node_trigger_source.GetEntryByName('Software')
            if not PySpin.IsAvailable(node_trigger_source_software) or not PySpin.IsReadable(
                    node_trigger_source_software):
                print('Unable to set trigger source (enum entry retrieval). Aborting...')
                return False
            node_trigger_source.SetIntValue(node_trigger_source_software.GetValue())

        elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
            node_trigger_source_hardware = node_trigger_source.GetEntryByName('Line0')
            if not PySpin.IsAvailable(node_trigger_source_hardware) or not PySpin.IsReadable(
                    node_trigger_source_hardware):
                print('Unable to set trigger source (enum entry retrieval). Aborting...')
                return False
            node_trigger_source.SetIntValue(node_trigger_source_hardware.GetValue())

        # Turn trigger mode on
        # Once the appropriate trigger source has been set, turn trigger mode
        # on in order to retrieve images using the trigger.
        node_trigger_mode_on = node_trigger_mode.GetEntryByName('On')
        if not PySpin.IsAvailable(node_trigger_mode_on) or not PySpin.IsReadable(node_trigger_mode_on):
            print('Unable to enable trigger mode (enum entry retrieval). Aborting...')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_on.GetValue())
        print('Trigger mode turned back on...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


def configure_custom_image_settings(nodemap):
    """
    Configures a number of settings on the camera including offsets  X and Y, width,
    height, and pixel format. These settings must be applied before BeginAcquisition()
    is called; otherwise, they will be read only. Also, it is important to note that
    settings are applied immediately. This means if you plan to reduce the width and
    move the x offset accordingly, you need to apply such changes in the appropriate order.

    :param nodemap: GenICam nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    print('\n*** CONFIGURING CUSTOM IMAGE SETTINGS *** \n')

    try:
        result = True

        # Apply minimum to offset X
        #
        # *** NOTES ***
        # Numeric nodes have both a minimum and maximum. A minimum is retrieved
        # with the method GetMin(). Sometimes it can be important to check
        # minimums to ensure that your desired value is within range.
        node_offset_x = PySpin.CIntegerPtr(nodemap.GetNode('OffsetX'))
        if PySpin.IsAvailable(node_offset_x) and PySpin.IsWritable(node_offset_x):

            node_offset_x.SetValue(node_offset_x.GetMin())
            print('Offset X: %i...' + str(node_offset_x.GetValue()))

        else:
            print('Offset X not available...')

        # Apply minimum to offset Y
        #
        # *** NOTES ***
        # It is often desirable to check the increment as well. The increment
        # is a number of which a desired value must be a multiple of. Certain
        # nodes, such as those corresponding to offsets X and Y, have an
        # increment of 1, which basically means that any value within range
        # is appropriate. The increment is retrieved with the method GetInc().
        node_offset_y = PySpin.CIntegerPtr(nodemap.GetNode('OffsetY'))
        if PySpin.IsAvailable(node_offset_y) and PySpin.IsWritable(node_offset_y):

            node_offset_y.SetValue(node_offset_y.GetMin())
            print('Offset Y: ..' + str(node_offset_y.GetMin()))

        else:
            print('Offset Y not available...')

        # Set maximum width
        #
        # *** NOTES ***
        # Other nodes, such as those corresponding to image width and height,
        # might have an increment other than 1. In these cases, it can be
        # important to check that the desired value is a multiple of the
        # increment. However, as these values are being set to the maximum,
        # there is no reason to check against the increment.
        node_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
        if PySpin.IsAvailable(node_width) and PySpin.IsWritable(node_width):
            print(node_width.GetListOfValidValues())
            width_to_set = 1280  # node_width.GetMax()
            node_width.SetValue(width_to_set)
            print('Width set to ..' + str(node_width.GetValue()))

        else:
            print('Width not available...')

        # Set maximum height
        #
        # *** NOTES ***
        # A maximum is retrieved with the method GetMax(). A node's minimum and
        # maximum should always be a multiple of its increment.
        node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
        if PySpin.IsAvailable(node_height) and PySpin.IsWritable(node_height):

            height_to_set = 720  # node_height.GetMax()
            node_height.SetValue(height_to_set)
            print('Height set to...' + str(node_height.GetValue()))

        else:
            print('Height not available...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


def grab_next_image_by_trigger(nodemap, cam):
    """
    This function acquires an image by executing the trigger node.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        # Use trigger to capture image
        # The software trigger only feigns being executed by the Enter key;
        # what might not be immediately apparent is that there is not a
        # continuous stream of images being captured; in other examples that
        # acquire images, the camera captures a continuous stream of images.
        # When an image is retrieved, it is plucked from the stream.

        if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
            # Get user input #Used during calibration
            #input('Press the Enter key to initiate software trigger.')
            
            # Block till a lidar message is received
            rclpy.spin_once(lidar_subscriber)

            # Execute software trigger
            node_softwaretrigger_cmd = PySpin.CCommandPtr(nodemap.GetNode('TriggerSoftware'))
            if not PySpin.IsAvailable(node_softwaretrigger_cmd) or not PySpin.IsWritable(node_softwaretrigger_cmd):
                print('Unable to execute trigger. Aborting...')
                return False

            node_softwaretrigger_cmd.Execute()

            # TODO: Blackfly and Flea3 GEV cameras need 2 second delay after software trigger

        elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
            print('Use the hardware to trigger image acquisition.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


def acquire_images(cam, nodemap, nodemap_tldevice):
    """
    Please see Acquisition example for more in-depth comments on acquiring images.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    print('*** IMAGE ACQUISITION ***\n')
    try:
        result = True

        # Set acquisition mode to continuous
        # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print('Acquisition mode set to continuous...')

        #  Begin acquiring images
        cam.BeginAcquisition()

        print('Acquiring images...')

        #  Retrieve device serial number for filename
        #
        #  *** NOTES ***
        #  The device serial number is retrieved in order to keep cameras from
        #  overwriting one another. Grabbing image IDs could also accomplish
        #  this.
        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

        # Retrieve, convert, and publish images
        try:
            while True:
                try:

                    #  Retrieve the next image from the trigger
                    result &= grab_next_image_by_trigger(nodemap, cam)

                    #  Retrieve next received image
                    image_result = cam.GetNextImage()

                    #  Ensure image completion
                    if image_result.IsIncomplete():
                        print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                    else:

                        #  Print image information; height and width recorded in pixels
                        #
                        #  *** NOTES ***
                        #  Images have quite a bit of available metadata including
                        #  things such as CRC, image status, and offset values, to
                        #  name a few.
                        '''width = image_result.GetWidth()
                        height = image_result.GetHeight()
                        print('Grabbed Image width = %d, height = %d' % (width, height))'''

                        #  *** NOTES ***
                        #  Images can be converted between pixel formats by using the appropriate enumeration value. Unlike the 
                        # original image, the converted one does not need to be released as it does not affect the camera buffer.
                        #
                        #  When converting images, color processing algorithm is an optional parameter.
                        #image_converted = image_result.Convert(PySpin.PixelFormat_BayerBG8, PySpin.HQ_LINEAR)

                        #  *** NOTES ***
                        #  The standard practice of the examples is to use device
                        #  serial numbers to keep images of one device from overwriting those of another.
                        image_data = image_result.GetNDArray()

                        #The original image format is unsuitable for use in OpenCV so a conversion is required
                        #This line has been moved to the subscriber to reduce risk of increased latency in the driver/triggering
                        #Another idea is to carry out this operation on a seperate thread
                        image_data = cv2.cvtColor(image_data, cv2.COLOR_BAYER_RG2RGB)

                        msg = br.cv2_to_imgmsg(image_data)
                        
                        publisher.publish(msg)

                        #image_result.Save(filename)

                        #  Release image
                        #
                        #  *** NOTES ***
                        #  Images retrieved directly from the camera (i.e. non-converted images) 
                        #  need to be released in order to keep from filling the buffer.
                        image_result.Release()

                except PySpin.SpinnakerException as ex:
                    print('Error: %s' % ex)
                    return False

            # End acquisition
            #
            #  *** NOTES ***
            #  Ending acquisition appropriately helps ensure that devices clean up
            #  properly and do not need to be power-cycled to maintain integrity.
        except KeyboardInterrupt:
            cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


def reset_trigger(nodemap):
    """
    This function returns the camera to a normal state by turning off trigger mode.
  
    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :returns: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
            print('Unable to disable trigger mode (node retrieval). Aborting...')
            return False

        node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
        if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
            print('Unable to disable trigger mode (enum entry retrieval). Aborting...')
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

        print('Trigger mode disabled...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def print_device_info(nodemap):
    """
    This function prints the device information of the camera from the transport
    layer; please see NodeMapInfo example for more in-depth comments on printing
    device information from the nodemap.

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :returns: True if successful, False otherwise.
    :rtype: bool
    """

    print('*** DEVICE INFORMATION ***\n')

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

        else:
            print('Device control information not available.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


def run_single_camera(cam):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to run on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        err = False

        # Retrieve TL device nodemap and print device information
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        result &= print_device_info(nodemap_tldevice)

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Configure trigger
        if configure_trigger(cam) is False:
            return False

        # Configure custom image settings
        if not configure_custom_image_settings(nodemap):
            return False

        # Acquire images
        result &= acquire_images(cam, nodemap, nodemap_tldevice)

        # Reset trigger
        result &= reset_trigger(nodemap)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def main():
    """
    Example entry point; please see Enumeration example for more in-depth
    comments on preparing and cleaning up the system.

    :return: True if successful, False otherwise.
    :rtype: bool
    """

    # Since this application saves images in the current folder
    # we must ensure that we have permission to write to this folder.
    # If we do not have permission, fail right away.
    try:
        test_file = open('test.txt', 'w+')
    except IOError:
        print('Unable to write to current directory. Please check permissions.')
        input('Press Enter to exit...')
        return False

    test_file.close()
    os.remove(test_file.name)

    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:
        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False

    # Run example on each camera
    for i, cam in enumerate(cam_list):

        print('Running example for camera %d...' % i)

        result &= run_single_camera(cam)
        print('Camera %d example complete... \n' % i)

    # Release reference to camera
    # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
    # cleaned up when going out of scope.
    # The usage of del is preferred to assigning the variable to None.
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    lidar_subscriber.destroy_node()
    rclpy.shutdown()

    input('Done! Press Enter to exit...')
    return result


if __name__ == '__main__':
    main()
