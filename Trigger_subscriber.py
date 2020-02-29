import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

br = CvBridge()
img = None

i = 0

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('Trigger_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'blackfly',
            self.listener_callback,
            1)
        self.subscription  #prevent unused variable warning

    def listener_callback(self, msg):
        global br
        global img
        img = br.imgmsg_to_cv2(msg)
        global i
        cv2.imwrite('img'+str(i)+'.png', img)
        i+=1
        print(time.time())


def main(args=None):
    rclpy.init(args=args)

    # Figure(1) is default so you can omit this line. Figure(0) will create a new window every time program hits this line
    #fig = plt.figure(1)    

    minimal_subscriber = MinimalSubscriber()

    try:
        while True:
            # Draws an image on the current figure
            #if img is not None:
            #    plt.imshow(img, cmap='gray')

            # Interval in plt.pause(interval) determines how fast the images are displayed in a GUI
            # Interval is in seconds.
            #plt.pause(0.0005)
            # Clear current reference of a figure. This will improve display speed significantly
            #plt.clf()

            #Wait on the subscriber callback
            rclpy.spin_once(minimal_subscriber)

    except KeyboardInterrupt:
        plt.close('all')
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        minimal_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
