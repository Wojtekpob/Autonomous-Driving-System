import time
import yaml
import threading
import numpy as np
import cv2
from lane_detection import LaneDetectionModule

from carla_client import CarlaClient

class AutonomousDrivingSystem:
    def __init__(self, config_path):
        """
        Initializes system.

        :param config_path: Path to YAML configuration file.
        """
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.client = CarlaClient(self.config)
        self.lane_detection = LaneDetectionModule(
            ckpt_path=self.config['lane_detection']['ckpt_path'],
            arch=self.config['lane_detection']['arch'],
            dual_decoder=self.config['lane_detection']['dual_decoder']
        )

        self.display = self.config['lane_detection']['show_results']
        if self.display:
            self.display_thread = threading.Thread(target=self._display_loop)
            self.display_thread.daemon = True
            self.frame_to_display = None

    def camera_callback(self, image):
        """
        Function invoked after every image from camera

        :param image: Image from camera.
        """
        image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
        image_data = image_data.reshape((image.height, image.width, 4)) 
        image_data = image_data[:, :, :3]  
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)  

        pred_mask = self.lane_detection.predict(image_data)

        if self.display:
            combined_img = self.lane_detection.visualize_prediction(
                image=image_data,
                pred_mask=pred_mask,
                show=False 
            )
            self.frame_to_display = combined_img

        # @TODO implement mpc control


    def _display_loop(self):
        """
        Thread responsible for showing prediction result.
        """
        while True:
            if self.frame_to_display is not None:
                cv2.imshow('Prediction', self.frame_to_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)
        cv2.destroyAllWindows()

    def run(self):
        """
        Starts system.
        """
        self.client.set_camera_callback(self.camera_callback)

        if self.display:
            self.display_thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print('Stopping Autonomous Driving System...')
        finally:
            self.client.stop()


if __name__=="__main__":
    drivingSystem = AutonomousDrivingSystem("config.yml")
    drivingSystem.run()
