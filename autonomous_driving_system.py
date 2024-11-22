import time
import yaml
from carla_client import CarlaClient

class AutonomousDrivingSystem:
    def __init__(self, config_path):
        """
        Initializes system.

        :param config_path: Path to YAML configuration file.
        """
        
        with open(config_path, 'r') as file:
            file_handle = yaml.safe_load(file)

        self.client = CarlaClient(file_handle)

    def camera_callback(self, image):
        """
        Function invoked after every image from camera

        :param image: Image from camera.
        """
        print(f'Received image frame: {image.frame}')

    def run(self):
        """
        Starts system
        """
        self.client.set_camera_callback(self.camera_callback)

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
