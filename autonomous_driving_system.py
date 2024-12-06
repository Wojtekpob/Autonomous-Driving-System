import time
import yaml
import threading
import numpy as np
import cv2
from lane_detection import LaneDetectionModule
from path_planning import PathPlanningModule
from visualization import VisualizationModule
from carla_client import CarlaClient
from mpc_controller import MPCController

class AutonomousDrivingSystem:
    def __init__(self, config_path):
        """
        Initializes the autonomous driving system.

        :param config_path: Path to the YAML configuration file.
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.client = CarlaClient(self.config)
        self.lane_detection = LaneDetectionModule(
            ckpt_path=self.config['lane_detection']['ckpt_path'],
            arch=self.config['lane_detection']['arch'],
            dual_decoder=self.config['lane_detection']['dual_decoder']
        )
        self.path_planning = PathPlanningModule(image_size=self.lane_detection.size)
        self.mpc_controller = MPCController()
        
        self.visualization = VisualizationModule(
            image_size=self.lane_detection.size
        )

        self.display = self.config['visualization']['show_results']
        if self.display:
            self.display_thread = threading.Thread(target=self._display_loop)
            self.display_thread.daemon = True
            self.frame_to_display = None
            self.frame_lock = threading.Lock()

    def get_vehicle_state(self):
        """
        Retrieves the current state of the vehicle from CARLA.

        :return: Dictionary containing position (x, y), orientation (psi), and speed (v) of the vehicle.
        """
        transform = self.client.vehicle.get_transform()
        velocity = self.client.vehicle.get_velocity()

        x = transform.location.x
        y = transform.location.y
        psi = np.deg2rad(transform.rotation.yaw)
        v = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        return {'x': x, 'y': y, 'psi': psi, 'v': v}

    def camera_callback(self, image):
        """
        Function called for each image received from the camera.

        :param image: Image from the camera.
        """
        image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
        image_data = image_data.reshape((image.height, image.width, 4))
        image_data = image_data[:, :, :3]
        image_resized = cv2.resize(image_data, self.lane_detection.size)

        pred_mask_resized, selected_params = self.lane_detection.predict(image_resized)

        pred_mask = cv2.resize(pred_mask_resized, (image_data.shape[1], image_data.shape[0]))

        vehicle_state = self.get_vehicle_state()

        trajectory_coeffs, cte, epsi = self.path_planning.plan_path(selected_params, vehicle_state)
        print(cte, epsi)
        if trajectory_coeffs is None:
            print("No sufficient data for path planning.")
            trajectory_coeffs = None
        else:
            state = np.array([0, 0, 0, vehicle_state['v'], cte, epsi])

            delta_opt, a_opt = self.mpc_controller.solve(state, trajectory_coeffs)
            print(delta_opt, a_opt)
            # self.client.apply_control(delta_opt, a_opt)

        if self.display:
            combined_img = self.visualization.visualize(
                image=image_data,
                lane_mask=pred_mask,
                trajectory_coeffs=trajectory_coeffs,
                lane_lines=selected_params,
                show=False
            )
            with self.frame_lock:
                self.frame_to_display = combined_img

    def _display_loop(self):
        """
        Thread responsible for displaying the visualization.
        """
        while True:
            with self.frame_lock:
                frame = self.frame_to_display
            if frame is not None:
                cv2.imshow('Autonomous Driving System', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)
        cv2.destroyAllWindows()

    def run(self):
        """
        Starts the autonomous driving system.
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


if __name__ == "__main__":
    driving_system = AutonomousDrivingSystem("config.yml")
    driving_system.run()