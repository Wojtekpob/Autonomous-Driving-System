import carla
import yaml
import os
import time
import numpy as np

class CarlaClient:
    def __init__(self, config):
        """
        Initializes the CARLA client based on configuration.

        :param config: Configuration dictionary.
        """
        self.config = config
        self.host = self.config['carla']['host']
        self.port = self.config['carla']['port']
        self.timeout = self.config['carla']['timeout']
        self.image_save_path = self.config['camera']['image_save_path']
        self.sampling_time = self.config['camera']['sampling_time']
        self.autopilot = self.config['vehicle'].get('autopilot', False)
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.map_name = self.config['carla'].get('map', 'Town01')
        self._load_map(self.map_name)
        self.world = self.client.get_world()
        self.vehicle = None
        self.camera = None
        self.camera_top = None
        self.camera_callback = None
        self.camera_top_callback = None
        self._last_image_time = None
        self._last_image_time_top = None
        self._initialize_simulation()

    def _load_map(self, map_name):
        """
        Loads the specified map in the CARLA simulator.

        :param map_name: Name of the map to load (e.g., 'Town01', 'Town02', etc.).
        """
        available_maps = self.client.get_available_maps()
        if f"/Game/Carla/Maps/{map_name}" in available_maps:
            print(f"Loading map: {map_name}")
            self.client.load_world(map_name)
        else:
            raise ValueError(f"Map {map_name} is not available. Available maps: {available_maps}")

    def _initialize_simulation(self):
        """
        Initializes the vehicle and cameras in the simulation.
        """
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(self.config['vehicle']['blueprint'])
        # print(len(self.world.get_map().get_spawn_points()))
        # print(self.world.get_map().get_spawn_points())
        spawn_point = self.world.get_map().get_spawn_points()[9] # 100
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f'Spawned vehicle: {self.vehicle.type_id}')
        self.vehicle.set_autopilot(self.autopilot)
        if self.autopilot:
            print('Autopilot is enabled for the vehicle.')
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.config['camera']['image_size_x']))
        camera_bp.set_attribute('image_size_y', str(self.config['camera']['image_size_y']))
        camera_bp.set_attribute('fov', str(self.config['camera']['fov']))
        camera_transform = carla.Transform(
            carla.Location(
                x=self.config['camera']['transform']['x'],
                y=self.config['camera']['transform']['y'],
                z=self.config['camera']['transform']['z']
            ),
            carla.Rotation(
                pitch=self.config['camera']['transform']['pitch'],
                yaw=self.config['camera']['transform']['yaw'],
                roll=self.config['camera']['transform']['roll']
            )
        )
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        print('Spawned camera')
        camera_bp_top = blueprint_library.find('sensor.camera.rgb')
        camera_bp_top.set_attribute('image_size_x', str(self.config['camera']['image_size_x']))
        camera_bp_top.set_attribute('image_size_y', str(self.config['camera']['image_size_y']))
        camera_bp_top.set_attribute('fov', str(self.config['camera']['fov']))
        camera_transform_top = carla.Transform(
            carla.Location(
                x=-8.0,
                y=0.0,
                z=5.0
            ),
            carla.Rotation(
                pitch=-6.0,
                yaw=0.0,
                roll=0.0
            )
        )
        self.camera_top = self.world.spawn_actor(camera_bp_top, camera_transform_top, attach_to=self.vehicle)
        print('Spawned top camera')
        if not os.path.exists(self.image_save_path):
            os.makedirs(self.image_save_path)

    def set_weather_for_lane_visibility(self):
        """
        Sets the weather in CARLA to ensure good lane visibility.
        """
        weather = carla.WeatherParameters(
            cloudiness=self.config['carla']['weather']['cloudiness'],
            precipitation=self.config['carla']['weather']['precipitation'],
            precipitation_deposits=self.config['carla']['weather']['precipitation_deposits'],
            wind_intensity=self.config['carla']['weather']['wind_intensity'],
            sun_azimuth_angle=self.config['carla']['weather']['sun_azimuth_angle'],
            sun_altitude_angle=self.config['carla']['weather']['sun_altitude_angle'],
            fog_density=self.config['carla']['weather']['fog_density'],
            fog_distance=self.config['carla']['weather']['fog_distance'],
            fog_falloff=self.config['carla']['weather']['fog_falloff']
        )
        self.world.set_weather(weather)
        print("Weather set.")

    def apply_control(self, steer, throttle):
        """
        Applies control commands to the vehicle.

        :param steer: Steering angle in radians.
        :param throttle: Throttle value between -1.0 and 1.0.
        """
        control = carla.VehicleControl()
        control.steer = np.clip(steer / 0.436332, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.brake = 0.0 if throttle >= 0 else -throttle
        self.vehicle.apply_control(control)

    def set_camera_callback(self, callback):
        """
        Sets the callback function for the front camera.

        :param callback: Function invoked after every image from the camera.
        """
        self.camera_callback = callback
        self.camera.listen(self._camera_listener)

    def set_camera_top_callback(self, callback):
        """
        Sets the callback function for the top camera.

        :param callback: Function invoked after every image from the top camera.
        """
        self.camera_top_callback = callback
        self.camera_top.listen(self._camera_listener_top)

    def _camera_listener(self, image):
        """
        Function invoked after every image from the front camera.
        """
        current_time = time.time()
        if self._last_image_time is not None:
            if current_time - self._last_image_time < self.sampling_time:
                return
        self._last_image_time = current_time
        if self.camera_callback is not None:
            self.camera_callback(image)

    def _camera_listener_top(self, image):
        """
        Function invoked after every image from the top camera.
        """
        current_time = time.time()
        if self._last_image_time_top is not None:
            if current_time - self._last_image_time_top < self.sampling_time:
                return
        self._last_image_time_top = current_time
        if self.camera_top_callback is not None:
            self.camera_top_callback(image)

    def move_vehicle_left(self, distance):
        """
        Moves the vehicle a specified distance to the left.

        :param distance: Distance to move the vehicle in meters (negative for left).
        """
        if self.vehicle is not None:
            current_transform = self.vehicle.get_transform()
            current_location = current_transform.location
            current_location.y -= distance
            new_transform = carla.Transform(
                location=current_location,
                rotation=current_transform.rotation
            )
            self.vehicle.set_transform(new_transform)
            print(f'Vehicle moved {distance} meters to the left.')
        else:
            print('Vehicle is not spawned yet.')

    def stop(self):
        """
        Frees up resources.
        """
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
            self.camera = None
            print('Camera destroyed')
        if self.camera_top is not None:
            self.camera_top.stop()
            self.camera_top.destroy()
            self.camera_top = None
            print('Top camera destroyed')
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None
            print('Vehicle destroyed')

    def __del__(self):
        """
        Destructor - ensures resources are freed.
        """
        self.stop()
