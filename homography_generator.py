import numpy as np
import cv2
import carla
import json
import math
import time

def initialize_carla_client(host, port, map_name):
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    if f"/Game/Carla/Maps/{map_name}" not in client.get_available_maps():
        raise ValueError(f"Map {map_name} is not available.")
    world = client.load_world(map_name)
    return client, world

def spawn_vehicle_and_camera(world, vehicle_blueprint, camera_blueprint, camera_transform):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find(vehicle_blueprint)
    spawn_point = world.get_map().get_spawn_points()[1]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    camera_bp = blueprint_library.find(camera_blueprint)
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')

    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    return vehicle, camera

def compute_ipm_transformation(image_points, camera_points):
    print("Punkty na obrazie (po transformacji):", image_points)
    print("Punkty w układzie kamery:", camera_points)
    image_points = np.array(image_points, dtype=np.float32)
    camera_points = np.array(camera_points, dtype=np.float32)
    transformation_matrix = cv2.getPerspectiveTransform(image_points, camera_points)
    return transformation_matrix

def save_resized_image(image, save_path):
    cv2.imwrite(save_path, image)

def save_transformation_matrix(matrix, file_path):
    with open(file_path, 'w') as file:
        json.dump(matrix.tolist(), file)

def camera_points_to_world_points(camera_transform, camera_points):
    world_points = []
    yaw_rad = math.radians(camera_transform.rotation.yaw)
    for (x_right, y_up) in camera_points:
        X_forward = y_up
        Y_left = x_right
        world_x = camera_transform.location.x + X_forward * math.cos(yaw_rad) - Y_left * math.sin(yaw_rad)
        world_y = camera_transform.location.y + X_forward * math.sin(yaw_rad) + Y_left * math.cos(yaw_rad)
        world_points.append((world_x, world_y))
    return world_points

def reorder_points(points):
    if len(points) != 4:
        raise ValueError("Input must be a list of exactly 4 points.")

    points = sorted(points, key=lambda p: p[1])

    top_points = sorted(points[:2], key=lambda p: p[0]) 
    bottom_points = sorted(points[2:], key=lambda p: p[0]) 

    return [top_points[0], top_points[1], bottom_points[0], bottom_points[1]]

def main_generate_image_and_matrix(host, port, map_name, vehicle_blueprint, camera_blueprint, camera_transform, save_image_path, save_matrix_path):
    client, world = initialize_carla_client(host, port, map_name)
    vehicle, camera = spawn_vehicle_and_camera(world, vehicle_blueprint, camera_blueprint, camera_transform)

    for _ in range(5):
        world.tick()
        time.sleep(0.1)

    actual_camera_transform = camera.get_transform()

    camera_points = [(-5.0, 30.0), (-3.5, 4.0), (3.5, 4.0), (5.0, 30.0)]

    world_points_2d = camera_points_to_world_points(actual_camera_transform, camera_points)
    wz = world.get_map().get_spawn_points()[1].location.z + 0.1
    for (wx, wy) in world_points_2d:
        world.debug.draw_point(
            carla.Location(x=wx, y=wy, z=wz),
            size=0.01,
            color=carla.Color(r=255, g=0, b=0),
            life_time=60.0
        )

    for _ in range(10):
        world.tick()
        time.sleep(0.1)

    image = None
    def capture_callback(data):
        nonlocal image
        image = np.array(data.raw_data).reshape((data.height, data.width, 4))[:, :, :3]

    camera.listen(capture_callback)
    while image is None:
        world.tick()
        time.sleep(0.05)
    camera.stop()

    save_resized_image(image, save_image_path)

    
    print("Na obrazie widoczne są czerwone punkty. Zaznacz je w kolejności: lewy górny, prawy górny, lewy dolny, prawy dolny.")
    for i, cp in enumerate(camera_points, start=1):
        print(f"Punkt {i}: {cp} (x w prawo, y w górę)")

    marked_image_points = []
    temp_image = image.copy()
    point_radius = 3
    current_point_index = 0
    num_points = len(camera_points)

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_point_index
        if event == cv2.EVENT_LBUTTONDOWN and current_point_index < num_points:
            marked_image_points.append((x, y))
            cv2.circle(temp_image, (x, y), point_radius, (0, 0, 255), -1)
            cv2.putText(temp_image, str(current_point_index+1), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            current_point_index += 1

    cv2.namedWindow('Calibration')
    cv2.setMouseCallback('Calibration', mouse_callback)
    print("Kliknij kolejno wszystkie 4 punkty. Następnie naciśnij 'q' aby zakończyć.")

    while True:
        cv2.imshow('Calibration', temp_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and len(marked_image_points) == num_points:
            break

    cv2.destroyAllWindows()

    if len(marked_image_points) != num_points:
        raise ValueError("Nie wybrano wszystkich punktów w odpowiedniej liczbie.")

    transformed_image_points = []
    for (xp, yp) in marked_image_points:
        x_prime = xp - 400
        y_prime = 600 - yp
        transformed_image_points.append((x_prime, y_prime))

    transformation_matrix = compute_ipm_transformation(transformed_image_points, [(0.0, 0.0), (0.0, 600.0), (800.0, 600.0), (800.0, 0.0)])
    save_transformation_matrix(transformation_matrix, save_matrix_path)

    print("Macierz transformacji została zapisana. Układ współrzędnych: x w prawo, y w górę.")
    print("Aby poprawić jakość macierzy, wybierz punkty tworzące większy i bardziej równomierny obszar (np. większy czworokąt).")

if __name__ == "__main__":
    host = '127.0.0.1'
    port = 2000
    map_name = 'Town04'
    vehicle_blueprint = 'vehicle.tesla.model3'
    camera_blueprint = 'sensor.camera.rgb'
    camera_transform = carla.Transform(
        carla.Location(x=1.0, y=0.0, z=2.0),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
    )

    save_image_path = 'resized_image.jpg'
    save_matrix_path = 'transformation_matrix.json'

    main_generate_image_and_matrix(host, port, map_name, vehicle_blueprint, camera_blueprint, camera_transform, save_image_path, save_matrix_path)
