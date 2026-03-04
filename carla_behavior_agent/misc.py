#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" Module with auxiliary functions. """

import math
import numpy as np
import carla

def get_speed(vehicle):
    """
    Compute speed of a vehicle in Km/h.

        :param vehicle: the vehicle for which speed is calculated
        :return: speed as a float in Km/h
    """
    vel = vehicle.get_velocity()

    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def get_acceleration(vehicle):
    """
    Compute acceleration of a vehicle in m/s^2 (km/h^2 = 0.27 m/s^2)

        :param vehicle: the vehicle for which acceleration is calculated
        :return: acceleration as a float in m/s^2
    """
    acc = vehicle.get_acceleration()

    return math.sqrt(acc.x ** 2 + acc.y ** 2 + acc.z ** 2)

def get_trafficlight_trigger_location(traffic_light):
    """
    Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
    """
    def rotate_point(point, radians):
        """
        rotate a given point by a given angle
        """
        rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
        rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

        return carla.Vector3D(rotated_x, rotated_y, point.z)

    base_transform = traffic_light.get_transform()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)
    area_ext = traffic_light.trigger_volume.extent

    point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
    point_location = area_loc + carla.Location(x=point.x, y=point.y)

    return carla.Location(point_location.x, point_location.y, point_location.z)


def is_within_distance(target_transform, reference_transform, max_distance, angle_interval=None):
    """
    Check if a location is both within a certain distance from a reference object.
    By using 'angle_interval', the angle between the location and reference transform
    will also be taken into account, being 0 a location in front and 180, one behind.

    :param target_transform: location of the target object
    :param reference_transform: location of the reference object
    :param max_distance: maximum allowed distance
    :param angle_interval: only locations between [min, max] angles will be considered. This isn't checked by default.
    :return: boolean
    """
    target_vector = np.array([
        target_transform.location.x - reference_transform.location.x,
        target_transform.location.y - reference_transform.location.y
    ])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    # Further than the max distance
    if norm_target > max_distance:
        return False

    # We don't care about the angle, nothing else to check
    if not angle_interval:
        return True

    min_angle = angle_interval[0]
    max_angle = angle_interval[1]

    fwd = reference_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return min_angle < angle < max_angle

def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2

        :param location_1, location_2: carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]

def compute_distance(location_1, location_2):
    """
    Euclidean distance between 3D points

        :param location_1, location_2: 3D points
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm

def compute_distance_from_center(actor1, actor2 = None, distance = 5):
    """
    Compute the distance between the center of two actors. 
    NOTE: We use the bounding boxes to calculate the actual distance.
    
        :param actor1: first actor (carla.Actor)
        :param actor2: second actor (carla.Actor)
        :param distance: distance between the two actors (float)
        
        :return: distance between the center of the two actors (float)
    """
    actor1_extent = max(actor1.bounding_box.extent.x, actor1.bounding_box.extent.y)
    actor2_extent = max(actor2.bounding_box.extent.x, actor2.bounding_box.extent.y) if actor2 else 0
    return distance - actor1_extent - actor2_extent

def get_distance(a, b):
        """
        Calculate the distance between two objects: a and b. These two objects can be of type carla.Vehicle, carla.Waypoint, or carla.Location.

        :param a: first object (carla.Actor, carla.Landmark, carla.Waypoint, or carla.Location)
        :param b: second object (carla.Actor, carla.Landmark, carla.Waypoint, or carla.Location)

        :return: distance between the two objects (float)
        """
        # Check if input 'a' is of type carla.Landmark and convert it to a carla.Waypoint object.
        if isinstance(a, carla.Landmark):
            a = a.waypoint
        # Check if input 'b' is of type carla.Landmark and convert it to a carla.Waypoint object.
        if isinstance(b, carla.Landmark):
            b = b.waypoint
            
        # Check if input 'a' is of type carla.Transform and convert it to a carla.Location object.
        if isinstance(a, carla.Transform):
            a = a.location
        # Check if input 'b' is of type carla.Transform and convert it to a carla.Location object.
        if isinstance(b, carla.Transform):
            b = b.location

        # Check if both input objects are of type carla.Actor.
        if isinstance(a, carla.Actor) and isinstance(b, carla.Actor):
            return a.get_location().distance(b.get_location())
        # Check if input 'a' is of type carla.Actor and input 'b' is of type carla.Waypoint.
        elif isinstance(a, carla.Actor) and isinstance(b, carla.Waypoint):
            return a.get_location().distance(b.transform.location)
        # Check if input 'a' is of type carla.Waypoint and input 'b' is of type carla.Actor.
        elif isinstance(a, carla.Waypoint) and isinstance(b, carla.Actor):
            return a.transform.location.distance(b.get_location())
        # Check if both input objects are of type carla.Waypoint.
        elif isinstance(a, carla.Waypoint) and isinstance(b, carla.Waypoint):
            return a.transform.location.distance(b.transform.location)
        # Check if input 'a' is of type carla.Location and input 'b' is of type carla.Location.
        elif isinstance(a, carla.Actor) and isinstance(b, carla.Location):
            return a.get_location().distance(b)
        elif isinstance(a, carla.Location) and isinstance(b, carla.Actor):
            return a.distance(b.get_location())
        elif isinstance(a, carla.Location) and isinstance(b, carla.Waypoint):
            return a.distance(b.transform.location)
        elif isinstance(a, carla.Waypoint) and isinstance(b, carla.Location):
            return a.transform.location.distance(b)
        elif isinstance(a, carla.Location) and isinstance(b, carla.Location):
            return a.distance(b)
        # If none of the above conditions are met, raise a ValueError.
        else:
            raise ValueError("Invalid input types. Please provide either carla.Actor, carla.Landmark, carla.Waypoint, or carla.Location objects.")