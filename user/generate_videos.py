"""Generate videos of NuScene object instances

Example usage: python3 generate_videos.py --dataroot ~/repositories/nuscenes-devkit/data/sets/v1.0-mini --version v1.0-mini -o ~/repositories/nuscenes-devkit/output -f 4 -s 150 150 2
"""

import argparse
from collections import defaultdict
import json
import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import pathlib
from shutil import rmtree

def convert_annotation_list_to_dict(annotation_list, categories=None, visibilities=['', '1', '2', '3', '4']):
  """
  When saving the list of annotations to a dictionary, special attention must be paid to the
  correct keys to use. 

  For example, you will have bounding boxes with the same instance_token and sample_annotation_token
  because there are multiple cameras on the car, so you can have the same object appearing across
  multiple sensors. Each sensor's data is identified with a sample_data_token. 
  {'attribute_tokens': ['58aa28b1c2a54dc88e169808c07331e3'], 'bbox_corners': [1370.3079971217335, 446.66394956158524, 1600.0, 607.4567037983365], 'category_name': 'vehicle.car', 'filename': 'samples/CAM_FRONT/n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385095912404.jpg', 'instance_token': '0f8696c5e7284236b29a806d3d6f3513', 'next': '624a662244a241529e9f4d42fe75d2bd', 'num_lidar_pts': 4, 'num_radar_pts': 2, 'prev': '8291db1bc2704230867275bad5f42297', 'sample_annotation_token': 'ee04de72a30e4517a366ddad89d64fef', 'sample_data_token': '60ade2dececb46c69b114ce4c8a0bd3e', 'visibility_token': '1'}
  {'attribute_tokens': ['58aa28b1c2a54dc88e169808c07331e3'], 'bbox_corners': [0.0, 446.3944232196225, 387.13952090477727, 618.0310593208171], 'category_name': 'vehicle.car', 'filename': 'samples/CAM_FRONT_RIGHT/n008-2018-08-27-11-48-51-0400__CAM_FRONT_RIGHT__1535385095920482.jpg', 'instance_token': '0f8696c5e7284236b29a806d3d6f3513', 'next': '624a662244a241529e9f4d42fe75d2bd', 'num_lidar_pts': 4, 'num_radar_pts': 2, 'prev': '8291db1bc2704230867275bad5f42297', 'sample_annotation_token': 'ee04de72a30e4517a366ddad89d64fef', 'sample_data_token': '92d49452e5804d0a9724ab4161a26147', 'visibility_token': '1'}

  A combination of [instance_token][sample_data_token] can be used to uniquely identify
  the bounding boxes. You can enumerate through [instance_token][x] to find all the different
  views of a single bounding box.
  """
  # Convert the list of instance to a dictionary that uses the
  # instance_token -> sample_annotation_token -> camera
  # to look up the instance
  bbox_2d_annotations = defaultdict(lambda: defaultdict(dict))

  num_dups = 0
  for instance in annotation_list:
    instance_token = instance['instance_token']

    # 3. `sample` - An annotated snapshot of a scene at a particular timestamp.
    #               This is identified by `sample_annotation_token`.
    # 4. `sample_data` - Data collected from a particular sensor.

    # sample_data refers to the picture captured by a single sensor at a single timestamp.
    # sample_annotation_token refers to a single bounding box, which might exist in multiple
    # sample_data (across the different cameras)
    # sample_token = instance['sample_annotation_token']
    sample_token = instance['sample_annotation_token']
    category = instance['category_name']
    visibility = instance['visibility_token']
    camera_name = extract_camera_key_from_filename(instance['filename'])

    # Append additional information
    instance['camera_name'] = camera_name
    instance['bbox_area'] = calculate_bb_area(instance['bbox_corners'])

    if (categories is not None and category not in categories) or visibility not in visibilities:
      continue

    if instance_token in bbox_2d_annotations and sample_token in bbox_2d_annotations[instance_token] and camera_name in bbox_2d_annotations[instance_token][sample_token]:
      num_dups += 1
      print('Duplicate instance {}, sample {}, and camera {}'.format(
          instance_token, sample_token, camera_name))

    bbox_2d_annotations[instance_token][sample_token][camera_name] = instance

  print("Number of duplicates (should be zero)", num_dups)
  return bbox_2d_annotations


def extract_camera_key_from_filename(filename):
  """
  Parameters:
  - filename: the name of the file where the samples image is stored. 
              Ex: 'samples/CAM_BACK/n015-2018-10-02-10-50-40+0800__CAM_BACK__1538448750037525.jpg',
  """

  camera_name = filename.split('/')[1]

  # Validate the camera name is valid
  camera_names = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
                  'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
  assert(camera_name in camera_names), "Invalid camera name: {} from path: {}".format(
      camera_name, filename)

  return camera_name


def calculate_bb_area(bounding_box):
  """
  Calculates area of a 2D bounding box

  Parameters:
  - bounding_box: np.array of length 4 (x min, y min, x max, y max)
  """
  x_min, y_min, x_max, y_max = bounding_box
  return (x_max - x_min) * (y_max - y_min)


def get_most_visible_camera_annotation(camera_data_dict):
  """
  Parameters:
  - camera_data_dict: dictionary of form:
    {
      'CAM_BACK': {'attribute_tokens': ['cb5118da1ab342aa947717dc53544259'],
        'bbox_corners': [600.8315617945755,
        426.38901275036744,
        643.6756536789582,
        476.66593163100237],
        'category_name': 'vehicle.bus.rigid',
        'filename': 'samples/CAM_BACK/n015-2018-10-02-10-50-40+0800__CAM_BACK__1538448750037525.jpg',
        'instance_token': '9cba9cd8af85487fb010652c90d845b5',
        'next': 'ef90c2e525244b7d9eeb759837cf2277',
        'num_lidar_pts': 0,
        'num_radar_pts': 0,
        'prev': '6628e81912584a72bd448a44931afb42',
        'sample_annotation_token': '06b4886e79d2435c80bd23e7ac60c618',
        'sample_data_token': '0008443755a14b3ca483f1489c767040',
        'visibility_token': '4'},
      'CAM_FRONT': ...
      ...
    }
  """

  # Loop through all the camera views to find the best view of this instance
  # Each of the cameras will have a corresponding bounding box and visibility
  # we want the largest bounding box and highest visibility
  best_visibility = ''
  largest_area = -1
  best_camera_token = None

  for camera_token in camera_data_dict:
    visibility = camera_data_dict[camera_token]['visibility_token']
    bbox_area = camera_data_dict[camera_token]['bbox_area']

    if visibility > best_visibility or (visibility == best_visibility and bbox_area > largest_area):
      best_camera_token = camera_token
      largest_area = bbox_area
      best_visibility = visibility

  if not best_camera_token:
    print('Unable to find any good views for camera data dict: {}'.format(
        camera_data_dict))

  best_instance_data = camera_data_dict[best_camera_token]
  return best_instance_data


def get_cropped_image_for_annotation(sample_data_annotation, data_directory, output_size):
  """
  Parameters:
  - sample_data_annotation: of form:
    ```
    {'attribute_tokens': ['cb5118da1ab342aa947717dc53544259'],
    'bbox_corners': [600.8315617945755,
    426.38901275036744,
    643.6756536789582,
    476.66593163100237],
    'category_name': 'vehicle.bus.rigid',
    'filename': 'samples/CAM_BACK/n015-2018-10-02-10-50-40+0800__CAM_BACK__1538448750037525.jpg',
    'instance_token': '9cba9cd8af85487fb010652c90d845b5',
    'next': 'ef90c2e525244b7d9eeb759837cf2277',
    'num_lidar_pts': 0,
    'num_radar_pts': 0,
    'prev': '6628e81912584a72bd448a44931afb42',
    'sample_annotation_token': '06b4886e79d2435c80bd23e7ac60c618',
    'sample_data_token': '0008443755a14b3ca483f1489c767040',
    'visibility_token': '4'},
    ```
  """
  data_path = os.path.join(data_directory,
                           sample_data_annotation['filename'])
  bbox = sample_data_annotation['bbox_corners']
  im = Image.open(data_path)
  im1 = im.crop(bbox)
  im1 = im1.resize(output_size)
  np_img = np.asarray(im1)
  return np_img


def sort_sample_annotations_chronologically(instance_dict):
  """
  Parameters:
  - instance_dict: taken by indexing bbox_2d_annotations[instance_token]

  Uses ['bd26c2cdb22d4bb1834e808c89128898'][sample_annotation_token]['best_annotation'] 
  to find the correct sequence
  """

  # Find the first sample token
  first_sample_token = None

  for sample_token in instance_dict:
    if instance_dict[sample_token]['best_annotation']['prev'] == '':
      first_sample_token = sample_token
      break

  if first_sample_token is None:
    print("Unable to find a start token")

  # Now iterate and find a list of the sample_tokens in order
  sequential_sample_tokens = [first_sample_token]

  while True:
    try:
      next_sample_token = instance_dict[sequential_sample_tokens[-1]
                                        ]['best_annotation']['next']
    except:
      print("Unrecognized sample annotaton token: {}", sequential_sample_tokens)
      break

    if next_sample_token == '':
      break

    sequential_sample_tokens.append(next_sample_token)

  return sequential_sample_tokens


def remove_bad_samples(instance_annotation, minimum_bb_area, minimum_visibility, image_area=1600*900):
    """Removes bad samples from an instance annotation's sample sequence

    Args:
        instance_annotation (object): an instance annotation
        minimum_bb_area (float): The minimum fraction of a frame a bounding box take up to be used (0, 1)
        minimum_visibility (string): The minimum visibility a frame is allowed to haev ('', '1', '2', '3', '4')
        image_area (int, optional): The area of an image frame. Defaults to 1600*900.

    Returns: a cleaned list of sample annotation tokens that meet requirements
    """
    sample_token_sequence = instance_annotation['sample_annotation_sequence']
    cleaned = []

    for sample_token in sample_token_sequence:
        area = instance_annotation[sample_token]['best_annotation']['bbox_area']
        visibility = instance_annotation[sample_token]['best_annotation']['visibility_token']
        if area / image_area > minimum_bb_area and visibility >= minimum_visibility:
            cleaned.append(sample_token)

    return cleaned

def main(version, dataroot, output, object_categories, fps, output_size, minimum_frames, minimum_bb_area, visibility):
    """Generates video sequences of NuScene object instances over time.

    Expects the data to be organized as:

    ```
    "$dataroot"/
        samples	-	Sensor data for keyframes.
        sweeps	-	Sensor data for intermediate frames.
        maps	-	Folder for all map files: rasterized .png images and vectorized .json files.
        v1.0-*	-	JSON tables that include all the meta data and annotations. Each split (trainval, test, mini) is provided in a separate folder.
                    Note that image_annotations.json should be inside this directory.
    ```
    
    Args:
        version (string): the NuScenes data version
        dataroot (string): the path to the data root directory
        output (string): the path to the output video directory
        fps: frames per second to use for the video
        output_size (tuple): the output dimension to resize every cropped bounding box to. Defaults to (112, 112)
        minimum_frames (int): the minimum number of frames an instance must have
        minimum_bb_area (float): the minimum fraction of a frame a bounding box take up to be used (0, 1)
        visibility (string): the minimum visibility a frame is allowed to have ('', '1', '2', '3', '4')
    """
    print('='*20)
    print('Generating video sequences:')
    print('\t* Size: {}'.format(output_size))
    print('\t* FPS: {}'.format(fps))
    print('\t* Minimum frame count: {}'.format(minimum_frames))
    print('\t* Minimum BB area: {}'.format(minimum_bb_area))
    print('\t* Minimum visibility: {}'.format(visibility))

    # ================================Load image annotations ========================================
    with open(os.path.join(dataroot, version, 'image_annotations.json')) as json_file:
        # A list of dictionaries
        bbox_2d_annotations_list = json.load(json_file)

    # These can be indexed with [instance_token][sample_annotation_token][camera_name] -> data about the annotation
    # You can use the sample_annotation_token with the nuscenes helper in order to get
    # the sample tokens
    bbox_2d_annotations = convert_annotation_list_to_dict(
        bbox_2d_annotations_list, categories=object_categories)
    print('Number of unique vehicle instances: {}'.format(len(bbox_2d_annotations)))
    # ==============================================================================================


    #  ===== For each instance and each sample annotation, find the best camera sensor to use ======
    # Get sorted sample annotation tokens per instance per camera
    for instance_token in bbox_2d_annotations:
        for sample_annotation_token in bbox_2d_annotations[instance_token]:
            bbox_2d_annotations[instance_token][sample_annotation_token]['best_annotation'] = get_most_visible_camera_annotation(
                bbox_2d_annotations[instance_token][sample_annotation_token])
    # ==============================================================================================


    # ====== For each instance, find the correct sequence of sample annotations ====================
    # Get sorted sample annotation tokens per instance per camera
    for instance_token in bbox_2d_annotations:
        bbox_2d_annotations[instance_token]['sample_annotation_sequence'] = sort_sample_annotations_chronologically(
            bbox_2d_annotations[instance_token])
    # ==============================================================================================

    # ====== Remove samples from sequence that don't meet requirements ====================
    for instance_token in bbox_2d_annotations:
        bbox_2d_annotations[instance_token]['sample_annotation_sequence'] = remove_bad_samples(
            bbox_2d_annotations[instance_token], minimum_bb_area, visibility)
    # ==============================================================================================

    # ====== Create videos for every instance ======================================================

    # Remove the directory if it already exists and create new one
    rmtree(output, ignore_errors=True)
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    print("Creating videos and storing in '{}'...".format(output))
    total_videos = 0
    for instance_token in tqdm(bbox_2d_annotations):
        sample_annotation_tokens = bbox_2d_annotations[instance_token]['sample_annotation_sequence']

        if len(sample_annotation_tokens) < minimum_frames:
            continue

        video_path = os.path.join(
            output, '{}.mp4'.format(instance_token))

        # Need to use vp09 to be able to upload to certain data annotation platforms
        out = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*'vp09'), fps, output_size)

        for sample_annotation_token in sample_annotation_tokens:
            best_annotation = bbox_2d_annotations[instance_token][sample_annotation_token]['best_annotation']
            cropped_img = get_cropped_image_for_annotation(
                best_annotation, dataroot, output_size)
            
            # Convert from PIL's RGB to cv2 BGR
            out.write(cropped_img[:, :, ::-1])

        out.release()

        total_videos += 1

    print('Created {} videos ({} did not meet requirements).'.format(
        total_videos, len(bbox_2d_annotations) - total_videos, minimum_frames))
    # ==============================================================================================
    print('='*20)


if __name__ == "__main__":
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataroot", type=str,
                    help="The path to the root directory where the data is stored")
    ap.add_argument("-v", "--version", type=str,
                    help="The NuScene's data version")
    ap.add_argument("-o", "--output", type=str,
                    help="The output video directory")
    ap.add_argument("-f", "--fps", type=int, default=2,
                    help="Frames per second for output video (use 2 to match speed of original data)")
    ap.add_argument("-m", "--minimum_frames", type=int, default=9,
                    help="The minimum number of frames an instance must have")
    ap.add_argument("-p", "--minimum_percentage", type=float, default=0.01,
                    help="The minimum fraction of a frame a bounding box take up to be used (0, 1)")
    ap.add_argument("--visibility", type=str, default='10',
                    help="The minimum visibility a frame is allowed ('', '1', '2', '3', '4')")
    ap.add_argument("-s", "--size", type=int, default=[1600, 900], nargs=2,
                    help="Size of the output video")

    # Excludes bicycle and motorcycle by default
    vehicle_categories = ['vehicle.bus.bendy', 'vehicle.bus.rigid',
                          'vehicle.car', 'vehicle.construction', 'vehicle.trailer', 'vehicle.truck']
    ap.add_argument("-c", "--categories", nargs='+',
                    help="The categories to extract videos for", required=False, default=vehicle_categories)

    args = vars(ap.parse_args())
    main(args['version'], args['dataroot'], args['output'], args['categories'],
         args['fps'], tuple(args['size']), args['minimum_frames'], args['minimum_percentage'], args["visibility"])