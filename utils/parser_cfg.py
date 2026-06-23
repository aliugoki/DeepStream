import toml
import gi
import os
import configparser
import numpy as np # Keep this if you use numpy for load_faces

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# --- Logging Functions (Add these if not already in parser_cfg.py for better debugging) ---
import sys
def log_error(message):
    sys.stderr.write(f"ERROR: {message}\n")
    sys.stderr.flush()
def log_info(message):
    print(f"INFO: {message}")
    sys.stdout.flush()
# -----------------------------------------------------------------------------

def parse_args(cfg_path):
    """Parses configuration from a TOML file."""
    try:
        cfg = toml.load(cfg_path)
        return cfg
    except toml.TomlDecodeError as e:
        log_error(f"Found invalid character in key name: {e}")
        raise
    except FileNotFoundError:
        log_error(f"Config file not found at {cfg_path}")
        raise
    except Exception as e:
        log_error(f"Error parsing config file {cfg_path}: {e}")
        raise

def set_property(cfg, gst_element, section_name):
    """
    Sets properties on a GStreamer element from the config.
    Skips properties that are not found on the element.
    """
    if section_name in cfg:
        properties = cfg[section_name]
        for key, value in properties.items():
            try:
                # Special handling for streammux: skip fps-n and fps-d
                if section_name == "streammux" and key in ["fps-n", "fps-d"]:
                    log_info(f"Skipping streammux property '{key}' as it's not a direct element property.")
                    continue

                gst_element.set_property(key, value)
                log_info(f"{gst_element.get_name()} set_property {key} {value}")
            except TypeError as e:
                log_error(f"Failed to set property '{key}' for element '{gst_element.get_name()}': {e}")
                log_error(f"Known properties for {gst_element.get_name()}:")
                # Optional: Iterate through properties to list them for debugging
                # for prop in gst_element.list_properties():
                #     log_error(f"  - {prop.name}")
                raise # Re-raise to stop if a critical property can't be set
            except Exception as e:
                log_error(f"An unexpected error occurred setting property '{key}' for element '{gst_element.get_name()}': {e}")
                raise

def set_tracker_properties(tracker_element, config_file_path):
    """Sets properties for the nvtracker element."""
    config = configparser.ConfigParser()
    config.read(config_file_path)
    # config.sections() # This line is not needed

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker_element.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker_element.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker_element.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker_element.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker_element.set_property('ll-config-file', tracker_ll_config_file)
    
    # Remove or comment out this line as it causes TypeError for older DeepStream versions
    # tracker_element.set_property("enable-batch-process", 1) 
    log_info(f"Tracker properties set from {config_file_path}")


def load_faces(path):
    loaded_faces = {}
    file_list = os.listdir(path)
    for file in file_list:
        if file.endswith('.npy'):
            face_feature = np.load(os.path.join(path, file)).reshape(-1, 1)
            name = file.split('.')[0]
            loaded_faces[name] = face_feature
    return loaded_faces

# def load_faces(path):
#     loaded_faces = {}
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             if file.endswith('.npy'):
#                 try:
#                     face_feature = np.load(os.path.join(root, file)).reshape(1, -1)
#                     name = os.path.splitext(file)[0]
#                     loaded_faces[name] = face_feature
#                 except Exception as e:
#                     log_error(f"Failed to load {file}: {e}")
#     return loaded_faces
