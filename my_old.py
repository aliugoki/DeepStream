import sys
import math
import traceback
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from utils.new_probe import *
from utils.parser_cfg import *
from utils.bus_call import bus_call
import os
from datetime import datetime
import json

def log_error(message):
    sys.stderr.write(f"ERROR: {message}\n")
    sys.stderr.flush()

def log_info(message):
    print(f"INFO: {message}")
    sys.stdout.flush()
def get_dynamic_sources():
    """Get RTSP sources dynamically from user input"""
    print("\n=== Dynamic Source Input ===")
    print("Enter RTSP sources (one per line). Press Enter on an empty line to finish:")

    sources = {}
    index = 0

    while True:
        url = input(f"Enter RTSP URL for source {index} (or leave empty to finish): ").strip()
        if not url:
            if index == 0:
                print("At least one source is required!")
                continue
            break

        name = input(f"Enter display name for source {index} [optional]: ").strip()
        if not name:
            name = f"Camera {index}"

        # Add RTSP parameters if not already present
        if url.startswith("rtsp://") and "?" not in url:
            url += "?latency=0&drop-on-latency=true&buffer-mode=auto&rtsp-transport=tcp"

        sources[f"source_{index}"] = {
            "uri": url,
            "name": name
        }
        index += 1

    return sources

def print_pad_info(element, name):
    try:
        pads = element.pads
        log_info(f"{name} pads:")
        for pad in pads:
            caps = pad.get_current_caps()
            caps_str = caps.to_string() if caps else "None"
            log_info(f"  {pad.name}: {pad.direction} (caps: {caps_str})")
    except Exception as e:
        log_error(f"Error getting pad info for {name}: {str(e)}")

def cb_newpad(decodebin, decoder_src_pad, data):
    try:
        log_info("Entered cb_newpad callback")
        caps = decoder_src_pad.get_current_caps()
        if not caps:
            log_info("No current caps, querying caps...")
            caps = decoder_src_pad.query_caps()

        log_info(f"Pad caps: {caps.to_string()}")
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        source_bin = data
        features = caps.get_features(0)

        log_info(f"Structure name: {gstname}")
        if "video" in gstname.lower():
            log_info("Found video pad")
            if features.contains("memory:NVMM"):
                log_info("NVMM memory feature found")
                bin_ghost_pad = source_bin.get_static_pad("src")
                if not bin_ghost_pad:
                    log_error("Failed to get source bin ghost pad")
                    return

                if not bin_ghost_pad.set_target(decoder_src_pad):
                    log_error("Failed to link decoder src pad to source bin ghost pad")
                else:
                    log_info("Successfully linked decoder src pad to source bin ghost pad")
            else:
                log_error("Decodebin did not pick nvidia decoder plugin - no NVMM memory feature")
        else:
            log_info(f"Ignoring non-video pad: {gstname}")
    except Exception as e:
        log_error(f"Exception in cb_newpad: {str(e)}\n{traceback.format_exc()}")

def decodebin_child_added(child_proxy, Object, name, user_data):
    try:
        log_info(f"Decodebin child added: {name}")
        if "decodebin" in name:
            log_info("Connecting child-added signal")
            Object.connect("child-added", decodebin_child_added, user_data)

        if "source" in name:
            log_info("Found source element")
            source_element = child_proxy.get_by_name("source")
            if source_element.find_property('drop-on-latency') is not None:
                log_info("Setting drop-on-latency property")
                Object.set_property("drop-on-latency", True)
    except Exception as e:
        log_error(f"Exception in decodebin_child_added: {str(e)}\n{traceback.format_exc()}")
def create_source_bin(index, uri):
    try:
        log_info(f"Creating source bin {index} for URI: {uri}")
        bin_name = f"source-bin-{index:02d}"
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            log_error(f"Unable to create source bin {index}")
            return None

        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{index}")
        if not uri_decode_bin:
            log_error(f"Unable to create uri decode bin {index}")
            return None

        log_info(f"Setting URI: {uri}")
        uri_decode_bin.set_property("uri", uri)

        log_info("Connecting signals...")
        uri_decode_bin.connect("pad-added", cb_newpad, nbin)
        uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

        log_info("Adding elements to bin...")
        Gst.Bin.add(nbin, uri_decode_bin)

        log_info("Creating ghost pad...")
        bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
        if not bin_pad:
            log_error("Failed to add ghost pad in source bin")
            return None

        log_info(f"Successfully created source bin {index}")
        return nbin

    except Exception as e:
        log_error(f"Exception in create_source_bin: {str(e)}\n{traceback.format_exc()}")
        return None
def handle_check_in(person_id, date, time):
    """Record check-in time for a person"""
    c = db_conn.cursor()

    # Check if there's an open session (checked in but not out)
    c.execute('''SELECT * FROM attendance
                 WHERE person_id=? AND date=? AND check_out IS NULL
                 ORDER BY check_in DESC LIMIT 1''',
              (person_id, date))
    existing = c.fetchone()

    if existing:
        log_info(f"Person {person_id} already checked in at {existing[2]} - skipping duplicate")
        return

    # Record new check-in
    c.execute('''INSERT INTO attendance (person_id, date, check_in)
                 VALUES (?, ?, ?)''',
              (person_id, date, f"{date} {time}"))
    db_conn.commit()
    log_info(f"Recorded check-in for {person_id} at {time}")

def handle_check_out(person_id, date, time):
    """Record check-out time for a person and calculate duration"""
    c = db_conn.cursor()

    # Find the most recent check-in without check-out
    c.execute('''SELECT rowid, check_in FROM attendance
                 WHERE person_id=? AND date=? AND check_out IS NULL
                 ORDER BY check_in DESC LIMIT 1''',
              (person_id, date))
    record = c.fetchone()

    if not record:
        log_info(f"No open session found for {person_id} - can't check out")
        return

    rowid, check_in = record
    check_in_time = datetime.strptime(check_in, "%Y-%m-%d %H:%M:%S")
    check_out_time = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")
    duration = int((check_out_time - check_in_time).total_seconds())

    # Update the record
    c.execute('''UPDATE attendance
                 SET check_out=?, duration=?
                 WHERE rowid=?''',
              (f"{date} {time}", duration, rowid))
    db_conn.commit()
    log_info(f"Recorded check-out for {person_id} at {time}, duration: {duration} seconds")

def generate_daily_report(date=None):
    """Generate a report of time spent per person per day"""
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    c = db_conn.cursor()

    # Get total time spent per person for the day
    c.execute('''SELECT person_id, SUM(duration)
                 FROM attendance
                 WHERE date=? AND duration IS NOT NULL
                 GROUP BY person_id''', (date,))

    report = {}
    for row in c.fetchall():
        person_id, total_seconds = row
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        report[person_id] = f"{hours}h {minutes}m {seconds}s"

    return report
def link_elements_with_checks(pipeline, element1, element2, name1, name2):
    try:
        log_info(f"Attempting to link {name1} to {name2}")

        if not element1 or not element2:
            log_error(f"Cannot link - {'element1' if not element1 else 'element2'} is None")
            return False

        src_pad = element1.get_static_pad("src")
        if not src_pad:
            log_error(f"No src pad found on {name1}")
            print_pad_info(element1, name1)
            return False

        sink_pad = element2.get_static_pad("sink")
        if not sink_pad:
            log_error(f"No sink pad found on {name2}")
            print_pad_info(element2, name2)
            return False

        log_info(f"Linking {name1} src pad to {name2} sink pad")
        link_result = src_pad.link(sink_pad)

        if link_result != Gst.PadLinkReturn.OK:
            log_error(f"Failed to link {name1} to {name2}: {link_result}")

            log_info(f"{name1} src pad info:")
            log_info(f"  Caps: {src_pad.get_current_caps()}")
            log_info(f"  Direction: {src_pad.direction}")

            log_info(f"{name2} sink pad info:")
            log_info(f"  Caps: {sink_pad.get_current_caps()}")
            log_info(f"  Direction: {sink_pad.direction}")

            return False
        else:
            log_info(f"Successfully linked {name1} to {name2}")
            return True

    except Exception as e:
        log_error(f"Exception linking {name1} to {name2}: {str(e)}\n{traceback.format_exc()}")
        return False


def main():
    try:
        config_path = "web_config.json"
        # if not os.path.exists(config_path):
        #     log_error(f"Configuration file not found at {config_path}")
        #     log_error("Please configure sources using the web interface first")
        #     return

        log_info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        # Validate configuration structure
        #if 'source' not in cfg:
        if 'sources' not in cfg or ('entrance' not in cfg['sources'] and 'exit' not in cfg['sources']):
            log_error("Invalid configuration: 'sources' key not found with 'entrance' or 'exit' keys")
            return

        dynamic_sources = {}
        for camera_type in ['entrance', 'exit']:
            for source_id, source_info in cfg['sources'].get(camera_type, {}).items():
                source_info['type'] = camera_type
                dynamic_sources[source_id] = source_info

        if not dynamic_sources:
            log_error("No sources configured in the configuration file")
            return
        log_info("Loading base configuration...")
        cfg = parse_args(cfg_path="config/config_pipeline.toml")

        # Get dynamic sources from user input
        #dynamic_sources = get_dynamic_sources()
        #cfg['source'] = dynamic_sources  # Replace the config sources with dynamic ones
        if 'pipeline' not in cfg:
            cfg['pipeline'] = {
                'known_face_dir': '/workspace/data/known_faces',
                'save_feature': True,
                'display': True,
                'is_aarch64': False,
                'enable_rtsp_streaming': True
            }

        log_info("Starting pipeline initialization")
        log_info(f"Number of sources: {len(dynamic_sources)}")


        faces_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recognized_faces")
        os.makedirs(faces_dir, exist_ok=True)
        log_info(f"Face images will be saved to: {faces_dir}")

        log_info("Loading known faces features...")
        try:
            known_face_features = load_faces(cfg['pipeline']['known_face_dir'])
            log_info(f"Loaded {len(known_face_features)} known face features")
        except Exception as e:
            log_error(f"Failed to load known faces: {str(e)}\n{traceback.format_exc()}")
            return

        #save_feature = cfg['pipeline']['save_feature']
        #save_path = None
        #if save_feature:
            #try:
                #save_path = cfg['pipeline']['save_feature_path']
                #log_info(f"Features will be saved to: {save_path}")
            #except Exception as e:
                #log_info(f"No save path specified or error getting path: {str(e)}")

        number_sources = len(dynamic_sources)
        log_info(f"Number of sources: {number_sources}")

        log_info("Initializing GStreamer...")
        Gst.init(None)

        log_info("Creating Pipeline...")
        pipeline = Gst.Pipeline()
        if not pipeline:
            log_error("Unable to create Pipeline")
            return

        log_info("Creating streammux...")
        streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        if not streammux:
            log_error("Unable to create NvStreamMux")
            return
        pipeline.add(streammux)
        source_idx = 0
        is_live = any(src['uri'].startswith('rtsp://') for src in dynamic_sources.values())

        for k, source_info in dynamic_sources.items():
            log_info(f"Processing source {k}: {source_info}")
            uri = source_info.get("uri", "")

            log_info(f"Creating source_bin {source_idx}")
            source_bin = create_source_bin(source_idx, uri)
            if not source_bin:
                log_error(f"Unable to create source bin {source_idx}")
                continue
            pipeline.add(source_bin)
            padname = f"sink_{source_idx}"

            log_info(f"Getting request pad {padname}")
            try:
                sinkpad = streammux.get_request_pad(padname)
                if not sinkpad:
                    log_error(f"Unable to create sink pad {padname}")
                    continue
                srcpad = source_bin.get_static_pad("src")
                if not srcpad:
                    log_error(f"Unable to create src pad for source {source_idx}")
                    continue
                log_info(f"Linking pads for source {source_idx}")
                if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
                    log_error(f"Failed to link pads for source {source_idx}")
                else:
                    log_info(f"Successfully linked pads for source {source_idx}")
            except Exception as e:
                log_error(f"Exception linking source {source_idx}: {str(e)}\n{traceback.format_exc()}")
            source_idx += 1

        # Create queues
        log_info("Creating queues...")
        queues = []
        for i in range(1, 8):
            queue = Gst.ElementFactory.make("queue", f"queue{i}")
            if not queue:
                log_error(f"Unable to create queue{i}")
                return

            queue.set_property("max-size-buffers", 10)
            queue.set_property("max-size-bytes", 0)
            queue.set_property("max-size-time", 0)
            queue.set_property("leaky", 2)
            pipeline.add(queue)
            queues.append(queue)
            log_info(f"Created queue{i}")


        log_info("Creating Primary PGIE for person detection...")
        pgie_person = Gst.ElementFactory.make("nvinfer", "primary-person-inference")
        if not pgie_person:
            log_error("Unable to create person PGIE")
            return

        log_info("Creating Pgie...")
        pgie_face = Gst.ElementFactory.make("nvinfer", "primary-face-inference")
        if not pgie_face:
            log_error("Unable to create pgie")
            return

        log_info("Creating Sgie...")
        sgie = Gst.ElementFactory.make("nvinfer", "secondary-inference")
        if not sgie:
            log_error("Unable to create sgie")
            return

        log_info("Creating Tracker...")
        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if not tracker:
            log_error("Unable to create tracker")
            return

        log_info("Creating tiler...")
        tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        if not tiler:
            log_error("Unable to create tiler")
            return

        log_info("Creating nvvidconv...")
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        if not nvvidconv:
            log_error("Unable to create nvvidconv")
            return

        log_info("Creating nvosd...")
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if not nvosd:
            log_error("Unable to create nvosd")
            return

        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "converter")
        caps = Gst.ElementFactory.make("capsfilter", "capsfilter")
        caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

        if not cfg['pipeline']['display']:
            log_info("Creating Fakesink...")
            sink = Gst.ElementFactory.make("fakesink", "fakesink")
            sink.set_property('enable-last-sample', 0)
            sink.set_property('sync', 0)
        #else:
        #    if cfg['pipeline']['is_aarch64']:
        #        log_info("Creating nv3dsink...")
        #        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        else:
            log_info("Creating EGLSink...")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")

        if not sink:
            log_error("Unable to create sink element")
            return
        if is_live:
            log_info("At least one of the sources is live")
            streammux.set_property('live-source', 1)
            streammux.set_property('batch-size', len(dynamic_sources))
            streammux.set_property('batched-push-timeout', 40000)
            streammux.set_property('max-latency', 1000000000)

        # Set properties from config (you might want to add more user input for these)
        log_info("Setting streammux properties...")
        streammux.set_property('width', 1280)
        streammux.set_property('height', 720)
        streammux.set_property('batched-push-timeout', 40000)
        
        log_info("Setting pgie_person properties...")
        pgie_person.set_property('config-file-path', 'config/config_person.txt')

        log_info("Setting pgie properties...")
        pgie_face.set_property('config-file-path', 'config/config_yolo.txt')

        log_info("Setting sgie properties...")
        sgie.set_property('config-file-path', 'config/config_arcface.txt')

        log_info("Setting nvosd properties...")
        nvosd.set_property('process-mode', 0)
        nvosd.set_property('display-text', 1)
        nvosd.set_property('display-clock', 1)

        log_info("Setting tiler properties...")
        tiler_rows = int(math.sqrt(number_sources))
        tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
        log_info(f"Setting tiler rows: {tiler_rows}, columns: {tiler_columns}")
        tiler.set_property("rows", tiler_rows)
        tiler.set_property("columns", tiler_columns)
        tiler.set_property("width", 1280)
        tiler.set_property("height", 720)

        log_info("Setting tracker properties...")
        set_tracker_properties(tracker, cfg['tracker']['config-file-path'])

        log_info("Adding elements to Pipeline...")
        elements_to_add = [pgie_face, sgie, tracker, tiler, nvvidconv, nvosd, caps, sink]
        for element in elements_to_add:
            pipeline.add(element)
            log_info(f"Added {element.name} to pipeline")

        log_info("Linking elements in the Pipeline...")
        elements = [streammux] + queues + [pgie_face, tracker, sgie, tiler, nvvidconv, nvosd, sink]

        # Link elements with verification
        for i in range(len(elements)-1):
            if not link_elements_with_checks(pipeline, elements[i], elements[i+1], elements[i].name, elements[i+1].name):
                log_error("Pipeline linking failed")
                return

        log_info("All elements linked successfully")

        log_info("Attaching probes...")
        try:
            # pgie_person_src_pad = pgie_person.get_static_pad("src")
            # if pgie_person_src_pad:
            #     log_info("Adding probe to pgie_person src pad")
            #     pgie_person_src_pad.add_probe(Gst.PadProbeType.BUFFER,pgie_person_src_pad_buffer_probe, {'sources':dynamic_sources} )
            # else:
            #     log_error("Could not get pgie_face src pad")



            caps_sink_pad = caps.get_static_pad("sink")
            if caps_sink_pad:
                log_info("Adding probe to pgie src pad")
                caps_sink_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_filter_probe, 0)
            else:
                log_error("Could not get pgie src pad")

            sgie_src_pad = sgie.get_static_pad("src")
            if sgie_src_pad:
                log_info("Adding probe to sgie src pad")
                data = {
                    'known_face_features': known_face_features,
                    #'save_feature': save_feature,
                    #'save_path': save_path,
                    'sources': dynamic_sources
                }
                sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, sgie_feature_extract_probe, data)
            else:
                log_error("Could not get sgie src pad")
        except Exception as e:
            log_error(f"Exception while attaching probes: {str(e)}\n{traceback.format_exc()}")

        log_info("Starting pipeline...")
        pipeline.set_state(Gst.State.PLAYING)
        log_info("Pipeline state set to PLAYING")

        loop = GLib.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, loop)

        log_info("Entering main loop...")
        try:
            loop.run()
        except KeyboardInterrupt:
            log_info("Received keyboard interrupt")
        except Exception as e:
            log_error(f"Exception in main loop: {str(e)}\n{traceback.format_exc()}")
        finally:
            log_info("Stopping pipeline...")
            pipeline.set_state(Gst.State.NULL)
            log_info("Pipeline stopped")

    except Exception as e:
        log_error(f"Exception in main: {str(e)}\n{traceback.format_exc()}")
        if 'pipeline' in locals():
            pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    try:
        log_info("Starting main execution...")
        main()
    except Exception as e:
        log_error(f"Exception in __main__: {str(e)}\n{traceback.format_exc()}")
