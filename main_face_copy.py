import sys
import math
import traceback
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from utils.probe import *
from utils.parser_cfg import *
from utils.bus_call import bus_call
import os

#Ali

def log_error(message):
    sys.stderr.write(f"ERROR: {message}\n")
    sys.stderr.flush()

def log_info(message):
    print(f"INFO: {message}")
    sys.stdout.flush()

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
def main(cfg):

    try:
        #import atexit
        #import signal

       # def handle_signal(sig, frame):
          #  debug_log(3, f"Received signal {sig}, initiating shutdown")
         #   cleanup()
        #    sys.exit(0)

       # signal.signal(signal.SIGINT, handle_signal)
       # signal.signal(signal.SIGTERM, handle_signal)
       # atexit.register(cleanup)
        # At the very end of your cleanup
        #import atexit
       # atexit.register(cleanup)
        log_info("Starting pipeline initialization")
        log_info(f"Configuration: {cfg}")
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

        save_feature = cfg['pipeline']['save_feature']
        save_path = None
        if save_feature:
            try:
                save_path = cfg['pipeline']['save_feature_path']
                log_info(f"Features will be saved to: {save_path}")
            except Exception as e:
                log_info(f"No save path specified or error getting path: {str(e)}")

        sources = cfg['source']
        number_sources = len(sources)
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
        is_live = False

        for k, v in sources.items():
            log_info(f"Processing source {k}: {v}")
            uri_name = v
            if uri_name.find("rtsp://") == 0:
                is_live = True
                if "?" not in uri_name:
                    uri_name += "?latency=0&drop-on-latency=true&buffer-mode=auto"
                log_info(f"Modified RTSP URI: {uri_name}")

            log_info(f"Creating source_bin {source_idx}")
            source_bin = create_source_bin(source_idx, uri_name)
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
            pipeline.add(queue)
            queues.append(queue)
            log_info(f"Created queue{i}")

        log_info("Creating Pgie...")
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if not pgie:
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

         # Create the recognized_faces directory
        #faces_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recognized_faces")
        #os.makedirs(faces_dir, exist_ok=True)
        #log_info(f"Face images will be saved to: {faces_dir}")

        # Attach the probe to nvosd sink pad
        #def osd_sink_pad_buffer_probe(pad, info, user_data):
            # [Insert the complete probe function from previous response here]
            pass

        #osd_sink_pad = nvosd.get_static_pad("sink")
        #if osd_sink_pad:
         #   osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, None)
          #  log_info("Added face saving probe to nvosd sink pad")
        #else:
         #   log_error("Could not get nvosd sink pad")
        #########################################################

      #
      #  if not cfg['pipeline']['display']:
      #      log_info("Creating Fakesink...")

        if not cfg['pipeline']['display']:
            log_info("Creating Fakesink...")
            sink = Gst.ElementFactory.make("fakesink", "fakesink")
            sink.set_property('enable-last-sample', 0)
            sink.set_property('sync', 0)
        else:
            if cfg['pipeline']['is_aarch64']:
                log_info("Creating nv3dsink...")
                sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
            else:
                log_info("Creating EGLSink...")
                sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")


        if cfg['pipeline']['enable_rtsp_streaming']:
            log_info("Creating RTSP streaming elements...")

#            # Create elements for RTSP streaming
#            rtsp_caps = Gst.ElementFactory.make("capsfilter", "rtsp_caps")
#            rtsp_caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
#
#            rtsp_conv = Gst.ElementFactory.make("nvvideoconvert", "rtsp_convert")
#            rtsp_conv2 = Gst.ElementFactory.make("videoconvert", "rtsp_convert2")
#            rtsp_enc = Gst.ElementFactory.make("x264enc", "rtsp_encoder")
#            rtsp_enc.set_property("bitrate", 2000)
#            rtsp_enc.set_property("speed-preset", "ultrafast")
#            rtsp_enc.set_property("tune", "zerolatency")
#
#            rtsp_rtppay = Gst.ElementFactory.make("rtph264pay", "rtsp_rtppay")
#            rtsp_rtppay.set_property("pt", 96)
#            rtsp_rtppay.set_property("config-interval", 1)
#
#            udpsink = Gst.ElementFactory.make("udpsink", "udpsink")
#            udpsink.set_property("host", "127.0.0.1")
#            udpsink.set_property("port", 5400)
#            udpsink.set_property("sync", 0)
#            udpsink.set_property("async", 0)
#
#            # Add all elements to pipeline
#            pipeline.add(rtsp_caps)
#            pipeline.add(rtsp_conv)
#            pipeline.add(rtsp_conv2)
#            pipeline.add(rtsp_enc)
#            pipeline.add(rtsp_rtppay)
#            pipeline.add(udpsink)
#
#            # Link elements
#            nvosd.link(rtsp_caps)
#            rtsp_caps.link(rtsp_conv)
#            rtsp_conv.link(rtsp_conv2)
#            rtsp_conv2.link(rtsp_enc)
#            rtsp_enc.link(rtsp_rtppay)
#            rtsp_rtppay.link(udpsink)
#
#            log_info("RTSP streaming elements created and linked")
#
        if not sink:
            log_error("Unable to create sink element")
            return

        if is_live:
            log_info("At least one of the sources is live")
            streammux.set_property('live-source', 1)


        # Set properties
        log_info("Setting streammux properties...")
        set_property(cfg, streammux, "streammux")

        log_info("Setting pgie properties...")
        set_property(cfg, pgie, "pgie")

        log_info("Setting sgie properties...")
        set_property(cfg, sgie, "sgie")

        log_info("Setting nvosd properties...")
        set_property(cfg, nvosd, "nvosd")

        log_info("Setting tiler properties...")
        set_property(cfg, tiler, "tiler")

        log_info("Setting sink properties...")
        set_property(cfg, sink, "sink")

        log_info("Setting tracker properties...")
        set_tracker_properties(tracker, cfg['tracker']['config-file-path'])

        tiler_rows = int(math.sqrt(number_sources))
        tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
        log_info(f"Setting tiler rows: {tiler_rows}, columns: {tiler_columns}")
        tiler.set_property("rows", tiler_rows)
        tiler.set_property("columns", tiler_columns)

        log_info("Adding elements to Pipeline...")
        elements_to_add = [pgie, sgie, tracker, tiler, nvvidconv, nvosd, sink]
        for element in elements_to_add:
            pipeline.add(element)
            log_info(f"Added {element.name} to pipeline")

        log_info("Linking elements in the Pipeline...")
        elements = [streammux] + queues + [pgie, tracker, sgie, tiler, nvvidconv, nvosd, sink]

        # Link elements with verification
        for i in range(len(elements)-1):
            if not link_elements_with_checks(pipeline, elements[i], elements[i+1], elements[i].name, elements[i+1].name):
                log_error("Pipeline linking failed")
                return

        log_info("All elements linked successfully")

        log_info("Attaching probes...")
        try:
            pgie_src_pad = pgie.get_static_pad("src")
            if pgie_src_pad:
                log_info("Adding probe to pgie src pad")
                pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_filter_probe, 0)
            else:
                log_error("Could not get pgie src pad")

            sgie_src_pad = sgie.get_static_pad("src")
            if sgie_src_pad:
                log_info("Adding probe to sgie src pad")
                data = [known_face_features, save_feature, save_path]
                sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, sgie_feature_extract_probe, data)
            else:
                log_error("Could not get sgie src pad")
        except Exception as e:
            log_error(f"Exception while attaching probes: {str(e)}\n{traceback.format_exc()}")

        log_info("Listing sources:")
        for key, value in sources.items():
            log_info(f"{key}: {value}")

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
        log_info("Loading configuration...")
        cfg = parse_args(cfg_path="config/config_pipeline.toml")
        log_info("Starting main execution...")
        main(cfg)
    except Exception as e:
        log_error(f"Exception in __main__: {str(e)}\n{traceback.format_exc()}")
