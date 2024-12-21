#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <gst/gst.h>
#include <glib.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <vector>

#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080
#define MUXER_BATCH_TIMEOUT_USEC 40000

#define TARGET_WIDTH 25
#define TARGET_HEIGHT 250

typedef struct {
    std::vector<double> y_coords;
} VehicleData;

typedef struct {
    cv::Mat m; // Perspective transformation matrix
} ViewTransformer;

void ViewTransformer_init(ViewTransformer *vt, const cv::Point2f source[4], const cv::Point2f target[4]) {
    cv::Mat srcMat(4, 1, CV_32FC2, (void*)source);
    cv::Mat tgtMat(4, 1, CV_32FC2, (void*)target);
    vt->m = cv::getPerspectiveTransform(srcMat, tgtMat);
}

void ViewTransformer_transform_points(ViewTransformer *vt, const cv::Point2f *points, cv::Point2f *transformed_points, int num_points) {
    if (num_points == 0) {
        return;
    }
    cv::Mat pointsMat(num_points, 1, CV_32FC2, (void*)points);
    cv::Mat transformedPointsMat(num_points, 1, CV_32FC2, (void*)transformed_points);
    cv::perspectiveTransform(pointsMat, transformedPointsMat, vt->m);
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    GMainLoop *loop = (GMainLoop *)data;
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_ERROR: {
            gchar *debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("ERROR from element %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
            if (debug)
                g_printerr("Error details: %s\n", debug);
            g_free(debug);
            g_error_free(error);
            g_main_loop_quit(loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

void cb_new_pad(GstElement *qtdemux, GstPad *pad, gpointer data) {
    GstElement *h264parser = (GstElement *)data;
    gchar *name = gst_pad_get_name(pad);
    if (strcmp(name, "video_0") == 0 && 
        !gst_element_link_pads(qtdemux, name, h264parser, "sink")) {
        g_printerr("Could not link %s pad of qtdemux to sink pad of h264parser", name);
    }
}

double calculate_speed(const std::vector<double>& y_coords, double fps) {
    if (y_coords.size() < fps / 2) {
        return -1; // Not enough data to calculate speed
    }
    double coordinate_start = y_coords.back();
    double coordinate_end = y_coords.front();
    double distance = fabs(coordinate_start - coordinate_end);
    //printf("coordinate start is %lf\n", coordinate_start);
    //printf("coordinate end is %lf\n", coordinate_end);
    double time = y_coords.size() / fps;
    //printf("time is %lf\n", time);
    return distance / time * 3.6; // Convert to km/h
}

static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    std::unordered_map<int, VehicleData>* vehicle_data_table = (std::unordered_map<int, VehicleData>*)u_data;

    // Assuming a constant frame rate of 30 FPS for simplicity
    guint fps = 25;

    // Define source and target points for perspective transformation
    cv::Point2f source_points[4] = { {1252, 787}, {2298, 803}, {5039, 2159}, {-550, 2159} };
    cv::Point2f target_points[4] = { {0, 0}, {TARGET_WIDTH - 1, 0}, {TARGET_WIDTH - 1, TARGET_HEIGHT - 1}, {0, TARGET_HEIGHT - 1} };

    // Initialize the ViewTransformer
    ViewTransformer vt;
    ViewTransformer_init(&vt, source_points, target_points);

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
            if (obj_meta->class_id == 0) { // Assuming class_id 0 is for vehicles
                // Get the center point of the bounding box
                double center_x = obj_meta->rect_params.left + obj_meta->rect_params.width / 2.0;
                //double center_y = obj_meta->rect_params.top + obj_meta->rect_params.height / 2.0;
                double y_min = obj_meta->rect_params.top;
                double y_max = obj_meta->rect_params.top + obj_meta->rect_params.height;
                double center_y = (y_max-y_min)/2;

                // Transform the center point using the perspective transformation
                cv::Point2f point(y_max, y_min);
                cv::Point2f transformed_point;
                ViewTransformer_transform_points(&vt, &point, &transformed_point, 1);

                // Retrieve or create vehicle data for the current vehicle
                int tracker_id = obj_meta->object_id;
                VehicleData& vehicle_data = (*vehicle_data_table)[tracker_id];

                // Store the transformed y coordinate
                vehicle_data.y_coords.push_back(center_y);

                // Calculate speed if enough coordinates are available
                double speed = calculate_speed(vehicle_data.y_coords, fps);
                if (speed != -1) {
                    g_print("Vehicle ID: %d, Speed: %f km/h\n", tracker_id, speed);
                }

                // Remove the oldest coordinate to keep the array size manageable
                if (vehicle_data.y_coords.size() > fps) {
                    vehicle_data.y_coords.erase(vehicle_data.y_coords.begin());
                }
            }
        }
    }

    return GST_PAD_PROBE_OK;
}

int main(int argc, char *argv[]) {
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL, *nvv4l2h264enc = NULL,
                *qtdemux = NULL, *nvv4l2decoder = NULL, *streammux = NULL, *sink = NULL,
                *nvvidconv = NULL, *qtmux = NULL, *pgie = NULL, *tracker = NULL,
                *nvvidconv2 = NULL, *nvosd = NULL, *h264parser2 = NULL;

    GstBus *bus = NULL;
    guint bus_watch_id;
    std::unordered_map<int, VehicleData> vehicle_data_table;

    if (argc != 2) {
        g_printerr("Usage: %s </path/to/input/video.mp4>\n", argv[0]);
        return -1;
    }

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    pipeline = gst_pipeline_new("deepstream_tutorial_app1_c++");
    source = gst_element_factory_make("filesrc", "file-source");
    qtdemux = gst_element_factory_make("qtdemux", "qtdemux");
    h264parser = gst_element_factory_make("h264parse", "h264-parser");
    nvv4l2decoder = gst_element_factory_make("nvv4l2decoder", "nvv4l2-decoder");
    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
    tracker = gst_element_factory_make("nvtracker", "tracker");
    nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
    nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
    nvvidconv2 = gst_element_factory_make("nvvideoconvert", "nvvideo-converter2");
    nvv4l2h264enc = gst_element_factory_make("nvv4l2h264enc", "nvv4l2h264enc");
    h264parser2 = gst_element_factory_make("h264parse", "h264parser2");
    qtmux = gst_element_factory_make("qtmux", "qtmux");
    sink = gst_element_factory_make("filesink", "filesink");

    if (!pipeline || !source || !h264parser || !qtdemux ||
        !nvv4l2decoder || !streammux || !pgie || !tracker || 
        !nvvidconv || !nvosd || !nvvidconv2 || !nvv4l2h264enc || 
        !h264parser2 || !qtmux || !sink) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

    g_object_set(G_OBJECT(source), "location", argv[1], NULL);
    g_object_set(G_OBJECT(streammux), "batch-size", 1,
                 "width", MUXER_OUTPUT_WIDTH, 
                 "height", MUXER_OUTPUT_HEIGHT,
                 "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
    g_object_set(G_OBJECT(pgie), "config-file-path", "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test1/dstest1_pgie_config.txt", NULL);
    g_object_set(G_OBJECT(tracker), "ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so", NULL);
    g_object_set(G_OBJECT(sink), "location", "vehicles_output.mp4", NULL);

    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    g_signal_connect(qtdemux, "pad-added", G_CALLBACK(cb_new_pad), h264parser);

    gst_bin_add_many(GST_BIN(pipeline), source, qtdemux, h264parser, nvv4l2decoder, streammux, pgie, tracker, nvvidconv, nvosd, nvvidconv2, nvv4l2h264enc, h264parser2, qtmux, sink, NULL);

    GstPad *sinkpad, *srcpad;
    gchar pad_name_sink[16] = "sink_0";
    gchar pad_name_src[16] = "src";

    sinkpad = gst_element_get_request_pad(streammux, pad_name_sink);
    if (!sinkpad) {
        g_printerr("Streammux request sink pad failed. Exiting.\n");
        return -1;
    }

    srcpad = gst_element_get_static_pad(nvv4l2decoder, pad_name_src);
    if (!srcpad) {
        g_printerr("Decoder request src pad failed. Exiting.\n");
        return -1;
    }

    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link decoder to stream muxer. Exiting.\n");
        return -1;
    }

    gst_object_unref(sinkpad);
    gst_object_unref(srcpad);

    if (!gst_element_link_many(source, qtdemux, NULL)) {
        g_printerr("Source and QTDemux could not be linked: 1. Exiting.\n");
        return -1;
    }

    if (!gst_element_link_many(h264parser, nvv4l2decoder, NULL)) {
        g_printerr("H264Parse and NvV4l2-Decoder could not be linked: 2. Exiting.\n");
        return -1;
    }

    if (!gst_element_link_many(streammux, pgie, tracker, nvvidconv, nvosd, nvvidconv2, nvv4l2h264enc, h264parser2, qtmux, sink, NULL)) {
        g_printerr("Rest of the pipeline elements could not be linked: 3. Exiting.\n");
        return -1;
    }

    GstPad *osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
    if (!osd_sink_pad) {
        g_printerr("Unable to get sink pad\n");
        return -1;
    }
    
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_sink_pad_buffer_probe, &vehicle_data_table, NULL);
    gst_object_unref(osd_sink_pad);

    g_print("Using file: %s\n", argv[1]);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    g_print("Running...\n");
    g_main_loop_run(loop);

    g_print("Returned, stopping playback\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    g_print("Deleting pipeline\n");
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);
    return 0;
}