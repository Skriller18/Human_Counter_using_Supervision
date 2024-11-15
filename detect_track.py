import onnxruntime
import cv2
import numpy as np
from supervision.annotators import core
import supervision as sv
import time

queue= []
ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

def load_model(onnx_path):
    # Load the ONNX model
    ort_session = onnxruntime.InferenceSession(onnx_path,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    return ort_session

def preprocess_image(frame):
    # Preprocess the input frame
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 640))  # Adjust the size as needed
    frame = frame.astype(np.uint8)
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    return frame

def main():
    onnx_model_path = 'od-8230_onnxrt_coco_edgeai-mmdet_yolox_m_lite_20220228_model_onnx/model/yolox_m_lite_20220228_model.onnx'
    frame_width, frame_height = 640,640
    image_area = frame_width * frame_height
    # Load the YOLOX ONNX model
    yolox_model = load_model(onnx_model_path)
     # Open the webcam
    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, 60)
    #Supervision algorithm
    box_annotator = sv.BoxAnnotator(thickness=2,text_thickness=2,text_scale=1)
    byte_tracker = sv.ByteTrack(match_thresh=0.3,track_buffer=60,track_thresh=0.7)
    label_annotator = sv.LabelAnnotator()
    #halo_annotator = sv.HaloAnnotator()
    #mask_annotator = sv.MaskAnnotator(color='r',opacity=0.7)
    #polygon_annotator = core.PolygonAnnotator()
    #circle_annotator = sv.CircleAnnotator()
    #ellipse_annotator = sv.EllipseAnnotator()
    start = (100,260)
    end = (200,470)
    line_annotator = sv.LineZone(start,end)

    zone_polygon = (ZONE_POLYGON * np.array([640,640])).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple([640,640]))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    # Find the correct input name by inspecting the ONNX model
    input_name = 'inputNet_IN'  # Replace with the correct input name

    while True:
        st=time.time()
        ret, frame = cap.read()

        # Preprocess the input frame
        input_frame = preprocess_image(frame)

        input_details = yolox_model.get_inputs()
        input_name = input_details[0].name
        output_data = yolox_model.run(None, {input_name: input_frame})
            
        # Assuming ort_outs[0] contains the detection outputs
        person_count = 0
        for i,r in enumerate(output_data):
            r = np.squeeze(r)
            if r.ndim == 1:
                r = np.expand_dims(r, 1)
            output_data[i] = r

        if output_data[-1].ndim < 2:
            output_data = output_data[:-1]
        out = np.concatenate(output_data, axis = -1)
        filtered_detections = out[(out[:, 4] >= 0.65) & (out[:, 5] == 0)]
        # print(filtered_detections.shape[0])

        # Add a 6th column for box IDs starting from zero
        num_boxes = filtered_detections.shape[0]
        box_ids = np.arange(num_boxes)  # Start from zero
        box_ids_column = box_ids.reshape(-1, 1)

        # Concatenate the box IDs to the original array
        filtered_detections = np.concatenate((filtered_detections, box_ids_column), axis=1)
    
        detections= sv.Detections.from_yolov5(filtered_detections)
        detections = detections[detections.class_id == 0]
        detections = detections[(detections.area / image_area) < 0.8]
        detections = byte_tracker.update_with_detections(detections)
        # print(len(detections[0]))
        labels = [
                f"#{tracker_id} {[class_id]}"
                for class_id, tracker_id
                in zip(detections.class_id, detections.tracker_id)
                ]

        frame = box_annotator.annotate(scene=frame,detections=detections)
        frame= label_annotator.annotate(frame, detections=detections)
        #frame = ellipse_annotator.annotate(frame, detections=detections)
        #frame = polygon_annotator.annotate(scene=frame,detections=detections)
        #frame = mask_annotator.annotate(scene=frame,detections=detections)
        #frame = halo_annotator.annotate(scene=frame,deqtections=detections)
        #frame = circle_annotator.annotate(scene=frame,detections=detections)
        #frame = line_annotator.trigger(detections=detections)
        
        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        '''for elements in detections.tracker_id:
            if(elements not in queue):
                queue.append(elements)'''
        
        for i in range(len(detections.tracker_id)):
            if detections.tracker_id[i] not in queue and zone.trigger(detections=detections)[i] == True:
                queue.append(detections.tracker_id[i])

        print(len(queue))
        #print(zone.trigger(detections=detections))
        #print(zone)

        # Postprocess the output
        #boxes, scores, classes = postprocess_output(result)

        # Draw bounding boxes on the frame
        #draw_boxes(frame, boxes, scores, classes)

        # Display the processed frame
        cv2.imshow('YOLOX Webcam',frame)
        # Break the loop if 'q' key is pressed
        ft=time.time()
        # print(1/(ft-st))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()

