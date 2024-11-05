import onnxruntime
import cv2
import numpy as np
import supervision as sv

ZONE_POLYGON = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
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
    onnx_model_path = 'od-8220_onnxrt_coco_edgeai-mmdet_yolox_s_lite_640x640_20220221_model_onnx\model\yolox_s_lite_640x640_20220221_model.onnx'
    frame_width, frame_height = 640,640
    # Load the YOLOX ONNX model
    yolox_model = load_model(onnx_model_path)
     # Open the webcam
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    #Supervision algorithm
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    label_annotator = sv.LabelAnnotator()

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
        ret, frame = cap.read()

        # Preprocess the input frame
        input_frame = preprocess_image(frame)
        input_details = yolox_model.get_inputs()
        input_name = input_details[0].name
        output_data = yolox_model.run(None, {input_name: input_frame})
            
        # # Assuming ort_outs[0] contains the detection outputs
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
        detections=sv.Detections.from_yolov5(filtered_detections)

        # print(len(detections[0]))
        labels = [
                f"{class_id} {confidence:0.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence) if confidence>=0.75 and class_id==0
                ]
        
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )
        print(labels)
        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)
        
        # Postprocess the output
        #boxes, scores, classes = postprocess_output(result)

        # Draw bounding boxes on the frame
        #draw_boxes(frame, boxes, scores, classes)

        # Display the processed frame
        cv2.imshow('YOLOX Webcam',frame)
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()

