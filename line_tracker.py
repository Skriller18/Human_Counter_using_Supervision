import onnxruntime
import cv2
import numpy as np
#from supervision.annotators import core
import supervision as sv
from supervision import Point
import time

queue= []
start = Point(400,0)
end = Point(400,640)

def load_model(onnx_path):
    # Load the ONNX model
    ort_session = onnxruntime.InferenceSession(onnx_path,providers=['CUDAExecutionProvider'])
    return ort_session

def preprocess_image(frame):
    #Preprocess the input frame
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 640))  # Adjust the size as needed
    frame = frame.astype(np.float32)
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    return frame

def main():
    onnx_model_path = 'yolox_m_ti_lite_45p5_64p2.onnx'
    frame_width, frame_height = 640,640
    image_area = frame_width * frame_height
    # Load the YOLOX ONNX model
    yolox_model = load_model(onnx_model_path)

     # Open the webcam
    #cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, 60)

    #Supervision algorithm
    zone = sv.LineZone(start=start,end=end)

    #initiate annotators
    box_annotator = sv.BoxAnnotator(thickness=2,text_thickness=2,text_scale=1)  
    zone_annotator = sv.LineZoneAnnotator()
    byte_tracker = sv.ByteTrack()


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
        filtered_detections = out[(out[:,4] >= 0.65) & (out[:,5] == 0)]
        # print(filtered_detections.shape[0])

        # Add a 6th column for box IDs starting from zero
        num_boxes = filtered_detections.shape[0]
        box_ids = np.arange(num_boxes)  # Start from zero
        box_ids_column = box_ids.reshape(-1, 1)

        # Concatenate the box IDs to the original array
        filtered_detections = np.concatenate((filtered_detections, box_ids_column), axis=1)
        detections= sv.Detections.from_yolov5(filtered_detections)
        detections = detections[detections.class_id == 0]
        #detections = detections[detections.confidence >= 0.8]
        detections = detections[(detections.area / image_area) < 0.8]
        detections = byte_tracker.update_with_detections(detections)

        zone.trigger(detections=detections)
        labels = [
                f"#{tracker_id} {[class_id]}"
                for class_id, tracker_id
                in zip(detections.class_id, detections.tracker_id)
                ]
        
        #annotate
        frame = box_annotator.annotate(scene=frame,detections=detections,labels=labels)
        frame = zone_annotator.annotate(frame=frame,line_counter=zone)      
        
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

