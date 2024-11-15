def main():
    onnx_model_path = 'path/to/your/yolox_model.onnx'

    # Load the YOLOX ONNX model
    yolox_model = load_model(onnx_model_path)

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Find the correct input name by inspecting the ONNX model
    input_name = 'inputNet_IN'  # Replace with the correct input name

    while True:
        ret, frame = cap.read()

        # Preprocess the input frame
        input_frame = preprocess_image(frame)

        # Run inference
        outputs = yolox_model.run(None, {input_name: input_frame})

        # Postprocess the output
        boxes, scores, classes = postprocess_output(outputs)

        # Draw bounding boxes on the frame
        draw_boxes(frame, boxes, scores, classes)

        # Display the processed frame
        cv2.imshow('YOLOX Webcam', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
