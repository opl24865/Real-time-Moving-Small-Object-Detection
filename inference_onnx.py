import numpy as np
import onnxruntime as ort
import time
from torchvision import transforms
import cv2
import torch
from collections import deque

class shuttlecock_detector:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.trans = transforms.Compose([transforms.ToTensor()])
        self.model_path = "./models_weight/new_diffnet_test_fp16.onnx"
        self.session = None  # 延後初始化
        self.input_name = None
        self.output_name = None

    def create_model(self):
        # 建立 ONNX Runtime session（使用 GPU）
        providers = ["AzureExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, images):
        """
        Args:
            images: list of 3 images in HWC format, uint8 [3, 480, 640, 3]
        Returns:
            input_tensor: np.array, shape [1, 3, 3, 480, 640], dtype float16
        """
        tensor_imgs = [self.trans(img) for img in images]       # [3, 3, 480, 640]            
        stacked = torch.stack(tensor_imgs, dim=0)               # Tensor(3, 3, 480, 640)
        input_tensor = stacked.unsqueeze(0).half().numpy()      # shape: (1, 3, 3, 480, 640)
        return input_tensor

    def get_object_center(self, heatmap):
        """ Get coordinates from the heatmap.

            args:
                heatmap - A numpy.ndarray of a single heatmap with shape (H, W)

            returns:
                ints specifying center coordinates of object
        """
        if np.amax(heatmap) == 0:
            # No respond in heatmap
            return 0, 0
        else:
            # Find all respond area in the heapmap
            (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(ctr) for ctr in cnts]

            # Find largest area amoung all contours
            max_area_idx = 0
            max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
            for i in range(len(rects)):
                area = rects[i][2] * rects[i][3]
                if area > max_area:
                    max_area_idx = i
                    max_area = area
            target = rects[max_area_idx]
        
        return int((target[0] + target[2] / 2)), int((target[1] + target[3] / 2))

    def predict(self, images):
        """
        Args:
            images: list of 3 RGB images, each shape (480, 640, 3), dtype uint8
        Returns:
            keypoint_2d: numpy array shape (2,) - (x, y)
        """
        # t0 = time.time()
        input_tensor = self.preprocess(images)
        # t1 = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        # t2 = time.time()
        heatmap = outputs[0][0, 2]
        heatmap = (heatmap > 0.5).astype(np.uint8) * 255
        cx_pred, cy_pred = self.get_object_center(heatmap)
        # t3 = time.time()

        # print(f'Preprocess Time:  {(t1 - t0) * 1000:.1f} ms')
        # print(f'Inference Time:   {(t2 - t1) * 1000:.1f} ms')
        # print(f'Postprocess Time:   {(t3 - t2) * 1000:.1f} ms')

        keypoints_2d = (cx_pred, cy_pred)  # shape: [2,], already numpy
        
        if keypoints_2d[0] == 0 and keypoints_2d[1] == 0:
            keypoints_2d = (-1, -1)

        return keypoints_2d
        
        
if __name__ == "__main__":
	model = shuttlecock_detector()
	model.create_model()
	cap = cv2.VideoCapture(0)
	q = deque(maxlen=3)
	if not cap.isOpened():
		print("Can't open camera")
		
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	cap.set(cv2.CAP_PROP_FPS, 30)
	 
	while True:
		ret, frame = cap.read()
		q.append(frame.copy())
		if len(q) == 3:
			frame1 = np.expand_dims(q[0], axis=0)
			frame2 = np.expand_dims(q[1], axis=0)
			frame3 = np.expand_dims(q[2], axis=0)
			imgstack = np.vstack([frame1, frame2, frame3])
			keypoint_2d = model.predict(imgstack)
			print(keypoint_2d)
		if not ret:
			print("Can't read frame from camera")
			break
		
		cv2.imshow("camera", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	cap.release()
	cv2.destroyAllWindows()

