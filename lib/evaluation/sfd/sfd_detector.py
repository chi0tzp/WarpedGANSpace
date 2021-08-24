from .core import FaceDetector
from .detect import *


class SFDDetector(FaceDetector):
	def __init__(self, path_to_detector=None, device="cuda", verbose=False):
		super(SFDDetector, self).__init__(device, verbose)
		self.device = device
		self.face_detector = s3fd()
		self.face_detector.load_state_dict(torch.load(path_to_detector))
		self.face_detector.eval()
		if self.device == "cuda":
			self.face_detector.cuda()

	def detect_from_image(self, tensor_or_path):
		image = self.tensor_or_path_to_ndarray(tensor_or_path)
		bboxlist = detect(self.face_detector, image, device=self.device)[0]
		keep = nms(bboxlist, 0.3)
		bboxlist = bboxlist[keep, :]
		bboxlist = [x for x in bboxlist if x[-1] > 0.5]

		return bboxlist

	def detect_from_batch(self, tensor):
		bboxlists = batch_detect(self.face_detector, tensor, device=self.device)
		error = False
		new_bboxlists = []
		error_index = -1
		for i in range(bboxlists.shape[0]):
			bboxlist = bboxlists[i]
			keep = nms(bboxlist, 0.3)
			if len(keep) > 0:
				bboxlist = bboxlist[keep, :]
				bboxlist = [x for x in bboxlist if x[-1] > 0.5]
				new_bboxlists.append(bboxlist)            
			else: 
				error = True
				error_index = i
				new_bboxlists.append([])

		return new_bboxlists, error, error_index

	@property
	def reference_scale(self):
		return 195

	@property
	def reference_x_shift(self):
		return 0

	@property
	def reference_y_shift(self):
		return 0
