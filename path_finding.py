import cv2

from cropped_image import CroppedImage

class PathFinder:
	def __init__(self, topo_map):
		self.topo_map = topo_map

		# global image start/end points
		self.startX = -1
		self.startY = -1
		self.endX = -1
		self.endY = -1

		# cropped image and relative start/end points
		self.cropped_img = None
		self.cropped_startX = -1
		self.cropped_startY = -1
		self.cropped_endX = -1
		self.cropped_endY = -1

	def find_path(self):
		if self.cropped_img is None:
			self.get_start_end_image()

		self.cropped_img.get_contours()

	def get_start_end_image(self, padding = 20, resize_factor = 6):
		self.__get_start_end_points()

		# get max of width and height of points
		dist = max(abs(self.startX - self.endX), abs(self.startY - self.endY))

		# calculate padding needed for each point  
		yPad = int((dist - abs(self.startY - self.endY)) / 2) + padding
		xPad = int((dist - abs(self.startX - self.endX)) / 2) + padding 

		# crop image around start and end points with padding
		minY = min(self.startY, self.endY) - yPad
		maxY = max(self.startY, self.endY) + yPad

		minX = min(self.startX, self.endX) - xPad
		maxX = max(self.startX, self.endX) + xPad

		img = self.topo_map.image[minY : maxY, minX : maxX]

		# calculate start/end points for cropped image
		# width/height of cropped image
		width = maxX - minX
		height = maxY - minY

		# ratio of start/end points to cropped image size
		sxFactor = ((self.startX - minX) / width)
		syFactor = ((self.startY - minY) / height)
		exFactor = ((self.endX - minX) / width)
		eyFactor = ((self.endY - minY) / height)

		# width/height of cropped and rescaled image
		width *= resize_factor
		height *= resize_factor

		# use ratios to find scaled start/end points 
		self.cropped_startX = int(sxFactor * width)
		self.cropped_startY = int(syFactor * height)
		self.cropped_endX = int(exFactor * width)
		self.cropped_endY = int(eyFactor * height)

		# scale image by resize factor
		img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor, 
			interpolation = cv2.INTER_LINEAR)
		
		# init cropped_img for extracting contours, etc.		
		self.cropped_img = CroppedImage(img)

	def __get_start_end_points(self):
		self.temp_img = self.topo_map.image.copy()

		cv2.namedWindow("image")
		cv2.setMouseCallback("image", self.__click_image)
		cv2.imshow("image", self.temp_img)

		while(self.startX < 0 or self.endX < 0):
			k = cv2.waitKey(1000) & 0xFF
			cv2.imshow("image", self.temp_img)

			if k == ord('q') or k == ord(' '):
				break

		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

	def __click_image(self, event, x, y, flags, param):
		if event == 1:
			if self.startX < 0:
				self.startX = x
				self.startY = y
				cv2.circle(self.temp_img, (x,y), 5, (0,255,0), 2)
			elif self.endX < 0:
				self.endX = x
				self.endY = y
				cv2.circle(self.temp_img, (x,y), 5, (0,0,255), 2)
