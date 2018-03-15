import cv2

from helper_functions import Point, Helper
from topographic_map import TopographicMap
from cropped_image import CroppedImage

class NodeMethod:
	nearest_grade = 1
	single_step = 2
	nearest_density_cell = 3

class UserSettings:

	def __init__(self):
		self.topo_map = None

		self.start = Point(350, 220)
		self.end = Point(850, 700)
		self.cropped_img = None

		self.avoid_water = False
		self.avoid_forrest = False
		self.max_grade = 30
		self.cell_width = 30

		self.node_method = NodeMethod.single_step 

	def __str__(self):
		ret = "1) filename: " + (self.topo_map.filename if self.topo_map is not None else "None") + "\n"
		ret += "2) start/end: " + str(self.start) + " --> " + str(self.end) + "\n"
		ret += "3) avoid water: " + str(self.avoid_water) + "\n"
		ret += "4) avoid forrest: " + str(self.avoid_forrest) + "\n"
		ret += "5) max grade: " + str(self.max_grade) + "\n"
		ret += "6) path precision (grid width): " + str(self.cell_width) + "\n"
		ret += "7) node method: "

		if self.node_method == NodeMethod.nearest_grade:
			ret += "nearest grade"
		elif self.node_method == NodeMethod.single_step:
			ret += "single step"
		elif self.node_method == NodeMethod.nearest_density_cell:
			ret += "nearest density cell"

		return ret

	@classmethod
	def initialized_from_filename(cls, filename):
		user_settings = cls()
		user_settings.set_topo_map(filename)

		return user_settings

	def set_topo_map(self, filename):
		self.topo_map = TopographicMap(filename)

	def find_start_end_points(self):
		self.temp_img = self.topo_map.image.copy()#[:][1500:]

		self.start = Point(-1, -1)
		self.end = Point(-1, -1)

		cv2.namedWindow("image")
		cv2.setMouseCallback("image", self.__click_image)
		cv2.imshow("image", self.temp_img)

		while(self.start.x < 0 or self.end.x < 0):
			k = cv2.waitKey(1000) & 0xFF
			cv2.imshow("image", self.temp_img)

			if k == ord('q') or k == ord(' '):
				break

		cv2.imshow("image", self.temp_img)
		# print(self.start)
		# print(self.end)

		# self.start = Point(1385, 620)
		# self.end = Point(1525, 462)

		# self.start = Point(400, 400)
		# self.end = Point(600, 600)
		self.find_cropped_image()

	def get_feet_per_pixel(self):
		return self.topo_map.image_data.feet_per_pixel

	def get_contour_interval_dist(self):
		return self.topo_map.image_data.contour_interval_dist

	def find_cropped_image(self, padding = 100):
		# get max of width and height of points
		dist = max(abs(self.start.x - self.end.x), abs(self.start.y - self.end.y))

		# calculate padding needed for each point  
		yPad = int((dist - abs(self.start.y - self.end.y)) / 2) + padding
		xPad = int((dist - abs(self.start.x - self.end.x)) / 2) + padding 

		# crop image around start and end points with padding
		minY = min(self.start.y, self.end.y) - yPad
		maxY = max(self.start.y, self.end.y) + yPad

		minX = min(self.start.x, self.end.x) - xPad
		maxX = max(self.start.x, self.end.x) + xPad

		img = self.topo_map.image[minY : maxY, minX : maxX]

		# calculate start/end points for cropped image
		# width/height of cropped image
		width = maxX - minX
		height = maxY - minY

		# ratio of start/end points to cropped image size
		sxFactor = ((self.start.x - minX) / width)
		syFactor = ((self.start.y - minY) / height)
		exFactor = ((self.end.x - minX) / width)
		eyFactor = ((self.end.y - minY) / height)

		# width/height of cropped and rescaled image
		width *= Helper.resize_factor
		height *= Helper.resize_factor

		# scale image by resize factor
		# img = cv2.resize(img, None, fx=Helper.resize_factor, fy=Helper.resize_factor, 
		# 	interpolation = cv2.INTER_LINEAR)
		
		# init cropped_img for extracting contours, etc.		
		self.cropped_img = CroppedImage(img)

		# use ratios to find scaled start/end points 
		self.cropped_img.start = Point(int(sxFactor * width), int(syFactor * height))
		self.cropped_img.end = Point(int(exFactor * width), int(eyFactor * height))

	def __click_image(self, event, x, y, flags, param):
		if event == 1:
			if self.start.x < 0:
				cv2.circle(self.temp_img, (x,y), 5, (0,255,0), 2)
				self.start.x = x #+ 1500
				self.start.y = y #+ 1500
			elif self.end.x < 0:
				cv2.circle(self.temp_img, (x,y), 5, (0,0,255), 2)
				self.end.x = x #+ 1500
				self.end.y = y #+ 1500