import cv2
import math

from helper_functions import Point
from cropped_image import CroppedImage

class UserInterface:
	resize_factor = 6

	def __init__(self, topo_map, path_resolution = 100):
		self.topo_map = topo_map
		self.path_resolution = path_resolution

		# global image start/end points
		self.start = Point(-1, -1)
		self.end = Point(-1, -1)
		
		self._cropped_start = None
		self._cropped_end = None
		self._cropped_img = None

		self._grid = None

	def __get_start_end_image(self, padding = 20):
		self.__get_start_end_points()

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

		# minX = 334
		# maxX = 488
		# minY = 279
		# maxY = 433

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
		width *= UserInterface.resize_factor
		height *= UserInterface.resize_factor

		# use ratios to find scaled start/end points 
		self._cropped_start = Point(int(sxFactor * width), int(syFactor * height))
		self._cropped_end = Point(int(exFactor * width), int(eyFactor * height))

		# scale image by resize factor
		img = cv2.resize(img, None, fx=UserInterface.resize_factor, fy=UserInterface.resize_factor, 
			interpolation = cv2.INTER_LINEAR)
		
		# init cropped_img for extracting contours, etc.		
		self._cropped_img = CroppedImage(img)
		self._grid = Grid(self._cropped_img.image_masks, self.path_resolution)

	def __get_start_end_points(self):
		self.temp_img = self.topo_map.image.copy()

		cv2.namedWindow("image")
		cv2.setMouseCallback("image", self.__click_image)
		cv2.imshow("image", self.temp_img)

		while(self.start.x < 0 or self.end.x < 0):
			k = cv2.waitKey(1000) & 0xFF
			cv2.imshow("image", self.temp_img)

			if k == ord('q') or k == ord(' '):
				break

		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

	def __click_image(self, event, x, y, flags, param):
		if event == 1:
			if self.start.x < 0:
				cv2.circle(self.temp_img, (x,y), 5, (0,255,0), 2)
				self.start.x = x
				self.start.y = y
			elif self.end.x < 0:
				cv2.circle(self.temp_img, (x,y), 5, (0,0,255), 2)
				self.end.x = x
				self.end.y = y

	def calculate_path(self):
		if self._cropped_img is None:
			self.__get_start_end_image()

		start = self._grid.convert_pixel_to_grid_point(self._cropped_start)
		end = self._grid.convert_pixel_to_grid_point(self._cropped_end)

		a_star = AStarRunner(self)
		path_node = a_star.find_path(start, end)

		return path_node

	def draw_path(self, image, node):
		path_img = image.copy()
		from_point = self._grid.convert_grid_to_pixel_point(node.coord)

		while node.parent is not None:
			parent = node.parent
			to_point = self._grid.convert_grid_to_pixel_point(parent.coord)

			cv2.line(path_img,(from_point.x, from_point.y),(to_point.x, to_point.y),(0,0,255),2)

			node = parent
			from_point = to_point

		return path_img

	@property
	def cropped_start(self):
		if self._cropped_start is None:
			self.__get_start_end_image()

		return self._cropped_start

	@cropped_start.setter
	def cropped_start(self, value):
		self._cropped_img = value

	@property
	def cropped_end(self):
		if self._cropped_end is None:
			self.__get_start_end_image()

		return self._cropped_end

	@cropped_end.setter
	def cropped_end(self, value):
		self._cropped_end = value

	@property
	def cropped_img(self):
		if self._cropped_img is None:
			self.__get_start_end_image()

		return self._cropped_img

	@cropped_img.setter
	def cropped_img(self, value):
		self._cropped_img = value

	@property
	def grid(self):
		if self._grid is None:
			self._grid = Grid(self._cropped_img.image_masks, self.path_resolution)

		return self._grid

	@grid.setter
	def grid(self, value):
		self._grid = value

	# def add_grid(img, grid_dim, lineThickness):
	# 	global startX, startY, endX, endY

	# 	copy = img.copy()
	# 	row, col, chan = copy.shape

	# 	dx = int(col / grid_dim)
	# 	dy = int(row / grid_dim)

	# 	i = 0
	# 	while i < row:
	# 		cv2.line(copy,(0,i),(col,i),(255,0,0),lineThickness)
	# 		i += dy

	# 	j = 0
	# 	while j < col:
	# 		cv2.line(copy,(j,0),(j,row),(255,0,0),lineThickness)
	# 		j += dx

	# 	x = (int(startX / dx) * dx) + int(dx / 2)
	# 	y = (int(startY / dy) * dy) + int(dy / 2)
	# 	cv2.circle(copy, (x,y), int(dx / 2), (0,255,0), 2)

	# 	x = (int(endX / dx) * dx) + int(dx / 2)
	# 	y = (int(endY / dy) * dy) + int(dy / 2)
	# 	cv2.circle(copy, (x,y), int(dx / 2), (0,0,255), 2)

	# 	return copy

	# def show_density(grid, img, grid_dim):
	# 	rows, cols, chan = img.shape
	# 	dx = int(cols / grid_dim)
	# 	dy = int(rows / grid_dim)

	# 	for c in range(grid_dim):
	# 		for r in range(grid_dim):
	# 			x = (c * dx) + int(dx / 2)
	# 			y = (r * dy) + int(dy / 2)

	# 			cell = grid[c][r]

	# 			if cell.density > 0:
	# 				cv2.circle(img, (x,y), int(dx/2), (255,0,0), 2)

	# 	return img





















