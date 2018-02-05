import cv2
import math

from cropped_image import CroppedImage

class ImageDensities:
	def __init__(self):
		self.density = 0
		self.water_density = 0

class Cell:
	def __init__(self):
		self.image_densities = ImageDensities()

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

class Grid:
	def __init__(self, image_masks, grid_dim):
		self.image_masks = image_masks
		self.grid_dim = grid_dim
		self.array = [[Cell() for x in range(grid_dim)] for y in range(grid_dim)]

		rows, cols = self.image_masks.topo_mask.shape
		self.dx = int(cols / grid_dim)
		self.dy = int(rows / grid_dim)

		self.__initialize_array()

	def get_cell(self, point):
		return self.array[point.x][point.y]

	def convert_pixel_to_grid_point(self, point):
		x = int(point.x / self.dx)
		y = int(point.y / self.dy)

		return Point(x,y)

	def convert_grid_to_pixel_point(self, point):
		x = (point.x * self.dx) + int(self.dx/2)
		y = point.y * self.dy + int(self.dy/2)

		return Point(x,y)

	def __initialize_array(self):
			for col in range(self.grid_dim):
				for row in range(self.grid_dim):
					self.__initialize_cell_densities(Point(col, row))

	def __initialize_cell_densities(self, point):
		cell_densities = self.get_cell(point).image_densities

		cell_densities.density = self.__get_image_density_at_pixel(point, self.image_masks.topo_mask)
		cell_densities.water_density = self.__get_image_density_at_pixel(point, self.image_masks.blue_mask)

	def __get_image_density_at_pixel(self, point, mask):
		cell_image = self.__get_cell_covered_image(point, mask)

		non_zero_pixels = cv2.countNonZero(cell_image)
		total_pixels = self.dx * self.dy

		return non_zero_pixels / total_pixels

	def __get_cell_covered_image(self, point, mask):
		minY = point.y * self.dy
		maxY = minY + self.dy
		minX = point.x * self.dx
		maxX = minX + self.dx

		cell_image = mask[minY:maxY, minX:maxX]

		return cell_image

class Node:
	def __init__(self, point, f=0.0, g=0.0, h=0.0, parent=None):
		self.coord = point
		self.f = f
		self.g = g
		self.h = h
		self.parent = parent

class AStarRunner:
	def __init__(self, grid, slope_constant):
		self._grid = grid
		self.slope_constant = 5

	def find_path(self, start_point, end_point):
		open_nodes = [Node(start_point)]
		closed_nodes = []

		print("start: " + str(start_point.x) + ", " + str(start_point.y))
		print("end: " + str(end_point.x) + ", " + str(end_point.y))

		while len(open_nodes) > 0:
			open_nodes.sort(key = lambda x: x.f, reverse=True)
			cur_node = open_nodes.pop()
			successors = self.__generate_successor_nodes(cur_node)

			for s in successors:
				if self.__are_equal_points(s.coord, end_point):
					return s

				self.__calculate_heuristic(cur_node, s, end_point)

				if self.__is_position_already_reached_with_lower_heuristic(s, open_nodes):
					continue

				if self.__is_position_already_reached_with_lower_heuristic(s, closed_nodes):
					continue

				open_nodes.append(s)

			closed_nodes.append(cur_node)

		return None

	def __generate_successor_nodes(self, cur_node):
		point = cur_node.coord
		successors = []

		# successors to the left
		x = point.x - 1
		
		if x >= 0:
			lm = Node(Point(x, point.y), parent=cur_node)
			successors.append(lm)

			y = point.y - 1

			if y >= 0:
				lt = Node(Point(x, y), parent=cur_node)
				successors.append(lt)

			y = point.y + 1

			if y < self._grid.grid_dim:
				lb = Node(Point(x, y), parent=cur_node)
				successors.append(lb)

		# successors to the right
		x = point.x + 1
		
		if x < self._grid.grid_dim:
			rm = Node(Point(x, point.y), parent=cur_node)
			successors.append(rm)

			y = point.y - 1

			if y >= 0:
				rt = Node(Point(x, y), parent=cur_node)
				successors.append(rt)

			y = point.y + 1

			if y < self._grid.grid_dim:
				rb = Node(Point(x, y), parent=cur_node)
				successors.append(rb)

		# top middle
		y = point.y + 1

		if y < self._grid.grid_dim:
			mb = Node(Point(point.x, y), parent=cur_node)
			successors.append(mb)

		# bottom middle
		y = point.y - 1

		if y > 0:
			mt = Node(Point(point.x, y), parent=cur_node)
			successors.append(mt)

		return successors

	def __are_equal_points(self, p1, p2):
		return p1.x == p2.x and p1.y == p2.y

	def __calculate_heuristic(self, cur_node, successor_node, end_point):
		successor_node.g = cur_node.g + self.__get_dist_between_nodes(cur_node, successor_node)
		successor_node.h = self.__get_flat_distance_between_points(successor_node.coord, end_point)
		successor_node.f = successor_node.g + successor_node.h

	def __is_position_already_reached_with_lower_heuristic(self, cur_node, reached_nodes):
		temp = list(filter(lambda x: x.coord.x == cur_node.coord.x and x.coord.y == cur_node.coord.y and x.f <= cur_node.f, reached_nodes))

		if len(temp) > 0:
			return True

		return False

	def __get_dist_between_nodes(self, start_node, end_node):
		slope = self.__get_slope_between_nodes(start_node, end_node)
		flat_distance = self.__get_flat_distance_between_points(start_node.coord, end_node.coord)
		vert_distance = flat_distance * slope
		direct_distance = math.sqrt(math.pow(flat_distance, 2) + math.pow(vert_distance, 2))

		return direct_distance

	def __get_slope_between_nodes(self, start_node, end_node):
		change = Point(end_node.coord.x - start_node.coord.x, end_node.coord.y - start_node.coord.y)
		
		nearest_density_point = self.__get_nearest_point_with_density(end_node.coord, change)
		dist_to_density = self.__get_flat_distance_between_points(start_node.coord, nearest_density_point)

		slope = self.slope_constant / dist_to_density

		return slope

	def __get_flat_distance_between_points(self, start, end):
		return math.sqrt(math.pow(start.x-end.x, 2) + math.pow(start.y-end.y, 2))

	def __get_nearest_point_with_density(self, cur_point, change):
		cur_cell = self._grid.get_cell(cur_point)

		while cur_cell.image_densities.density == 0: 
			next_point = Point(cur_point.x + change.x, cur_point.y + change.y)

			if self.__is_point_in_grid(next_point):
				cur_point = next_point
				cur_cell = self._grid.get_cell(cur_point)
			else:
				break

		return cur_point

	def __is_point_in_grid(self, point):
		if point.x >= 0 and point.x < self._grid.grid_dim and point.y >= 0 and point.y < self._grid.grid_dim:
			return True

		return False

class PathFinder:
	def __init__(self, topo_map, path_resolution = 100):
		self.topo_map = topo_map
		self.path_resolution = path_resolution

		# global image start/end points
		self.start = Point(-1, -1)
		self.end = Point(-1, -1)

		# cropped image and relative start/end points
		# self._cropped_start = Point(-1, -1)
		# self._cropped_end = Point(-1, -1)
		# self._cropped_img = self.__get_start_end_image()

		# self._grid = Grid(self._cropped_img.image_masks, self.path_resolution)
		
		self._cropped_start = None
		self._cropped_end = None
		self._cropped_img = None

		self._grid = None

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

	def __get_start_end_image(self, padding = 20, resize_factor = 6):
		# self.__get_start_end_points()

		# # get max of width and height of points
		# dist = max(abs(self.start.x - self.end.x), abs(self.start.y - self.end.y))

		# # calculate padding needed for each point  
		# yPad = int((dist - abs(self.start.y - self.end.y)) / 2) + padding
		# xPad = int((dist - abs(self.start.x - self.end.x)) / 2) + padding 

		# # crop image around start and end points with padding
		# minY = min(self.start.y, self.end.y) - yPad
		# maxY = max(self.start.y, self.end.y) + yPad

		# minX = min(self.start.x, self.end.x) - xPad
		# maxX = max(self.start.x, self.end.x) + xPad

		minX = 324		
		maxX = 576
		minY = 303
		maxY = 555
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
		width *= resize_factor
		height *= resize_factor

		# use ratios to find scaled start/end points 
		self._cropped_start = Point(int(sxFactor * width), int(syFactor * height))
		self._cropped_end = Point(int(exFactor * width), int(eyFactor * height))

		# scale image by resize factor
		img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor, 
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
				self.start.x = x
				self.start.y = y
				cv2.circle(self.temp_img, (x,y), 5, (0,255,0), 2)
			elif self.end.x < 0:
				self.end.x = x
				self.end.y = y
				cv2.circle(self.temp_img, (x,y), 5, (0,0,255), 2)

	def calculate_path(self):
		if self._cropped_img is None:
			self.__get_start_end_image()

		start = self._grid.convert_pixel_to_grid_point(self._cropped_start)
		end = self._grid.convert_pixel_to_grid_point(self._cropped_end)

		a_star = AStarRunner(self._grid, slope_constant=5)
		path_node = a_star.find_path(start, end)

		return path_node

	def draw_path(self, image, node):
		path_img = image.copy()
		from_point = self._grid.convert_grid_to_pixel_point(node.coord)

		while node.parent is not None:
			parent = node.parent
			to_point = self._grid.convert_grid_to_pixel_point(parent.coord)

			cv2.line(path_img,(from_point.x, from_point.y),(to_point.x, to_point.y),(0,255,255),2)

			node = parent
			from_point = to_point

		return path_img


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
		rows, cols, chan = img.shape
		dx = int(cols / grid_dim)
		dy = int(rows / grid_dim)

		for c in range(grid_dim):
			for r in range(grid_dim):
				x = (c * dx) + int(dx / 2)
				y = (r * dy) + int(dy / 2)

				cell = grid[c][r]

				if cell.density > 0:
					cv2.circle(img, (x,y), int(dx/2), (255,0,0), 2)

		return img





















