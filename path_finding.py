import cv2
import math

from helper_functions import Point, Helper
from topographic_map import TopographicMap
from cropped_image import CroppedImage

class Cell:
	def __init__(self):
		self.density = 0
		self.water_density = 0
		self.forrest_density = 0

class Grid:
	def __init__(self, cropped_img, cell_width):
		self.cropped_img = cropped_img
		self.cell_width = cell_width

		rows, cols = self.cropped_img.image_masks.topo_mask.shape
		self.grid_resolution = int(rows / self.cell_width)
		self.array = [[Cell() for x in range(self.grid_resolution)] for y in range(self.grid_resolution)]

		self.__initialize_array()

	def get_cell(self, point):
		if point.x >= 0 and point.x < len(self.array):
			if point.y >= 00 and point.y < len(self.array[point.x]):
				return self.array[point.x][point.y]		

		return None

	def convert_pixel_to_grid_point(self, point):
		x = int(point.x / self.cell_width)
		y = int(point.y / self.cell_width)

		return Point(x,y)

	def convert_grid_to_pixel_point(self, point):
		x = (point.x * self.cell_width) + int(self.cell_width/2)
		y = point.y * self.cell_width + int(self.cell_width/2)

		return Point(x,y)

	def __initialize_array(self):
			for col in range(self.grid_resolution):
				for row in range(self.grid_resolution):
					self.__initialize_cell_densities(Point(col, row))

	def __initialize_cell_densities(self, point):
		cell = self.get_cell(point)

		cell.density = self.__get_image_density_at_pixel(point, self.cropped_img.contours)
		cell.water_density = self.__get_image_density_at_pixel(point, self.cropped_img.image_masks.blue_mask)
		cell.forrest_density = self.__get_image_density_at_pixel(point, self.cropped_img.image_masks.green_mask)

	def __get_image_density_at_pixel(self, point, mask):
		cell_image = self.__get_cell_covered_image(point, mask)

		non_zero_pixels = cv2.countNonZero(cell_image)
		total_pixels = self.cell_width * self.cell_width

		return non_zero_pixels / total_pixels

	def __get_cell_covered_image(self, point, mask):
		minY = point.y * self.cell_width
		maxY = minY + self.cell_width
		minX = point.x * self.cell_width
		maxX = minX + self.cell_width

		cell_image = mask[minY:maxY, minX:maxX]

		return cell_image

	def add_grid_to_image(self, img, lineThickness):
		copy = img.copy()
		row, col, chan = copy.shape

		i = 0
		while i < row - self.cell_width:
			cv2.line(copy,(0,i),(col,i),(255,0,0),lineThickness)
			i += self.cell_width

		j = 0
		while j < col - self.cell_width:
			cv2.line(copy,(j,0),(j,row),(255,0,0),lineThickness)
			j += self.cell_width

		return copy

	def add_density_to_image(self, img):
		copy = img.copy()

		for c in range(self.grid_resolution):
			for r in range(self.grid_resolution):
				x = (c * self.cell_width) + int(self.cell_width / 2)
				y = (r * self.cell_width) + int(self.cell_width / 2)

				cell = self.get_cell(Point(c,r))

				if cell.density > 0.1:
					cv2.circle(copy, (x,y), int(self.cell_width/2), (0,255,0), 2)

		return copy

class Node:
	def __init__(self, point, f=0.0, g=0.0, h=0.0, parent=None):
		self.coord = point
		self.f = f
		self.g = g
		self.h = h
		self.parent = parent

class UserSettings:
	# resize_factor = 6
	# resize_factor = 2

	def __init__(self, topo_map):
		self.topo_map = topo_map

		self.start = None
		self.end = None
		self.cropped_img = None

		self.avoid_water = None
		self.max_angle = None
		self.cell_width = None

	@classmethod
	def initialized_from_filename(cls, filename):
		topo_map = TopographicMap(filename)
		user_settings = cls(topo_map)
		user_settings.init_settings()

		return user_settings

	def init_settings(self):
		if self.start is None or self.end is None:
			self.find_start_end_points()

		if self.avoid_water is None or self.max_angle is None or self.cell_width is None:
			self.avoid_water = True
			self.max_angle = 45
			self.cell_width = 50

	def find_start_end_points(self):
		# self.temp_img = self.topo_map.image.copy()

		# self.start = Point(-1, -1)
		# self.end = Point(-1, -1)

		# cv2.namedWindow("image")
		# cv2.setMouseCallback("image", self.__click_image)
		# cv2.imshow("image", self.temp_img)

		# while(self.start.x < 0 or self.end.x < 0):
		# 	k = cv2.waitKey(1000) & 0xFF
		# 	cv2.imshow("image", self.temp_img)

		# 	if k == ord('q') or k == ord(' '):
		# 		break

		# print(self.start)
		# print(self.end)

		# self.start = Point(342, 357)
		# self.end = Point(586, 527)

		self.start = Point(400, 400)
		self.end = Point(1000, 1000)

		self.__find_cropped_image()

	def get_feet_per_pixel(self):
		return self.topo_map.image_data.feet_per_pixel

	def get_contour_interval_dist(self):
		return self.topo_map.image_data.contour_interval_dist

	def __find_cropped_image(self, padding = 50):
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
		img = cv2.resize(img, None, fx=Helper.resize_factor, fy=Helper.resize_factor, 
			interpolation = cv2.INTER_LINEAR)
		
		# init cropped_img for extracting contours, etc.		
		self.cropped_img = CroppedImage(img)

		# use ratios to find scaled start/end points 
		self.cropped_img.start = Point(int(sxFactor * width), int(syFactor * height))
		self.cropped_img.end = Point(int(exFactor * width), int(eyFactor * height))

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

class PathFinder:
	step_cost = 1
	diag_step_cost = math.sqrt(2)
	max_angle = 90
	max_cost = 1000

	def __init__(self, user_settings):
		self.user_settings = user_settings
		self.grid = Grid(user_settings.cropped_img, user_settings.cell_width)

	@classmethod
	def run_from_user_settings(cls, user_settings):
		path_finder = cls(user_settings)
		path = path_finder.find_path()

		if path is not None:
			path_img = path_finder.draw_path(path)
		else:
			path_img = None

		return path, path_img

	def find_path(self):
		grid_start = self.grid.convert_pixel_to_grid_point(self.user_settings.cropped_img.start)
		grid_end = self.grid.convert_pixel_to_grid_point(self.user_settings.cropped_img.end)

		return self.__calculate_path(grid_start, grid_end)

	def find_path_from_pixel_coords(self, start_point, end_point):
		grid_start = self.grid.convert_pixel_to_grid_point(self._cropped_start)
		grid_end = self.grid.convert_pixel_to_grid_point(self._cropped_end)

		return self.__calculate_path(grid_start, grid_end)

	def __calculate_path(self, start_point, end_point):
		open_nodes = [Node(start_point)]
		closed_nodes = []

		count = 0

		print("start: " + str(start_point.x) + ", " + str(start_point.y))
		print("end: " + str(end_point.x) + ", " + str(end_point.y))

		while len(open_nodes) > 0:
			open_nodes.sort(key = lambda x: x.f, reverse=True)
			cur_node = open_nodes.pop()
			successors = self.__generate_successor_nodes(cur_node)

			for successor in successors:
				if self.__are_equal_points(successor.coord, end_point):
					return successor

				self.__calculate_heuristic(cur_node, successor, end_point)

				if not self.__is_position_already_reached_with_lower_heuristic(successor, open_nodes) and \
				not self.__is_position_already_reached_with_lower_heuristic(successor, closed_nodes):
					open_nodes.append(successor)

			closed_nodes.append(cur_node)

			count += 1

			if count % 100 == 0:
				print("open nodes: " + str(len(open_nodes)))
				print("closed nodes: " + str(len(closed_nodes)))
				print("---------")

		return None

	def __generate_successor_nodes_original(self, cur_node):
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

			if y < self.grid.grid_resolution:
				lb = Node(Point(x, y), parent=cur_node)
				successors.append(lb)

		# successors to the right
		x = point.x + 1
		
		if x < self.grid.grid_resolution:
			rm = Node(Point(x, point.y), parent=cur_node)
			successors.append(rm)

			y = point.y - 1

			if y >= 0:
				rt = Node(Point(x, y), parent=cur_node)
				successors.append(rt)

			y = point.y + 1

			if y < self.grid.grid_resolution:
				rb = Node(Point(x, y), parent=cur_node)
				successors.append(rb)

		# top middle
		y = point.y + 1

		if y < self.grid.grid_resolution:
			mb = Node(Point(point.x, y), parent=cur_node)
			successors.append(mb)

		# bottom middle
		y = point.y - 1

		if y > 0:
			mt = Node(Point(point.x, y), parent=cur_node)
			successors.append(mt)

		successors = self.__remove_invalid_nodes(successors)

		return successors

	def __generate_successor_nodes(self, cur_node):
		# variable names indicate top, middle, bottom -> left, middle, right -> (optional) left, middle, right, top, bottom
		# ie: the locations around cur_node upto to cells away, roughly every 22 degrees

		tl = self.__get_nearest_contour_node(cur_node, Point(-1, -1))
		tlr = self.__get_nearest_contour_node(cur_node, Point(-1, -2))
		tm = self.__get_nearest_contour_node(cur_node, Point(0, -1))
		tr = self.__get_nearest_contour_node(cur_node, Point(1, -1))
		trl = self.__get_nearest_contour_node(cur_node, Point(1, -2))
		
		ml = self.__get_nearest_contour_node(cur_node, Point(-1, 0))
		mlt = self.__get_nearest_contour_node(cur_node, Point(-2, -1))
		mlb = self.__get_nearest_contour_node(cur_node, Point(-2, 1))
		mr = self.__get_nearest_contour_node(cur_node, Point(1, 0))
		mrt = self.__get_nearest_contour_node(cur_node, Point(2, -1))
		mrb = self.__get_nearest_contour_node(cur_node, Point(2, 1))
		
		bl = self.__get_nearest_contour_node(cur_node, Point(-1, 1))
		blr = self.__get_nearest_contour_node(cur_node, Point(-1, 2))
		bm = self.__get_nearest_contour_node(cur_node, Point(0, 1))
		br = self.__get_nearest_contour_node(cur_node, Point(1, 1))
		brl = self.__get_nearest_contour_node(cur_node, Point(1, 2))

		return list(filter(lambda x: x is not None, [tl, tlr, tm, tr, trl, ml, mlt, mlb, mr, mrt, mrb, bl, blr, bm, br, brl]))

	def __are_equal_points(self, p1, p2):
		return p1.x == p2.x and p1.y == p2.y

	def __calculate_heuristic(self, cur_node, successor_node, end_point):
		successor_node.g = cur_node.g + self.__get_cost_between_nodes(cur_node, successor_node)
		successor_node.h = self.__get_grid_distance_between_points(successor_node.coord, end_point)
		successor_node.f = successor_node.g + successor_node.h

	def __is_position_already_reached_with_lower_heuristic(self, cur_node, reached_nodes):
		temp = list(filter(lambda x: 
			self.__are_equal_points(x.coord, cur_node.coord) and 
			x.f <= cur_node.f, 
			reached_nodes))

		if len(temp) > 0:
			return True

		return False

	def __get_nearest_contour_node(self, cur_node, direction):
		cur_point = self.__get_next_point(cur_node.coord, direction)
		nearest_node = None

		count = 0

		while nearest_node is None:
			if self.__is_point_valid(cur_point):
				cur_cell = self.grid.get_cell(cur_point)

				if cur_cell.density > 0:
					nearest_node = Node(cur_point, parent=cur_node)
				else:
					cur_point = self.__get_next_point(cur_point, direction)
			else:
				break

			# count += 1
			# print("===" + str(count))
			# print(str(cur_point) + "===")

		return nearest_node

	def __get_cost_between_nodes(self, start_node, end_node):
		angle_cost = self.__get_angle_cost(start_node, end_node)
		terrain_cost = self.__get_terrain_cost(end_node)
		distance = self.__get_grid_distance_between_points(start_node.coord, end_node.coord)

		return angle_cost + terrain_cost + distance

	def __get_grid_distance_between_points(self, start, end):
		dx = abs(start.x - end.x)
		dy = abs(start.y - end.y)

		diag_cost = min(dx, dy) * PathFinder.diag_step_cost
		straight_cost = abs(dx  - dy) * PathFinder.step_cost

		distance = diag_cost + straight_cost

		return distance

	def __get_next_point(self, cur_point, direction):
		if abs(direction.x) > 1:
			temp_point = Point(cur_point.x + int(direction.x/2), cur_point.y + direction.y)
			temp_point2 = Point(cur_point.x + int(direction.x/2), cur_point.y + int(direction.y/2))

			if not self.__is_point_valid(temp_point) or not self.__is_point_valid(temp_point2):
				return None

		if abs(direction.y) > 1:
			temp_point = Point(cur_point.x + int(direction.x/2), cur_point.y + int(direction.y/2))
			temp_point2 = Point(cur_point.x + direction.x, cur_point.y + int(direction.y/2))

			if not self.__is_point_valid(temp_point) or not self.__is_point_valid(temp_point2):
				return None

		next_point = Point(cur_point.x + direction.x, cur_point.y + direction.y)

		return next_point

	def __is_point_valid(self, point):
		if point is None:
			return False

		if not self.__is_point_in_grid(point):
			return False

		if self.user_settings.avoid_water and self.grid.get_cell(point).water_density != 0:
			return False

		return True

	def __get_angle_cost(self, start_node, end_node):
		angle = self.__get_angle_between_nodes(start_node, end_node)
		
		if angle <= self.user_settings.max_angle:
			angle_cost = (angle / PathFinder.max_angle)
		else:
			angle_cost = PathFinder.max_cost

		return angle_cost

	def __get_terrain_cost(self, node):
		cell = self.grid.get_cell(node.coord)

		terrain_cost = cell.water_density * 10
		terrain_cost += cell.forrest_density * 10

		return terrain_cost

	def __get_angle_between_nodes(self, start_node, end_node):
		direction = Point(end_node.coord.x - start_node.coord.x, end_node.coord.y - start_node.coord.y)
		
		nearest_density_point = end_node.coord#self.__get_nearest_contour_point(end_node.coord, direction)

		start = self.grid.convert_grid_to_pixel_point(start_node.coord)
		end = self.grid.convert_grid_to_pixel_point(nearest_density_point)

		pixel_dist_to_density = self.__get_pixel_distance_between_points(start, end)
		feet_dist_to_density = pixel_dist_to_density / self.user_settings.get_feet_per_pixel()
		feet_dist_to_density = min(feet_dist_to_density, self.user_settings.get_contour_interval_dist())

		theta = math.acos(feet_dist_to_density / self.user_settings.get_contour_interval_dist())

		return math.degrees(theta)

	def __get_nearest_contour_point(self, cur_point, direction):
		cur_cell = self.grid.get_cell(cur_point)

		while cur_cell.density == 0: 
			next_point = Point(cur_point.x + direction.x, cur_point.y + direction.y)

			if self.__is_point_in_grid(next_point):
				cur_point = next_point
				cur_cell = self.grid.get_cell(cur_point)
			else:
				break

		return cur_point

	def __get_pixel_distance_between_points(self, start, end):
		resized_distance = math.sqrt(math.pow(start.x-end.x, 2) + math.pow(start.y-end.y, 2))
		actual_distance = resized_distance / UserSettings.resize_factor

		return actual_distance

	def __is_point_in_grid(self, point):
		if point.x >= 0 and point.x < self.grid.grid_resolution and point.y >= 0 and point.y < self.grid.grid_resolution:
			return True

		return False

	def draw_path(self, node):
		path_img = self.user_settings.cropped_img.cv_image.copy()

		from_point = self.grid.convert_grid_to_pixel_point(node.coord)

		while node.parent is not None:
			parent = node.parent
			to_point = self.grid.convert_grid_to_pixel_point(parent.coord)

			cv2.line(path_img,(from_point.x, from_point.y),(to_point.x, to_point.y),(0,0,255),2)

			node = parent
			from_point = to_point

		return path_img






















