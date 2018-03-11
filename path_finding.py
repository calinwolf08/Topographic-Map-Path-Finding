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
		while i <= row - self.cell_width:
			cv2.line(copy,(0,i),(col,i),(0,255,0),lineThickness)
			i += self.cell_width

		j = 0
		while j <= col - self.cell_width:
			cv2.line(copy,(j,0),(j,row),(0,255,0),lineThickness)
			j += self.cell_width

		return copy

	def add_density_to_image(self, img):
		copy = img.copy()

		for c in range(self.grid_resolution):
			for r in range(self.grid_resolution):
				x = (c * self.cell_width) + int(self.cell_width / 2)
				y = (r * self.cell_width) + int(self.cell_width / 2)

				cell = self.get_cell(Point(c,r))

				if cell.forrest_density > 0:
					cv2.circle(copy, (x,y), int(self.cell_width/2), (0,255,0), 1)

				if cell.water_density > 0:
					cv2.circle(copy, (x,y), int(self.cell_width/2), (255,0,0), 1)

		return copy

	def add_boundary_to_image(self, img, boundary_points):
		copy = img.copy()

		for point in boundary_points:
			pt = self.convert_grid_to_pixel_point(point)
			cv2.circle(copy, (pt.x,pt.y), int(self.cell_width/2), (255,0,255), 2)

		return copy

	def add_heat_map_to_image(self, img):
		copy = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		for c in range(self.grid_resolution):
			for r in range(self.grid_resolution):
				x = (c * self.cell_width) + int(self.cell_width / 2)
				y = (r * self.cell_width) + int(self.cell_width / 2)

				cell = self.get_cell(Point(c,r))

				cv2.circle(copy, (x,y), int(self.cell_width/2), (abs(1-cell.density) * 70,255,255), -1)

		copy = cv2.cvtColor(copy, cv2.COLOR_HSV2BGR)

		return copy

class Node:
	def __init__(self, point, f=0.0, g=0.0, h=0.0, parent=None):
		self.coord = point
		self.f = f
		self.g = g
		self.h = h
		self.parent = parent

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

	def __str__(self):
		ret = "1) filename: " + (self.topo_map.filename if self.topo_map is not None else "None") + "\n"
		ret += "2) start/end: " + str(self.start) + " --> " + str(self.end) + "\n"
		ret += "3) avoid water: " + str(self.avoid_water) + "\n"
		ret += "4) avoid terrain: " + str(self.avoid_forrest) + "\n"
		ret += "5) max grade: " + str(self.max_grade) + "\n"
		ret += "6) path precision (grid width): " + str(self.cell_width)

		return ret

	@classmethod
	def initialized_from_filename(cls, filename):
		user_settings = cls()
		user_settings.set_topo_map(filename)

		return user_settings

	def set_topo_map(self, filename):
		self.topo_map = TopographicMap(filename)

	def find_start_end_points(self):
		self.temp_img = self.topo_map.image.copy()[:][1500:]

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

	def find_cropped_image(self, padding = 50):
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
				self.start.x = x 
				self.start.y = y + 1500
			elif self.end.x < 0:
				cv2.circle(self.temp_img, (x,y), 5, (0,0,255), 2)
				self.end.x = x
				self.end.y = y + 1500

class PathFinder:
	step_cost = 1
	diag_step_cost = math.sqrt(2)
	max_grade = 100
	max_cost = 1000

	def __init__(self, user_settings, previous = None):
		self.user_settings = user_settings
		self.previous = previous
		self.boundary_points = None
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
		# print("pixel start: " + str(self.user_settings.start))
		# print("pixel end: " + str(self.user_settings.end))
		self.grid_start = self.grid.convert_pixel_to_grid_point(self.user_settings.cropped_img.start)
		self.grid_end = self.grid.convert_pixel_to_grid_point(self.user_settings.cropped_img.end)

		return self.__calculate_path()

	def find_path_from_pixel_coords(self, start_point, end_point):
		self.grid_start = self.grid.convert_pixel_to_grid_point(self._cropped_start)
		self.grid_end = self.grid.convert_pixel_to_grid_point(self._cropped_end)

		return self.__calculate_path()

	def __calculate_path(self):
		open_nodes = [Node(self.grid_start)]
		closed_nodes = []

		# count = 0

		# print("grid start: " + str(self.grid_start))
		# print("grid end: " + str(self.grid_end))
		# print("total cells: " + str(self.grid.grid_resolution * self.grid.grid_resolution))

		while len(open_nodes) > 0:
			open_nodes.sort(key = lambda x: x.f, reverse=True)
			cur_node = open_nodes.pop()
			successors = self.__generate_successor_nodes(cur_node)

			for successor in successors:
				if self.__are_equal_points(successor.coord, self.grid_end):
					return successor

				self.__calculate_heuristic(cur_node, successor, self.grid_end)

				if not self.__is_position_already_reached_with_lower_heuristic(successor, open_nodes) and \
				not self.__is_position_already_reached_with_lower_heuristic(successor, closed_nodes):
					open_nodes.append(successor)

			closed_nodes.append(cur_node)

			# count += 1

			# if count % 250 == 0:
			# 	print("successors: " + str(len(successors)))
			# 	print("open nodes: " + str(len(open_nodes)))
			# 	print("closed nodes: " + str(len(closed_nodes)))
			# 	print("---------")

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

		successors = [tl, tlr, tm, tr, trl, ml, mlt, mlb, mr, mrt, mrb, bl, blr, bm, br, brl]

		return list(filter(lambda x: x is not None and self.__point_within_boundary(x), successors))

	def __point_within_boundary(self, node):
		if self.previous is None or self.previous.boundary_points is None:
			return True

		pixel_point = self.grid.convert_grid_to_pixel_point(node.coord)
		previous_grid_point = self.previous.grid.convert_pixel_to_grid_point(pixel_point)

		if previous_grid_point in self.previous.boundary_points:
			return True

		return False

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

	def __get_nearest_contour_node(self, start_node, direction):
		start_point = start_node.coord
		nearest_node = None

		while nearest_node is None:
			cur_point = self.__get_next_point(start_point, direction)

			if self.__is_point_valid(cur_point):
				cur_cell = self.grid.get_cell(cur_point)

				if cur_cell.density > 0 or self.__are_equal_points(cur_point, self.grid_end):
					nearest_node = Node(cur_point, parent=start_node)
				else:
					start_point = cur_point
			elif not self.__is_point_in_grid(cur_point) and \
			not self.__are_equal_points(start_point, start_node.coord):
				nearest_node = Node(start_point, parent=start_node)
			else:
				break

		return nearest_node

	def __get_cost_between_nodes(self, start_node, end_node):
		angle_cost = self.__get_angle_cost(start_node, end_node)
		terrain_cost = self.__get_terrain_cost(end_node)
		distance = self.__get_grid_distance_between_points(start_node.coord, end_node.coord)
		# print("angle_cost: " + str(angle_cost))
		# print("terrain_cost: " + str(terrain_cost))
		# print("distance_cost: " + str(distance))
		# print("-------")

		return angle_cost + terrain_cost + distance

	def __get_grid_distance_between_points(self, start, end):
		dx = abs(start.x - end.x)
		dy = abs(start.y - end.y)

		diag_cost = min(dx, dy) * PathFinder.diag_step_cost
		straight_cost = abs(dx  - dy) * PathFinder.step_cost

		distance = diag_cost + straight_cost

		return distance

	def __get_next_point(self, cur_point, direction):
		temp_point, temp_point2 = self.__get_points_between(cur_point, direction)

		if temp_point is not None and temp_point2 is not None:
			if not self.__is_point_valid(temp_point) or not self.__is_point_valid(temp_point2):
				return None

		next_point = Point(cur_point.x + direction.x, cur_point.y + direction.y)

		return next_point

	def __get_points_between(self, cur_point, direction):
		point = None
		point2 = None

		if abs(direction.x) > 1:
			point = Point(cur_point.x + int(direction.x/2), cur_point.y + direction.y)
			point2 = Point(cur_point.x + int(direction.x/2), cur_point.y + int(direction.y/2))

		elif abs(direction.y) > 1:
			point = Point(cur_point.x + int(direction.x/2), cur_point.y + int(direction.y/2))
			point2 = Point(cur_point.x + direction.x, cur_point.y + int(direction.y/2))

		return point, point2

	def __is_point_valid(self, point):
		if point is None:
			return False

		if not self.__is_point_in_grid(point):
			return False

		return True

	def __get_angle_cost(self, start_node, end_node):
		grade = self.__get_grade_between_nodes(start_node, end_node)
		
		if grade <= self.user_settings.max_grade:
			angle_cost = (grade / PathFinder.max_grade)
			angle_cost = math.pow(angle_cost, 1)
		else:
			angle_cost = PathFinder.max_cost

		return angle_cost * 10

	def __get_terrain_cost(self, node):
		cell = self.grid.get_cell(node.coord)

		if self.user_settings.avoid_water:
			water_cost = cell.water_density * PathFinder.max_cost
		else:
			water_cost = cell.water_density

		if self.user_settings.avoid_forrest:
			forrest_cost = cell.forrest_density * PathFinder.max_cost
		else:
			forrest_cost = cell.forrest_density

		terrain_cost = water_cost + forrest_cost

		return terrain_cost * 10

	def __get_grade_between_nodes(self, start_node, end_node):
		direction = Point(end_node.coord.x - start_node.coord.x, end_node.coord.y - start_node.coord.y)
		nearest_density_point = self.__get_nearest_contour_point(start_node.coord, direction)

		start = self.grid.convert_grid_to_pixel_point(start_node.coord)
		end = self.grid.convert_grid_to_pixel_point(nearest_density_point)

		pixel_dist_to_density = self.__get_pixel_distance_between_points(start, end)
		feet_dist_to_density = pixel_dist_to_density * self.user_settings.get_feet_per_pixel()

		theta = math.atan(self.user_settings.get_contour_interval_dist() / feet_dist_to_density)
		angle = math.degrees(theta)

		grade = Helper.convert_angle_to_grade(theta)

		return grade

	def __get_nearest_contour_point(self, start_point, direction):
		cur_point = start_point
		next_point = Point(cur_point.x + direction.x, cur_point.y + direction.y)

		while self.__is_density_between_points(cur_point, direction) == False: 
			if self.__is_point_in_grid(next_point):
				cur_point = next_point
				next_point = Point(cur_point.x + direction.x, cur_point.y + direction.y)
			else:
				break

		return next_point

	def __is_density_between_points(self, cur_point, direction):
		density = 0

		if abs(direction.x) > 1 or abs(direction.y > 1):
			temp_point, temp_point2 = self.__get_points_between(cur_point, direction)
			density += self.__get_density_from_point(temp_point)
			density += self.__get_density_from_point(temp_point2)

		density += self.__get_density_from_point(cur_point)

		return density == 0

	def __get_density_from_point(self, point):
		if self.__is_point_in_grid(point):
			cell = self.grid.get_cell(point)
			return cell.density
		else:
			return 0

	def __get_pixel_distance_between_points(self, start, end):
		resized_distance = math.sqrt(math.pow(start.x-end.x, 2) + math.pow(start.y-end.y, 2))
		actual_distance = resized_distance / Helper.resize_factor

		return actual_distance

	def __is_point_in_grid(self, point):
		if point is None:
			return False

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

	def set_boundary_points(self, node, distance):
		points = []
		
		cur_point = node.coord

		while node.parent is not None:
			parent = node.parent
			next_point = parent.coord

			direction = self.__get_direction_vector_to_point(cur_point, parent.coord)

			while not self.__are_equal_points(cur_point, parent.coord):
				self.__add_neighbor_points(cur_point, distance, points)

				cur_point.x += direction.x
				cur_point.y += direction.y

			node = parent
			cur_point = next_point 

		self.__add_neighbor_points(node.coord, distance, points)

		self.boundary_points = points

	def __get_direction_vector_to_point(self, from_point, to_point):
		direction = Point(to_point.x - from_point.x, to_point.y - from_point.y)

		if (abs(direction.x) >= 2 and abs(direction.y) >= 2) or abs(direction.x) > 2 or abs(direction.y) > 2:
			factor = max(min(abs(direction.x), abs(direction.y)), 1)

			direction.x = int(direction.x / factor)
			direction.y = int(direction.y / factor)

		return direction

	def __add_neighbor_points(self, point, distance, points):
		for x in range(point.x - distance, point.x + distance + 1):
			for y in range(point.y - distance, point.y + distance + 1):
				cur_point = Point(x,y)
				
				if self.__is_point_in_grid(cur_point) and cur_point not in points:
					points.append(cur_point)






























