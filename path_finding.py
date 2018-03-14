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
		self.max_density = 0

		self.__initialize_array()

	def get_cell(self, point):
		if point.x >= 0 and point.x < len(self.array):
			if point.y >= 00 and point.y < len(self.array[point.x]):
				return self.array[point.x][point.y]		

		print("here: " + str(len(self.array)))
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

		if cell.density > self.max_density:
			self.max_density = cell.density

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

				density = cell.density / self.max_density
				# multiplier = 1 - density
				# cv2.circle(copy, (x,y), int(self.cell_width/2), (int(122 * multiplier),255,255), 2)
				if density > 0.5:
					cv2.circle(copy, (x,y), int(self.cell_width/2), (0,255,255), 2)
				else:
					cv2.circle(copy, (x,y), int(self.cell_width/2), (122,255,255), 2)

		copy = cv2.cvtColor(copy, cv2.COLOR_HSV2BGR)

		return copy

class Node:
	def __init__(self, point, f=0.0, g=0.0, h=0.0, parent=None, grade = None):
		self.coord = point
		self.f = f
		self.g = g
		self.h = h
		self.parent = parent
		self.grade = grade

class UserSettings:

	def __init__(self):
		self.topo_map = None

		self.start = Point(350, 220)
		self.end = Point(850, 700)
		# self.end = Point(450, 320)
		self.cropped_img = None

		self.avoid_water = False
		self.avoid_forrest = False
		self.max_grade = 30
		self.cell_width = 30
		self.grade_in_direction = True
		self.single_step = True

	def __str__(self):
		ret = "1) filename: " + (self.topo_map.filename if self.topo_map is not None else "None") + "\n"
		ret += "2) start/end: " + str(self.start) + " --> " + str(self.end) + "\n"
		ret += "3) avoid water: " + str(self.avoid_water) + "\n"
		ret += "4) avoid terrain: " + str(self.avoid_forrest) + "\n"
		ret += "5) max grade: " + str(self.max_grade) + "\n"
		ret += "6) path precision (grid width): " + str(self.cell_width) + "\n"
		ret += "7) grade in direction: " + str(self.grade_in_direction) + "\n"
		ret += "8) single step: " + str(self.single_step)

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
	max_cost = 10000

	def __init__(self, user_settings, previous = None):
		self.user_settings = user_settings
		self.previous = previous
		self.boundary_points = None
		self.min_pixel_dist = self.__get_min_contour_dist()

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

	def __get_min_contour_dist(self):
		angle = Helper.convert_grade_to_angle(self.user_settings.max_grade)
		min_feet_dist = self.user_settings.get_contour_interval_dist() / math.tan(math.radians(angle))
		min_pixel_dist = int(min_feet_dist / self.user_settings.get_feet_per_pixel()) + 1

		return min_pixel_dist

	def find_path(self):
		# print("pixel start: " + str(self.user_settings.start))
		# print("pixel end: " + str(self.user_settings.end))
		# self.grid_start = self.grid.convert_pixel_to_grid_point(self.user_settings.cropped_img.start)
		# self.grid_end = self.grid.convert_pixel_to_grid_point(self.user_settings.cropped_img.end)

		if self.user_settings.grade_in_direction:
			self.grid_start = self.user_settings.cropped_img.start
			self.grid_end = self.user_settings.cropped_img.end
		else:
			self.grid_start = self.grid.convert_pixel_to_grid_point(self.user_settings.cropped_img.start)
			self.grid_end = self.grid.convert_pixel_to_grid_point(self.user_settings.cropped_img.end)

		return self.__calculate_path()

	def find_path_from_pixel_coords(self, start_point, end_point):
		if self.user_settings.grade_in_direction:
			self.grid_start = self.user_settings.cropped_img.start
			self.grid_end = self.user_settings.cropped_img.end
		else:
			self.grid_start = self.grid.convert_pixel_to_grid_point(self.user_settings.cropped_img.start)
			self.grid_end = self.grid.convert_pixel_to_grid_point(self.user_settings.cropped_img.end)

		return self.__calculate_path()

	def __calculate_path(self):
		open_nodes = [Node(self.grid_start)]
		closed_nodes = []

		count = 0

		print("grid start: " + str(self.grid_start))
		print("grid end: " + str(self.grid_end))
		print("total cells: " + str(self.grid.grid_resolution * self.grid.grid_resolution))

		while len(open_nodes) > 0:
			open_nodes.sort(key = lambda x: x.f, reverse=True)
			cur_node = open_nodes.pop()
			successors = self.__generate_successor_nodes(cur_node)

			for successor in successors:
				if self.user_settings.grade_in_direction:
					cur_point = self.grid.convert_pixel_to_grid_point(successor.coord)
					end_point = self.grid.convert_pixel_to_grid_point(self.grid_end)
				else:
					cur_point = successor.coord
					end_point = self.grid_end
				
				if cur_point == end_point:
					print("at end: " + str(cur_point))
					return successor

				self.__calculate_heuristic(cur_node, successor, self.grid_end)

				if not self.__position_already_reached_with_lower_heuristic(successor, open_nodes) and \
				not self.__position_already_reached_with_lower_heuristic(successor, closed_nodes):
					open_nodes.append(successor)

			closed_nodes.append(cur_node)

			count += 1

			if count % 5 == 0:
				print("current place: " + str(self.grid.convert_pixel_to_grid_point(cur_node.coord)))
				print("end_point: " + str(end_point))
				print("distance: " + str(self.__get_distance_between_points(cur_node.coord, self.grid_end)))
			# 	print("successors: " + str(len(successors)))
			# 	print("open nodes: " + str(len(open_nodes)))
			# 	print("closed nodes: " + str(len(closed_nodes)))
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
		# ie: the locations around cur_node upto two cells away, roughly every 22 degrees
		tl = Point(-1, -1)
		tlr = Point(-1, -2)
		tm = Point(0, -1)
		tr = Point(1, -1)
		trl = Point(1, -2)
		
		ml = Point(-1, 0)
		mlt = Point(-2, -1)
		mlb = Point(-2, 1)
		mr = Point(1, 0)
		mrt = Point(2, -1)
		mrb = Point(2, 1)
		
		bl = Point(-1, 1)
		blr = Point(-1, 2)
		bm = Point(0, 1)
		br = Point(1, 1)
		brl = Point(1, 2)
		
		all_directions = [tl, tlr, tm, tr, trl, ml, mlt, mlb, mr, mrt, mrb, bl, blr, bm, br, brl]
		directions = [tl, tm, tr, ml, mr, bl, bm, br]

		if self.user_settings.grade_in_direction:
			successors = list(map(lambda x: self.__get_node_by_grade_in_direction(cur_node, x), all_directions))
			# successors = list(map(lambda x: self.__get_node_by_grade_in_direction(cur_node, x), directions))
		elif self.user_settings.single_step:
			successors = list(map(lambda x: self.__get_node_single_step_in_direction(cur_node, x), all_directions))
			# successors = list(map(lambda x: self.__get_node_single_step_in_direction(cur_node, x), directions))
		else:
			successors = list(map(lambda x: self.__get_node_by_nearest_contour_in_direction(cur_node, x), all_directions))
			# successors = list(map(lambda x: self.__get_node_by_nearest_contour_in_direction(cur_node, x), directions))

		return list(filter(lambda x: x is not None and self.__point_within_boundary(x), successors))

	def __point_within_boundary(self, node):
		if self.previous is None or self.previous.boundary_points is None:
			return True

		if not self.user_settings.grade_in_direction:
			pixel_point = self.grid.convert_grid_to_pixel_point(node.coord)
		else:
			pixel_point = node.coord

		previous_grid_point = self.previous.grid.convert_pixel_to_grid_point(pixel_point)

		if previous_grid_point in self.previous.boundary_points:
			return True

		return False

	def __calculate_heuristic(self, cur_node, successor_node, end_point):
		successor_node.g = cur_node.g + self.__get_cost_between_nodes(cur_node, successor_node)
		successor_node.h = self.__get_distance_between_points(successor_node.coord, end_point)
		successor_node.f = successor_node.g + successor_node.h

	def __position_already_reached_with_lower_heuristic(self, cur_node, reached_nodes):
		temp = list(filter(lambda x: x.coord == cur_node.coord and x.f <= cur_node.f, reached_nodes))

		if len(temp) > 0:
			return True

		return False

	def __get_node_by_grade_in_direction(self, start_node, direction):
		# first_contour_point, second_contour_point = self.__get_two_nearest_contour_points_in_direction(start_node.coord, direction)

		# if first_contour_point is None or second_contour_point is None:
		# 	return None

		# distance = self.__get_distance_between_points(first_contour_point, second_contour_point)
		# grade = self.__get_grade_for_pixel_distance(distance)

		# node = Node(second_contour_point, parent=start_node, grade=grade)
		# node.num_contours = 2
		# node.avg_distance = distance
		# node.min_distance = distance

		return node

	def __get_two_nearest_contour_points_in_direction(self, point, direction):
		first = second = None
		
		first = self.__get_nearest_contour_point(point, direction)

		if first is not None:
			second = self.__get_nearest_contour_point(first, direction)

		return first, second

	def __get_nearest_contour_point(self, start_point, direction):
		cur_point = start_point
		next_point = Point(cur_point.x + direction.x, cur_point.y + direction.y)

		# while self.__point_in_image(next_point): 
		while self.__point_in_grid(self.grid.convert_pixel_to_grid_point(next_point)): 
			if self.__contour_at_point(next_point) or self.__contour_in_between(cur_point, direction):  
				return next_point 

			cur_point = next_point
			next_point = Point(cur_point.x + direction.x, cur_point.y + direction.y)
			
		return None

	def __point_in_image(self, point):
		w, h, c = self.user_settings.cropped_img.cv_image.shape

		if point.x >= 0 and point.x < w and point.y >= 0 and point.y < h:
			return True

		return False

	def __contour_in_between(self, start_point, direction):
		if abs(direction.x) > 1 or abs(direction.y > 1):
			points = self.__get_points_in_direction(start_point, direction)
			
			for p in points:
				if self.__contour_at_point(p):
					return True

		return False

	def __get_node_single_step_in_direction(self, start_node, direction):
		point = self.__get_next_point(start_node.coord, direction)

		if not self.__point_valid(point):
			return None
				
		start_point = self.grid.convert_grid_to_pixel_point(start_node.coord)
		end_point = self.grid.convert_grid_to_pixel_point(point)

		# grade, num_contours, avg_distance, min_distance = self.__get_grade_by_nearest_contour(start_point, end_point, direction)

		grade, num_contours, avg_distance, min_distance = self.__get_grade_by_nearest_two_contours(start_point, direction)

		node = Node(point, parent=start_node, grade = grade)
		
		node.num_contours = num_contours
		node.avg_distance = avg_distance
		node.min_distance = min_distance

		return node

	def __get_grade_by_nearest_two_contours(self, start_point, direction):
		first_contour_point, second_contour_point = self.__get_two_nearest_contour_points_in_direction(start_point, direction)

		if first_contour_point is None or second_contour_point is None:
			return 0, 0, 0, 0

		distance = self.__get_distance_between_points(first_contour_point, second_contour_point)
		grade = self.__get_grade_for_pixel_distance(distance)

		return grade, 2, distance, distance

	def __get_grade_by_nearest_contour(self, start_point, end_point, direction):
		cur_point = start_point + direction
		min_distance = self.grid.grid_resolution * self.grid.cell_width

		num_contours = 0
		avg_distance = 0

		reached_end_point = False

		while self.__point_in_grid(self.grid.convert_pixel_to_grid_point(cur_point)) and \
		not reached_end_point:
			if self.__contour_at_point(cur_point):
				cur_distance = self.__get_pixel_distance_between_points(start_point, cur_point)
				
				if cur_distance < min_distance:
					min_distance = cur_distance
				
				num_contours += 1
				avg_distance = ((avg_distance * (num_contours-1)) + cur_distance) / num_contours

				start_point = cur_point

			cur_point += direction

			if cur_point == end_point:
				reached_end_point = True

		# if avg_distance == 0:
		# 	grade = 0
		# else:
		# 	grade = self.__get_grade_for_pixel_distance(avg_distance)
		
		if min_distance == self.grid.grid_resolution * self.grid.cell_width:
			grade = 0
		else:
			grade = self.__get_grade_for_pixel_distance(min_distance)

		return grade, num_contours, avg_distance, min_distance

	def __contour_at_point(self, point):
		return self.user_settings.cropped_img.contours[point.x][point.y] > 0

	def __get_node_by_nearest_contour_in_direction(self, start_node, direction):
		start_point = start_node.coord
		nearest_node = None

		while nearest_node is None:
			cur_point = self.__get_next_point(start_point, direction)

			if self.__point_valid(cur_point):
				cur_cell = self.grid.get_cell(cur_point)

				if cur_cell.density > 0 or cur_point == self.grid_end:
					nearest_node = Node(cur_point, parent=start_node)
				else:
					start_point = cur_point
			elif not self.__point_in_grid(cur_point) and \
			not start_point == start_node.coord:
				nearest_node = Node(start_point, parent=start_node)
			else:
				break

		return nearest_node

	def __get_cost_between_nodes(self, start_node, end_node):
		grade_cost = self.__get_grade_cost(start_node, end_node)
		terrain_cost = self.__get_terrain_cost(end_node)
		distance = self.__get_distance_between_points(start_node.coord, end_node.coord)
		# print("grade_cost: " + str(grade_cost))
		# print("terrain_cost: " + str(terrain_cost))
		# print("distance_cost: " + str(distance))
		# print("-------")

		return grade_cost + terrain_cost + distance

	def __get_distance_between_points(self, start_point, end_point):
		if self.user_settings.grade_in_direction:
			return self.__get_pixel_distance_between_points(start_point, end_point)
		else:
			return self.__get_grid_distance_between_points(start_point, end_point)

	def __get_grid_distance_between_points(self, start, end):
		dx = abs(start.x - end.x)
		dy = abs(start.y - end.y)

		diag_cost = min(dx, dy) * PathFinder.diag_step_cost
		straight_cost = abs(dx  - dy) * PathFinder.step_cost

		distance = diag_cost + straight_cost

		return distance

	def __get_next_point(self, cur_point, direction):
		points = self.__get_points_in_direction(cur_point, direction)

		if len(points) > 0:
			for p in points:
				if not self.__point_valid(p):
					return None

		next_point = Point(cur_point.x + direction.x, cur_point.y + direction.y)

		return next_point

	def __get_points_in_direction(self, cur_point, direction):
		points = []

		if abs(direction.x) > 1:
			points.append(Point(cur_point.x + int(direction.x/2), cur_point.y + direction.y))
			points.append(Point(cur_point.x + int(direction.x/2), cur_point.y + int(direction.y/2)))

		elif abs(direction.y) > 1:
			points.append(Point(cur_point.x + int(direction.x/2), cur_point.y + int(direction.y/2)))
			points.append(Point(cur_point.x + direction.x, cur_point.y + int(direction.y/2)))

		return points

	def __point_valid(self, point):
		if point is None:
			return False

		if not self.__point_in_grid(point):
			return False

		return True

	def __get_grade_cost(self, start_node, end_node):
		if self.user_settings.single_step or self.user_settings.grade_in_direction:
			grade = end_node.grade
		else:
			grade = self.__get_grade_between_nodes(start_node, end_node)
		
		grade_cost = (grade / PathFinder.max_grade)
		
		if grade > self.user_settings.max_grade:
			grade_cost += PathFinder.max_cost

		return grade_cost * 10

	def __get_terrain_cost(self, node):
		if self.user_settings.grade_in_direction:
			# print(node.coord)
			cell_coord = self.grid.convert_pixel_to_grid_point(node.coord)
		else:
			cell_coord = node.coord

		# print(cell_coord)
		# print("---")
		cell = self.grid.get_cell(cell_coord)

		if self.user_settings.avoid_water:
			water_cost = cell.water_density + PathFinder.max_cost
		else:
			water_cost = cell.water_density

		if self.user_settings.avoid_forrest:
			forrest_cost = cell.forrest_density + PathFinder.max_cost
		else:
			forrest_cost = cell.forrest_density

		terrain_cost = water_cost + forrest_cost

		return terrain_cost * 10 

	def __get_grade_for_pixel_distance(self, pixel_dist):
		feet_dist = (pixel_dist * self.user_settings.get_feet_per_pixel())

		theta = math.atan(self.user_settings.get_contour_interval_dist() / feet_dist)
		angle = math.degrees(theta)

		grade = Helper.convert_angle_to_grade(angle)

		return grade

	def __get_grade_between_nodes(self, start_node, end_node):
		direction = Point(end_node.coord.x - start_node.coord.x, end_node.coord.y - start_node.coord.y)
		nearest_density_point = self.__get_nearest_density_point(start_node.coord, direction)

		start = self.grid.convert_grid_to_pixel_point(start_node.coord)
		end = self.grid.convert_grid_to_pixel_point(nearest_density_point)

		pixel_dist_to_density = self.__get_pixel_distance_between_points(start, end)
		grade = self.__get_grade_for_pixel_distance(pixel_dist_to_density)

		return grade

	def __get_nearest_density_point(self, start_point, direction):
		cur_point = start_point
		next_point = Point(cur_point.x + direction.x, cur_point.y + direction.y)

		while self.__density_between_points(cur_point, direction) == False: 
			if self.__point_in_grid(next_point):
				cur_point = next_point
				next_point = Point(cur_point.x + direction.x, cur_point.y + direction.y)
			else:
				break

		return next_point

	def __density_between_points(self, cur_point, direction):
		density = 0

		if abs(direction.x) > 1 or abs(direction.y > 1):
			points = self.__get_points_in_direction(cur_point, direction)
			
			for p in points: 
				density += self.__get_density_from_point(p)

		density += self.__get_density_from_point(cur_point)

		return density == 0

	def __get_density_from_point(self, point):
		if self.__point_in_grid(point):
			cell = self.grid.get_cell(point)
			return cell.density
		else:
			return 0

	def __get_pixel_distance_between_points(self, start, end):
		resized_distance = math.sqrt(math.pow(start.x-end.x, 2) + math.pow(start.y-end.y, 2))
		actual_distance = resized_distance / Helper.resize_factor

		return actual_distance

	def __point_in_grid(self, point):
		if point is None:
			return False

		if point.x >= 0 and point.x < self.grid.grid_resolution and point.y >= 0 and point.y < self.grid.grid_resolution:
			return True

		return False

	def __get_point_in_pixel_coord(self, point):
		if self.user_settings.grade_in_direction:
			return point

		return self.grid.convert_grid_to_pixel_point(point)

	def draw_path(self, node):
		path_img = self.user_settings.cropped_img.cv_image.copy()
		from_point = self.__get_point_in_pixel_coord(node.coord)

		while node.parent is not None:
			parent = node.parent
			to_point = self.__get_point_in_pixel_coord(parent.coord)

			print(self.grid.convert_pixel_to_grid_point(from_point))
			print(self.grid.convert_pixel_to_grid_point(to_point))
			print("grade: " + str(node.grade))
			print("num contours: " + str(node.num_contours))
			print("avg distance " + str(node.avg_distance))
			print("min distance: " + str(node.min_distance))
			print("-------")

			if node.grade > self.user_settings.max_grade:
				cv2.line(path_img,(from_point.x, from_point.y),(to_point.x, to_point.y),(0,0,255),2)
			elif node.grade == 0:
				cv2.line(path_img,(from_point.x, from_point.y),(to_point.x, to_point.y),(0,255,0),2)
			else:	
				cv2.line(path_img,(from_point.x, from_point.y),(to_point.x, to_point.y),(255,0,0),2)
			
			# cv2.circle(path_img, (from_point.x,from_point.y), 3, (100,255,255), -1)

			node = parent
			from_point = to_point

		print("min: " + str(self.min_pixel_dist))

		return path_img

	def set_boundary_points(self, node, distance):
		points = []
		
		cur_point = node.coord
		cur_parent = node.parent

		while cur_parent is not None:
			next_point = cur_parent.coord
			direction = self.__get_direction_vector_to_point(cur_point, cur_parent.coord)

			temp_point = Point(cur_point.x, cur_point.y)
			while not temp_point == cur_parent.coord:
				self.__add_neighbor_points(temp_point, distance, points)

				temp_point.x += direction.x
				temp_point.y += direction.y

			cur_point = next_point
			cur_parent = cur_parent.parent

		self.__add_neighbor_points(cur_point, distance, points)

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
				
				if self.__point_in_grid(cur_point) and cur_point not in points:
					points.append(cur_point)






























