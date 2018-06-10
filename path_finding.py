import cv2
import math
import operator

from helper_functions import Point, Helper
from topographic_map import TopographicMap
from cropped_image import CroppedImage
from grid import Grid, DensityGrid, GradeGrid
from user_settings import UserSettings, NodeMethod

class Node:
	def __init__(self, point, f=0.0, g=0.0, h=0.0, c=0.0, parent=None, grade = None):
		self.coord = point
		self.f = f
		self.g = g
		self.h = h
		self.c = c
		self.parent = parent
		self.grade = grade

		self.num_contours = 0
		self.avg_distance = 0
		self.min_distance = 0

		self.in_bounds = True # true if node isn't breaking user settings

	def __eq__(self, other):
		return self.coord == other.coord

	def __hash__(self):
		return hash((self.coord.x, self.coord.y))

class PathFinder:
	step_cost = 1
	diag_step_cost = math.sqrt(2)
	max_cost = 10000
	terrain_cost = 5000

	nearest_grade = 1
	single_step = 2
	nearest_density_cell = 3

	def __init__(self, user_settings, previous = None):
		self.user_settings = user_settings
		self.previous = previous
		self.boundary_points = None
		self.min_pixel_dist = self.__get_min_contour_dist()

		self.grid = DensityGrid(user_settings.cropped_img, user_settings.cell_width)
		# self.grade_grid = GradeGrid(user_settings.cropped_img, user_settings.cell_width, user_settings.max_grade, self.min_pixel_dist)
		self.grade_grid = GradeGrid(user_settings.cropped_img, self.min_pixel_dist, user_settings.max_grade, self.min_pixel_dist)

		print('min contour: ' + str(self.min_pixel_dist))
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

	def __convert_pixel_points(self, points, on_nearest_grade = False):
		converted_points = []

		if (self.user_settings.node_method == NodeMethod.nearest_grade) == on_nearest_grade:
			for p in points:
				converted_point = self.grid.convert_pixel_to_grid_point(p)
				converted_points.append(converted_point)
		else:
			converted_points = points

		return converted_points

	def find_path(self):
		points = self.__convert_pixel_points([self.user_settings.cropped_img.start, self.user_settings.cropped_img.end])
		self.grid_start = Point(points[0].x, points[0].y)
		self.grid_end = Point(points[1].x, points[1].y)

		# print(self.grid_start)
		# print(self.grid_end)
		# print(self.__point_in_grid(self.grid_start))
		# print(self.__point_in_grid(self.grid_end))
		# print(self.grid.resolution_x)
		# print(self.grid.resolution_y)
		# print(self.grid.cell_width)

		return self.__calculate_path()

	def find_path_from_pixel_coords(self, start_point, end_point):
		points = self.__convert_pixel_points([self.user_settings.cropped_img.start, self.user_settings.cropped_img.end])
		self.grid_start = points[0].copy()
		self.grid_end = points[1].copy()

		return self.__calculate_path()

	def __calculate_path(self):
		first_node = Node(self.grid_start)

		self.open_nodes = {first_node: first_node.f}
		self.closed_nodes = {}

		while len(self.open_nodes) > 0:
			cur_node = self.__get_minimum_node_key(self.open_nodes)
			self.open_nodes.pop(cur_node)
			successors = self.__generate_successor_nodes(cur_node)

			for successor in successors:
				points = self.__convert_pixel_points([successor.coord, self.grid_end], on_nearest_grade = True)
				cur_point = points[0]
				end_point = points[1]

				if cur_point == end_point:
					print("open nodes: " + str(len(self.open_nodes)))
					print("closed nodes: " + str(len(self.closed_nodes)))
					return successor
					
				self.__calculate_heuristic(cur_node, successor, self.grid_end)

				if not self.__position_already_reached_with_lower_heuristic(successor, self.open_nodes) and \
				not self.__position_already_reached_with_lower_heuristic(successor, self.closed_nodes):
					self.open_nodes[successor] = successor.f

			self.closed_nodes[cur_node] = cur_node.f

		return None

	def __get_minimum_node_key(self, open_nodes):
		sorted_nodes = sorted(open_nodes.items(), key=operator.itemgetter(1), reverse=True)
		minimum_node_key = sorted_nodes.pop()[0]

		return minimum_node_key

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

			if y < self.grid.resolution_y:
				lb = Node(Point(x, y), parent=cur_node)
				successors.append(lb)

		# successors to the right
		x = point.x + 1
		
		if x < self.grid.resolution_x:
			rm = Node(Point(x, point.y), parent=cur_node)
			successors.append(rm)

			y = point.y - 1

			if y >= 0:
				rt = Node(Point(x, y), parent=cur_node)
				successors.append(rt)

			y = point.y + 1

			if y < self.grid.resolution_y:
				rb = Node(Point(x, y), parent=cur_node)
				successors.append(rb)

		# top middle
		y = point.y + 1

		if y < self.grid.resolution_y:
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
		
		#all_directions = [tl, tlr, tm, tr, trl, ml, mlt, mlb, mr, mrt, mrb, bl, blr, bm, br, brl]
		directions = [tl, tm, tr, ml, mr, bl, bm, br]

		successors = []

		# for direction in all_directions:
		for direction in directions:
			successors.append(self.__get_node(cur_node, direction))

		# successors = list(map(lambda x: self.__get_node(cur_node, x), all_directions))
		# successors = list(map(lambda x: self.__get_node(cur_node, x), directions))

		return list(filter(lambda x: x is not None and self.__point_within_boundary(x), successors))

	def __point_within_boundary(self, node):
		if self.previous is None or self.previous.boundary_points is None:
			return True

		if self.user_settings.node_method == NodeMethod.nearest_grade:
			pixel_point = node.coord
		else:
			pixel_point = self.grid.convert_grid_to_pixel_point(node.coord)

		previous_grid_point = self.previous.grid.convert_pixel_to_grid_point(pixel_point)

		if previous_grid_point in self.previous.boundary_points:
			return True

		return False

	def __calculate_heuristic(self, cur_node, successor_node, end_point):
		successor_node.c = self.__get_cost_between_nodes(cur_node, successor_node)
		successor_node.g = cur_node.g + successor_node.c
		successor_node.h = self.__get_distance_between_points(successor_node.coord, end_point)
		successor_node.f = successor_node.g + successor_node.h
		# successor_node.f = successor_node.h + cur_node.g

	def __position_already_reached_with_lower_heuristic(self, cur_node, reached_nodes):
		if cur_node in reached_nodes.keys() and reached_nodes[cur_node] < cur_node.f:
			return True

		return False

	# node finding methods
	def __get_node(self, start_node, direction):
		if self.user_settings.node_method == NodeMethod.nearest_grade:
			return self.__get_node_by_grade_in_direction(start_node, direction)
		elif self.user_settings.node_method == NodeMethod.single_step:
			return self.__get_node_by_single_step_in_direction(start_node, direction)
		elif self.user_settings.node_method == NodeMethod.nearest_density_cell:
			return self.__get_node_by_nearest_contour_in_direction(start_node, direction)
		else:
			return None

	def __get_node_by_grade_in_direction(self, start_node, direction):
		first_contour_point, second_contour_point = self.__get_two_nearest_contour_points_in_direction(start_node.coord, direction)

		if first_contour_point is None or second_contour_point is None:
			return None

		distance = self.__get_distance_between_points(first_contour_point, second_contour_point)
		grade = self.__get_grade_for_pixel_distance(distance)

		node = Node(second_contour_point, parent=start_node, grade=grade)
		node.num_contours = 2
		node.avg_distance = distance
		node.min_distance = distance

		return node

	def __get_node_by_single_step_in_direction(self, start_node, direction):
		point = self.__get_next_point(start_node.coord, direction)

		if not self.__point_valid(point):
			return None
				
		# start_point = self.grid.convert_grid_to_pixel_point(start_node.coord)
		# end_point = self.grid.convert_grid_to_pixel_point(point)

		# grade, num, avg_dist, min_dist = self.__get_grade_by_contours_between_endpoints(start_point, end_point, direction)

		# node = Node(point, parent=start_node, grade = grade)
		
		# node.num_contours = num
		# node.avg_distance = avg_dist
		# node.min_distance = min_dist
		node = Node(point, parent=start_node)
		node.grade = self.__get_grade_by_grid_value(point, direction)

		return node

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

		if nearest_node is not None:
			nearest_node.grade = self.__get_grade_by_grid_value(nearest_node.coord)

		return nearest_node

	# grade calculation methods
	def __get_grade_by_nearest_two_contours(self, start_point, direction):
		first_contour_point, second_contour_point = self.__get_two_nearest_contour_points_in_direction(start_point, direction)

		if first_contour_point is None or second_contour_point is None:
			return 0, 0, 0, 0

		distance = self.__get_distance_between_points(first_contour_point, second_contour_point)
		grade = self.__get_grade_for_pixel_distance(distance)

		return grade, 2, distance, distance

	def __get_grade_by_contours_between_endpoints(self, start_point, end_point, direction):
		cur_point = start_point + direction
		min_distance = max(self.grid.resolution_x, self.grid.resolution_y) * self.grid.cell_width

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
		
		if min_distance == max(self.grid.resolution_x, self.grid.resolution_y) * self.grid.cell_width:
			grade = 0
		else:
			grade = self.__get_grade_for_pixel_distance(min_distance)

		return grade, num_contours, avg_distance, min_distance
#2205,140,935,852 r d
#2235,2425,2032,1395 r d
	def __get_grade_by_grid_value(self, point, direction):
		pixel_point = self.grid.convert_grid_to_pixel_point(point)
		cell = self.grade_grid.get_cell_from_pixel_point(pixel_point)

		# return cell.grade
		return self.grade_grid.get_cell_grade_in_direction(cell, direction)

	def __get_grade_by_nearst_contour(self, start_point, direction):
		nearest_density_point = self.__get_nearest_density_point(start_point, direction)

		start = self.grid.convert_grid_to_pixel_point(start_point)
		end = self.grid.convert_grid_to_pixel_point(nearest_density_point)

		pixel_dist_to_density = self.__get_pixel_distance_between_points(start, end)
		grade = self.__get_grade_for_pixel_distance(pixel_dist_to_density)

		return grade

	# other methods
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

	def __contour_at_point(self, point):
		return self.user_settings.cropped_img.contours[point.x][point.y] > 0

	def __get_cost_between_nodes(self, start_node, end_node):
		grade_cost = self.__get_grade_cost(start_node, end_node)
		terrain_cost = self.__get_terrain_cost(end_node)
		distance = self.__get_distance_between_points(start_node.coord, end_node.coord)

		# print(grade_cost)
		# print(terrain_cost)
		# print(distance)
		# print('------')
		total = ((grade_cost*10) + (10*terrain_cost) + distance) / 3
		# total = ((grade_cost*10) + (terrain_cost*10) + distance) / 3

		# if terrain_cost > 0:
		# 	print("terrain_cost: " + str(terrain_cost*10))
		# 	print("grade_cost: " + str(grade_cost*10))
		# 	print("distance: " + str(distance))
		# 	print("total: " + str(total))
		# 	print("-------")

		return total

	def __get_distance_between_points(self, start_point, end_point):
		if self.user_settings.node_method == NodeMethod.nearest_grade:
			return self.__get_pixel_distance_between_points(start_point, end_point)
		elif self.user_settings.node_method == NodeMethod.single_step:
			distance = self.__get_grid_distance_between_points(start_point, end_point)
			# distance /= PathFinder.diag_step_cost
			return distance 
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
		grade = end_node.grade

		grade_cost = grade / self.user_settings.max_grade

		if grade > self.user_settings.max_grade:
			grade_cost = PathFinder.max_cost + (grade_cost * grade_cost * PathFinder.max_cost)
			end_node.in_bounds = False
		# elif grade_cost > 0:
		# 	if grade_cost > 0:
		# 		print('grade: ' + str(grade_cost))
		# print('grade: ' + str(grade_cost))
		return grade_cost

	def __get_terrain_cost(self, node):
		cell_coord = self.__convert_pixel_points([node.coord], on_nearest_grade = True)[0]
		cell = self.grid.get_cell(cell_coord)

		# if cell.road_density < 0.05 or cell.water_density > 0.3:
		water_cost = cell.water_density / self.grid.max_water_density
		
		# if cell.road_density == 0:
		if cell.water_density > 0.3:
			water_cost += PathFinder.max_cost * PathFinder.max_cost
			node.in_bounds = False
		elif self.user_settings.avoid_water and cell.water_density > 0:
			water_cost = PathFinder.max_cost + (water_cost * water_cost * PathFinder.max_cost)
			node.in_bounds = False

		# if cell.road_density == 0 or cell.water_density > 0.3:
			
		# 	if self.user_settings.avoid_water and cell.water_density > 0:
		# 		water_cost = PathFinder.max_cost + (water_cost * water_cost * PathFinder.max_cost)
		# 		node.in_bounds = False
		# 	# elif cell.water_density > 0.0 and cell.water_density < 0.3:
		# 	# # elif cell.water_density > 0.1 and cell.water_density < 0.4:
		# 	# 	water_cost = PathFinder.terrain_cost + (water_cost * PathFinder.terrain_cost)
		# 	elif cell.water_density >= 0.3:
		# 	# elif cell.water_density >= 0.4:
		# 		water_cost += PathFinder.max_cost * PathFinder.max_cost
		# 		node.in_bounds = False
		
		forest_cost = cell.forest_density / self.grid.max_forest_density
		
		if self.user_settings.avoid_forest and cell.forest_density > 0:
			forest_cost = (forest_cost * forest_cost * PathFinder.max_cost) + PathFinder.max_cost
			node.in_bounds = False
		# elif cell.forest_density > 0:
		# 	forest_cost += PathFinder.terrain_cost + (forest_cost * PathFinder.terrain_cost)
		# else:
		# 	# forest_cost = 100*cell.forest_density
		# if forest_cost > 0:
		# 	print('forest: ' + str(forest_cost))
		
		terrain_cost = water_cost + forest_cost

		return terrain_cost 

	def __get_grade_for_pixel_distance(self, pixel_dist):
		feet_dist = (pixel_dist * self.user_settings.get_feet_per_pixel())

		theta = math.atan(self.user_settings.get_contour_interval_dist() / feet_dist)
		angle = math.degrees(theta)

		grade = Helper.convert_angle_to_grade(angle)

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

		if point.x >= 0 and point.x < self.grid.resolution_x and point.y >= 0 and point.y < self.grid.resolution_y:
			return True

		return False

	def __get_point_in_pixel_coord(self, point):
		if self.user_settings.node_method == NodeMethod.nearest_grade:
			return point

		return self.grid.convert_grid_to_pixel_point(point)

	def draw_costs(self, cost_img=None):
		if cost_img is None:
			cost_img = self.user_settings.cropped_img.cv_image.copy()

		points = set()
		max_c = 0
		max_cb = 0
		max_cb2 = 0

		for n in list(self.open_nodes.keys()) + list(self.closed_nodes.keys()):
			if n.c > PathFinder.max_cost * PathFinder.max_cost:
				if n.c > max_cb2:
					max_cb2 = n.c
			elif n.c > PathFinder.max_cost:
				if n.c > max_cb:
					max_cb = n.c
			else:
				if n.c > max_c:
					max_c = n.c

			points.add(n)

		print("points: " +  str(len(points)) + ": " + str(self.grid.cell_width))
		
		offset = int(self.grid.cell_width/2)

		a = 0
		b = 0
		c = 0
		for n in points:
			if n.c >= PathFinder.max_cost*PathFinder.max_cost:
				cost = n.c / (max_cb2 if max_cb2 > 0 else 1)
				# color = (int(255 * cost), int(255 * (1-cost)), 0)
				color = (150 + int(105 * cost), 0, 0)
				a+=1
				# print('a')
			elif n.c > PathFinder.max_cost:
				cost = n.c / max_cb
				# color = (0, int(255 * cost), int(255 * (1-cost)))
				color = (0, 0, 100 + int(155 * cost))
				b+=1
				# print('b')
			else:
				cost = n.c / max_c
				# color = (int(255 * (1-cost)), 0, int(255 * cost))
				color = (0, 0 + int(255 * cost), 0)
				c+=1
				# print('c')

			p = self.grid.convert_grid_to_pixel_point(n.coord)

			pt1 = (p.x - offset, p.y - offset)
			pt2 = (p.x + offset, p.y + offset)

			# print(n.c)
			# print(cost)
			# print(color)
			# print('----')
			# print("(" + str(p.x) + ", " + str(p.y) + ")")
			# cv2.rectangle(cost_img, pt1, pt2, color, thickness=-1)
			cv2.circle(cost_img, (p.x,p.y), int(self.grid.cell_width/2), color, 2)

		print(a)
		print(b)
		print(c)
		print('------')
		print(max_cb2)
		print(max_cb)
		print(max_c)
		print('-----')
		return cost_img
#2205,140,935,852 r d
	def draw_path(self, node, path_img = None):
		if path_img is None:
			path_img = self.user_settings.cropped_img.cv_image.copy()
		from_point = self.__get_point_in_pixel_coord(node.coord)

		while node.parent is not None:
			parent = node.parent
			to_point = self.__get_point_in_pixel_coord(parent.coord)

			# print("f: " + str(node.f) + ", g: " + str(node.g) + ", h: " + str(node.h))

			if node.grade is not None:
				cv2.line(path_img,(from_point.x, from_point.y),(to_point.x, to_point.y),(0,0,255),2)
				# if node.grade > self.user_settings.max_grade:
				# 	cv2.line(path_img,(from_point.x, from_point.y),(to_point.x, to_point.y),(0,0,255),2)
				# elif node.grade == 0:
				# 	cv2.line(path_img,(from_point.x, from_point.y),(to_point.x, to_point.y),(0,255,0),2)
				# else:	
				# 	cv2.line(path_img,(from_point.x, from_point.y),(to_point.x, to_point.y),(255,0,0),2)
			else:
				cv2.line(path_img,(from_point.x, from_point.y),(to_point.x, to_point.y),(255,0,0),2)

			node = parent
			from_point = to_point

		return path_img

	# (return straight path length, path length) 
	# (max grade, avg grade)
	# (path length in high grade, path length in water, path length in forest)
	def get_path_stats(self, node):
		start = self.user_settings.cropped_img.start
		end = self.user_settings.cropped_img.end

		feet_per_pixel = self.user_settings.get_feet_per_pixel()
		straight_distance = self.__get_pixel_distance_between_points(start, end) * feet_per_pixel
		total_distance = 0
		
		total_grade = 0
		num_nodes = 0
		max_grade = 0

		total_cost = 0
		total_cost_in_bounds = 0
		max_cost = 0
		max_cost_in_bounds = 0

		total_high_grade_steps = 0
		total_water_steps = 0
		total_forest_steps = 0

		while node.parent is not None:
			num_nodes += 1
			
			start = self.grid.convert_grid_to_pixel_point(node.coord)
			end = self.grid.convert_grid_to_pixel_point(node.parent.coord)
			cur_distance = self.__get_pixel_distance_between_points(start, end) * feet_per_pixel

			total_distance += cur_distance

			cell = self.grid.get_cell(node.coord) 
			if cell.water_density > 0 and cell.road_density == 0:
				total_water_steps += 1
			if cell.forest_density > 0:
				total_forest_steps += 1

			total_cost += node.g
			if node.g > max_cost:
				max_cost = node.g

			if node.in_bounds:
				total_cost_in_bounds += node.g
				if node.g > max_cost_in_bounds:
					max_cost_in_bounds = node.g

			if node.grade is not None:
				total_grade += node.grade

				if node.grade > max_grade:
					max_grade = node.grade

				if node.grade > self.user_settings.max_grade:
					total_high_grade_steps += 1
	
			node = node.parent

		avg_grade = total_grade / num_nodes

		avg_cost = total_cost / num_nodes
		avg_cost_in_bounds = total_cost_in_bounds / num_nodes

		algorithm_info = AlgorithmInfo(avg_cost, max_cost, avg_cost_in_bounds, max_cost_in_bounds)
		path_length_info = PathLengthInfo(straight_distance, total_distance, num_nodes)
		grade_info = GradeInfo(max_grade, avg_grade)
		terrain_info = TerrainLengthInfo(total_high_grade_steps, total_water_steps, total_forest_steps)

		return algorithm_info, path_length_info, grade_info, terrain_info

	def set_boundary_points(self, node, distance):
		points = {}
		
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
				
				if self.__point_in_grid(cur_point):
					points[cur_point] = True

class AlgorithmInfo:
	def __init__(self, avg_cost, max_cost, avg_cost_in_bounds, max_cost_in_bounds):
		self.avg_cost_in_bounds = avg_cost_in_bounds
		self.max_cost_in_bounds = max_cost_in_bounds
		self.avg_cost = avg_cost
		self.max_cost = max_cost
class PathLengthInfo:
	def __init__(self, straight_path_length, path_length, num_nodes):
		self.straight_path_length = straight_path_length
		self.path_length = path_length
		self.num_nodes = num_nodes
class GradeInfo:
	def __init__(self, max_grade, avg_grade):
		self.max_grade = max_grade
		self.avg_grade = avg_grade
class TerrainLengthInfo:
	def __init__(self, high_grade_steps, water_steps, forest_steps):
		self.high_grade_steps = high_grade_steps
		self.water_steps = water_steps
		self.forest_steps = forest_steps