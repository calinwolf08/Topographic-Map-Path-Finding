import math, cv2
from helper_functions import Point

class Cell:
	def __init__(self, point):
		self.point = point

class DensityCell(Cell):
	def __init__(self, point):
		Cell.__init__(self, point)
		self.density = 0
		self.water_density = 0
		self.forest_density = 0
		self.road_density = 0

class GradeCell(Cell):
	def __init__(self, point):
		Cell.__init__(self, point)
		self.grade = 0
		self.r_grade = None
		self.b_grade = None
		self.rb_grade = None

class Grid:
	def __init__(self, cropped_img, cell_width):
		self.cropped_img = cropped_img
		self.cell_width = cell_width

		rows, cols = self.cropped_img.image_masks.topo_mask.shape
		self.resolution_y = int(rows / self.cell_width)
		self.resolution_x = int(cols / self.cell_width)

		self.initialize_array()

	def get_cell(self, point):
		if point.y >= 0 and point.y < len(self.array):
			if point.x >= 0 and point.x < len(self.array[point.y]):
				return self.array[point.y][point.x]		

		print("bad point:")
		print(len(self.array))
		print(len(self.array[0]))
		print(point)
		return None

	def get_cell_from_pixel_point(self, point):
		grid_point = self.convert_pixel_to_grid_point(point)

		return self.get_cell(grid_point)

	def convert_pixel_to_grid_point(self, point):
		x = int(point.x / self.cell_width)
		y = int(point.y / self.cell_width)

		if x >= self.resolution_x:
			x = self.resolution_x - 1

		if y >= self.resolution_y:
			y = self.resolution_y - 1

		return Point(x,y)

	def convert_grid_to_pixel_point(self, point):
		x = (point.x * self.cell_width) + int(self.cell_width/2)
		y = (point.y * self.cell_width) + int(self.cell_width/2)

		return Point(x,y)

	def initialize_array(self):
		self.array = [[Cell(Point(x,y)) for x in range(self.resolution_x)] for y in range(self.resolution_y)]
		self.extra_array = [[Cell(Point(x,y)) for x in range((2*self.resolution_x)-1)] for y in range((2*self.resolution_y)-1)]

	def get_image_density_at_cell(self, cell, mask):
		cell_image = self.get_cell_covered_image(cell.point, mask)

		non_zero_pixels = cv2.countNonZero(cell_image)
		total_pixels = self.cell_width * self.cell_width

		return non_zero_pixels / total_pixels

	def get_cell_covered_image(self, point, mask):
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
			cv2.line(copy,(0,i),(col,i),(255,0,0),lineThickness)
			i += self.cell_width

		j = 0
		while j <= col - self.cell_width:
			cv2.line(copy,(j,0),(j,row),(255,0,0),lineThickness)
			j += self.cell_width

		return copy

class GradeGrid(Grid):
	def __init__(self, cropped_img, cell_width, max_grade, min_contour_distance):
		self.max_grade = max_grade
		self.min_contour_distance = min_contour_distance
		self.density_for_max_grade = self.__get_density_for_max_grade(self.min_contour_distance)

		Grid.__init__(self, cropped_img, cell_width)

	def initialize_array(self):
		self.array = [[GradeCell(Point(x,y)) for x in range(self.resolution_x)] for y in range(self.resolution_y)]

		for col in range(self.resolution_x):
			for row in range(self.resolution_y):
				self.__initialize_cell_grade(self.get_cell(Point(col, row)))

	def __get_density_for_max_grade(self, min_contour_distance):
		total_cell_pixels = math.pow(min_contour_distance, 2)
		pixels_for_max_grade = min_contour_distance * 3
		density_for_max_grade = pixels_for_max_grade / total_cell_pixels

		return density_for_max_grade

	def get_density_from_cell_image(self, cell_image):
		non_zero_pixels = cv2.countNonZero(cell_image)
		total_pixels = self.cell_width * self.cell_width

		return non_zero_pixels / total_pixels

	def get_image_density_at_cell(self, cell, mask):
		cell_image = self.get_cell_covered_image(cell.point, mask, 0, 0)
		density = self.get_density_from_cell_image(cell_image)

		if cell.point.x < self.resolution_x - 1:
			r_cell_image = self.get_cell_covered_image(cell.point, mask, int(self.cell_width/2), 0)
			r_density = self.get_density_from_cell_image(r_cell_image)
		else:
			r_density = None

		if cell.point.y < self.resolution_y - 1:
			b_cell_image = self.get_cell_covered_image(cell.point, mask, 0, int(self.cell_width/2))
			b_density = self.get_density_from_cell_image(b_cell_image)
		else:
			b_density = None

		if r_density is not None and b_density is not None:
			rb_cell_image = self.get_cell_covered_image(cell.point, mask, int(self.cell_width/2), int(self.cell_width/2))
			rb_density = self.get_density_from_cell_image(rb_cell_image)
		else:
			rb_density = None

		return (density, r_density, b_density, rb_density)

	def get_cell_covered_image(self, point, mask, x_offset, y_offset):
		minY = point.y * self.cell_width + y_offset
		maxY = minY + self.cell_width
		minX = point.x * self.cell_width + x_offset
		maxX = minX + self.cell_width

		cell_image = mask[minY:maxY, minX:maxX]

		return cell_image

	def __initialize_cell_grade(self, cell):
		(density, r_density, b_density, rb_density) = self.get_image_density_at_cell(cell, self.cropped_img.contours)
		cell.grade = self.__get_cell_grade_from_density(density)
		cell.r_grade = self.__get_cell_grade_from_density(r_density)
		cell.b_grade = self.__get_cell_grade_from_density(b_density)
		cell.rb_grade = self.__get_cell_grade_from_density(rb_density) 

	def __get_cell_grade_from_density(self, density):
		if density is None:
			return None

		grade_density = density / self.density_for_max_grade
		grade = grade_density * self.max_grade

		return grade
	
	def get_cell_grade_in_direction(self, cell, direction):
		next_p = cell.point + direction

		if next_p.x < 0 or next_p.x >= self.resolution_x or next_p.y < 0 or next_p.y >= self.resolution_y:
			return cell.grade

		# r
		if direction == Point(1,0):
			# return (cell.grade + cell.r_grade) / 2
			return max(cell.grade,cell.r_grade)
		# b
		elif direction == Point(0, 1):
			# return (cell.grade + cell.b_grade) / 2
			return max(cell.grade, cell.b_grade)
		# br
		elif direction == Point(1, 1):
			# return (cell.grade + cell.rb_grade) / 2
			return max(cell.grade, cell.rb_grade)
		# l
		elif direction == Point(-1, 0):
			c = self.get_cell(cell.point + direction)
			# return (cell.grade + c.r_grade) / 2
			return max(cell.grade, c.r_grade)
		# t
		elif direction == Point(0, -1):
			c = self.get_cell(cell.point + direction)
			# return (cell.grade + c.b_grade) / 2
			return max(cell.grade, c.b_grade)
		# bl
		elif direction == Point(-1, 1):
			c = self.get_cell(cell.point + Point(-1,0)) # need point to left
			# return (cell.grade + c.rb_grade) / 2
			return max(cell.grade, c.rb_grade)
		# tr
		elif direction == Point(1, -1):
			c = self.get_cell(cell.point + Point(0,-1)) # need point above
			# return (cell.grade + c.rb_grade) / 2
			return max(cell.grade, c.rb_grade)
		# tl
		elif direction == Point(-1, -1):
			c = self.get_cell(cell.point + Point(-1,-1)) # need point tl
			# return (cell.grade + c.rb_grade) / 2
			return max(cell.grade, c.rb_grade)
		
		return cell.grade

	def add_grade_to_image(self, img):
		copy = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		for c in range(self.resolution_x):
			for r in range(self.resolution_y):
				x = (c * self.cell_width) + int(self.cell_width / 2)
				y = (r * self.cell_width) + int(self.cell_width / 2)

				cell = self.get_cell(Point(c,r))

				if cell.grade > self.max_grade:
					cv2.circle(copy, (x,y), int(self.cell_width/2), (0,255,255), 1)
				# else:
				# 	cv2.circle(copy, (y,x), int(self.cell_width/2), (122,255,255), 1)

		copy = cv2.cvtColor(copy, cv2.COLOR_HSV2BGR)

		return copy

class DensityGrid(Grid):
	def __init__(self, cropped_img, cell_width):
		self.max_density = 0
		self.max_water_density = 0
		self.max_forest_density = 0
		Grid.__init__(self, cropped_img, cell_width)

	def initialize_array(self):
		self.array = [[DensityCell(Point(x,y)) for x in range(self.resolution_x)] for y in range(self.resolution_y)]

		for col in range(self.resolution_x):
			for row in range(self.resolution_y):
				self.__initialize_cell_densities(self.get_cell(Point(col, row)))

	def __initialize_cell_densities(self, cell):
		cell.density = self.get_image_density_at_cell(cell, self.cropped_img.contours)

		if cell.density > self.max_density:
			self.max_density = cell.density

		cell.water_density = self.get_image_density_at_cell(cell, self.cropped_img.image_masks.blue_mask)

		if cell.water_density > self.max_water_density:
			self.max_water_density = cell.water_density

		cell.forest_density = self.get_image_density_at_cell(cell, self.cropped_img.image_masks.green_mask)

		if cell.forest_density > self.max_forest_density:
			self.max_forest_density = cell.forest_density

		cell.road_density = self.get_image_density_at_cell(cell, self.cropped_img.image_masks.black_mask)

	def add_density_to_image(self, img):
		copy = img.copy()

		for c in range(self.resolution_x):
			for r in range(self.resolution_y):
				x = (c * self.cell_width) + int(self.cell_width / 2)
				y = (r * self.cell_width) + int(self.cell_width / 2)

				cell = self.get_cell(Point(c,r))

				if cell.forest_density > 0:
					cv2.circle(copy, (x,y), int(self.cell_width/2), (0,255,0), 1)

				if cell.water_density > 0:
					if cell.road_density > 0:
						cv2.circle(copy, (x,y), int(self.cell_width/2), (255,0,255), 1)
					else:
						cv2.circle(copy, (x,y), int(self.cell_width/2), (255,0,0), 1)

		return copy

	def add_boundary_to_image(self, img, boundary_points):
		copy = img.copy()

		for point in boundary_points:
			pt = self.convert_grid_to_pixel_point(point)
			cv2.circle(copy, (pt.x,pt.y), int(self.cell_width/2), (255,0,255), 2)

		return copy























