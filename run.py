import cv2
import time
import sys
import os

from topographic_map import TopographicMap
from path_finding import PathFinder, GradeGrid
from path_finding import UserSettings
from cropped_image import CroppedImage
from user_settings import UserSettings

from helper_functions import Point
from draw_route import draw_save_route

# TODO: testing framework
	# comparing path images
		# pixel minimum threshold
		# compare output of different approaches
	# compare run times
		# differing path lengths
		# differing cropped image approaches
			# single image, start -> end
			# multi image in parts
		# A* grid densities approach

# TODO: improve path finding
	# get scale info
		# use hardcoded value
			# estimate error
		# pull from scale on image
		# use meters from equator values
	# improve heuristic to use terrain, slope accurately
	# somehow incorporate hiking techniques
		# zig zags up slope (may get this for free)
	# determine altitude, ascent vs descent
	# different grid approaches
		# start with wide grid move to denser grid for precision
		# single grid or multiple iteratively

# TODO: improve efficiency
	# concurrency where applicable?

# TODO: dealing with thick red lines

# Select images to test with (5)
	# select paths for each image (5)
		# run with +w+f, +w-f, -w+f, -w-f
			# 100 total tests
			# save path, density, grade #, contours, boundary, red, blue, black, green
			# 300 #900 total images

class RunTime:
	def __init__(self):
		self.path_times = []
		self.total_time = 0

class DataEntry:
	def __init__(self, run_time, path_length_info, grade_info, terrain_info, user_settings):
		self.run_time = run_time
		self.path_length_info = path_length_info
		self.grade_info = grade_info
		self.terrain_info = terrain_info
		self.avoid_water = user_settings.avoid_water
		self.avoid_forest = user_settings.avoid_forest
		self.filename = user_settings.topo_map.filename[5:-4]
		self.points = UserInterface.points_to_path_string(user_settings)

	def __str__(self):
		ret = self.filename + "," + self.points + ","

		ret += str(self.run_time.path_times[0]) + "," + str(self.run_time.path_times[1]) + ","
		if len(self.run_time.path_times) > 2:
			ret += str(self.run_time.path_times[2])
		ret += ","
		if len(self.run_time.path_times) > 3:
			ret += str(self.run_time.path_times[3])
		ret += ","
		ret += str(self.run_time.total_time) + ","
		
		ret += str(self.path_length_info.straight_path_length) + "," + str(self.path_length_info.path_length) + "," + str(self.path_length_info.num_nodes) + ","
		
		ret += str(self.grade_info.max_grade) + "," + str(self.grade_info.avg_grade) + ","

		ret += str(self.terrain_info.high_grade_steps) + "," + str(self.terrain_info.water_steps) + "," + str(self.terrain_info.forest_steps) + ","

		ret += str(self.avoid_water) + "," + str(self.avoid_forest)

		return ret

class UserInterface:
	def __init__(self):
		self.filename = None
		self.commands = []
		self.data = []
		self.run_time = None

	def start(self):
		self.user_settings = UserSettings()
		self.path_finder = None
		self.filename = self.get_filename()
		self.set_user_settings()

		self.interface_loop()

	def interface_loop(self):
		run = True

		while run:
			if len(self.commands) == 0:
				print("\n" + str(self.user_settings) + "\n")
				message = "Enter # to change a setting, r to run path finding, c to draw route, d to display images, p to print path stats, q to quit.\n"
				message += "Or enter commands as a sequence.\n"
				message += "(ex: 1name 20,0,1,1 3 -> filename = name, start = (0,0), end = (1,1), flip avoid water)\n"
				command = input(message)
				self.commands = command.split() 
			
			full_command = self.commands.pop(0)
			command = full_command[0]
			option = full_command[1:]

			if command == "1":
				self.filename = self.get_filename(option)
				self.set_user_settings()
			elif command == "2":
				self.find_start_end_points(option)
			elif command == "3":
				self.user_settings.avoid_water = not self.user_settings.avoid_water
			elif command == "4":
				self.user_settings.avoid_forest = not self.user_settings.avoid_forest
			elif command == "5":
				self.user_settings.max_grade = self.get_max_grade(option)
			elif command == "6":
				self.user_settings.cell_width = self.get_cell_width(option)
			elif command == "7":
				self.get_node_method(option)
			elif command == "8":
				self.user_settings.save_image = not self.user_settings.save_image
			elif command == "d":
				self.display_images()
			elif command == "r":
				self.path_finder = None
				self.run_time = RunTime()
				start = time.time()
				self.run()
				end = time.time()
				self.run_time.total_time = end-start
				print("total path finding time: " + str(end - start))
			elif command == "c":
				draw_save_route(self.user_settings.start, self.user_settings.end, self.user_settings.topo_map)
			elif command == "p":
				if self.path is not None:
					self.print_stats()
			elif command == "q":
				run = False
			else:
				print("command invalid")
		
		if self.user_settings.save_image and len(self.data) > 0:
			self.write_data()

	def run(self):
		self.find_path_with_resolution(30, 3)
		self.find_path_with_resolution(20, 2)

		if self.user_settings.save_image:
			self.save_path()

	def find_path_with_resolution(self, resolution, boundary_distance):
		self.user_settings.cell_width = resolution
		self.path_finder = PathFinder(self.user_settings, previous = self.path_finder)

		start = time.time()
		self.path = self.path_finder.find_path()
		end = time.time()

		self.run_time.path_times.append(end - start)

		print("path finding time (" + str(resolution) + "):" + str(end - start))

		if self.path is None:
			print("no path found")
		else:
			self.path_finder.set_boundary_points(self.path, distance = boundary_distance)
			# self.display_images()

	def display_images(self):
		if self.path_finder is None:
			cv2.imshow("image" + str(time.time()), self.user_settings.cropped_img.cv_image)
			# self.user_settings.cropped_img.image_masks.show_masks()
		else:
			path_img = self.path_finder.draw_path(self.path)

			# image = self.user_settings.cropped_img.contours.copy()
			# image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
			# grid_img = self.path_finder.grid.add_grid_to_image(image, 1)
			density = self.path_finder.grid.add_density_to_image(path_img)
			grade = self.path_finder.grade_grid.add_grade_to_image(path_img)
			# boundary_img = self.path_finder.grid.add_boundary_to_image(grid_img, self.path_finder.boundary_points)

			# cv2.imshow("contours" + str(time.time()), self.user_settings.cropped_img.contours)
			# cv2.imshow("boundary" + str(time.time()), boundary_img)
			# cv2.imshow("grade" + str(time.time()), grade)
			# cv2.imshow("path" + str(time.time()), path_img)

			self.draw_image("grade", grade)
			self.draw_image("density", density)
			self.draw_image("path", path_img)

		cv2.waitKey(1)
		cv2.destroyAllWindows()

	def print_stats(self):
		feet_per_mile = 5280
		path_length_info, grade_info, terrain_info = self.path_finder.get_path_stats(self.path)
		print("path length info:")
		print("\tstraight path length: " + str(path_length_info.straight_path_length / feet_per_mile) + " miles")
		print("\tpath length: " + str(path_length_info.path_length / feet_per_mile) + " miles")
		print("\tnum nodes: " + str(path_length_info.num_nodes))
		print("grade info:")
		print("\tmax_grade: " + str(grade_info.max_grade))
		print("\tavg_grade: " + str(grade_info.avg_grade))
		print("terrain info:")
		print("\thigh grade steps: " + str(terrain_info.high_grade_steps))
		print("\twater steps: " + str(terrain_info.water_steps))
		print("\tforest steps: " + str(terrain_info.forest_steps))

	def save_path(self):
		path_img = self.path_finder.draw_path(self.path)
		density = self.path_finder.grid.add_density_to_image(path_img)
		grade = self.path_finder.grade_grid.add_grade_to_image(path_img)
		# contours = self.user_settings.cropped_img.contours
		# boundary = self.path_finder.grid.add_boundary_to_image(path_img, self.path_finder.boundary_points)

		# image_masks = self.user_settings.cropped_img.image_masks

		# blue_mask = image_masks.blue_mask
		# black_mask = image_masks.black_mask
		# red_mask = image_masks.red_mask
		# green_mask = image_masks.green_mask

		path = "images/"+self.filename[5:-4] + "/" + UserInterface.points_to_path_string(self.user_settings) +\
			"/" + UserInterface.terrain_to_path_string(self.user_settings)
		
		if not os.path.exists(path):
			os.makedirs(path)

		cv2.imwrite(path + '/path.png', path_img)
		cv2.imwrite(path + '/density.png', density)
		cv2.imwrite(path + '/grade.png', grade)
		# cv2.imwrite(path + '/boundary.png', boundary)
		# cv2.imwrite(path + '/contours.png', contours)
		# cv2.imwrite(path + '/blue.png', blue_mask)
		# cv2.imwrite(path + '/black.png', black_mask)
		# cv2.imwrite(path + '/red.png', red_mask)
		# cv2.imwrite(path + '/green.png', green_mask)

		path_length_info, grade_info, terrain_info = self.path_finder.get_path_stats(self.path)
		data_entry = DataEntry(self.run_time, path_length_info, grade_info, terrain_info, self.user_settings)

		self.data.append(data_entry)

	def write_data(self):
		out = open('test_data.csv', 'w')

		header = "filename,start-end points (x-ytox-y),"
		header += "grid size 30 time (s),grid size 20 time (s),grid size 10 time (s),grid size 5 time (s),total time (s),"
		header += "straight distance (ft),path length (ft),num nodes,"
		header += "max grade,avg grade,"
		header += "high grade steps,water steps,forest steps,"
		header += "avoid water,avoid forest\n"
		out.write(header)

		for d in self.data:
			out.write(str(d))
			out.write("\n")
		
		out.close()
	@staticmethod
	def points_to_path_string(user_settings):
		start = user_settings.cropped_img.start
		end = user_settings.cropped_img.end
		return str(start.x) + "_" + str(start.y) + "to" + str(end.x) + "_" + str(end.y)
	
	@staticmethod
	def terrain_to_path_string(user_settings):
		ret = ""

		if user_settings.avoid_water:
			ret += "water"
		if user_settings.avoid_forest:
			ret += "forest"

		return ret if len(ret) > 0 else "nothing"

	def draw_image(self, img_name, img):
		h = img.shape[0]
		w = img.shape[1]

		max_h = 500
		num = int(h / max_h)

		for i in range(num - 1):
			start = i*max_h
			end = start + max_h
			cv2.imshow(str(i) + '_' + img_name + str(time.time()), img[start:end,:])

		start = (num-1) * max_h
		cv2.imshow(str(num-1) + '_' + img_name + str(time.time()), img[start:,:])

	def set_user_settings(self):
		self.user_settings.set_topo_map(self.filename)
		self.user_settings.find_cropped_image()

	def find_start_end_points(self, option=None):
		if option is None or len(option) == 0:
			self.user_settings.find_start_end_points()
		else:
			vals = option.split(',')
			self.user_settings.start = Point(int(vals[0]), int(vals[1]))
			self.user_settings.end = Point(int(vals[2]), int(vals[3]))

	def get_filename(self, option=None):
		if option is None or len(option) == 0:
			if self.filename is None and len(sys.argv) > 1:
				filename = sys.argv[1]
			else:
				filename = input("Enter the topographic map filename: ")
		else:
			filename = option
		return filename

	def get_max_grade(self, option=None):
		if option is None or len(option) == 0:
			has_new_grade = False
			grade = None

			while has_new_grade == False:

				grade = input("Enter your maximum grade: ")

				if grade.isdigit() and int(grade) <= 100:
					has_new_grade = True
				else:
					print("Please enter a grade between 0 and 100")
		else:
			grade = option
			assert(grade.isdigit() and int(grade) <= 100)

		return int(grade)

	def get_cell_width(self, option=None):
		if option is None or len(option) == 0:
			has_new_width = False
			width = None

			while has_new_width == False:

				width = input("Enter your grid cell width: ")

				if width.isdigit() and int(width) <= 100:
					has_new_width = True
				else:
					print("Please enter a width greater than 0")
		else:
			width = option
			assert(width.isdigit() and int(width) <= 100)

		return int(width)

	def get_node_method(self, option=None):
		if option is None or len(option) == 0:
			print("\t1) nearest grade")
			print("\t2) single step")
			print("\t3) nearest density cell")

			method = input("Enter number for node method:\n")
		else:
			method = option

		if method == "1":
			self.user_settings.node_method = PathFinder.nearest_grade
		elif method == "2":
			self.user_settings.node_method = PathFinder.single_step
		elif method == "3":
			self.user_settings.node_method = PathFinder.nearest_density_cell
		else:
			print("invalid method chosen: " + method + "\n")
			self.get_node_method()

if __name__ == '__main__':
	user_interface = UserInterface()
	user_interface.start()

	# run("MountStickney.jpg")
	# run("SanLuisObispo.jpg")
	# run("Snoqualmie.jpg")


	










