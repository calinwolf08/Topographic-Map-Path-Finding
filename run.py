import cv2
import time
import sys

from topographic_map import TopographicMap
from path_finding import PathFinder, GradeGrid
from path_finding import UserSettings
from cropped_image import CroppedImage
from user_settings import UserSettings

from helper_functions import Point

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

class UserInterface:
	def __init__(self):
		self.filename = None
		self.commands = []

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
				message = "Enter # to change a setting, r to run path finding, d to display images, q to quit.Or enter commands as a sequence.\n"
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
				self.user_settings.avoid_forrest = not self.user_settings.avoid_forrest
			elif command == "5":
				self.user_settings.max_grade = self.get_max_grade(option)
			elif command == "6":
				self.user_settings.cell_width = self.get_cell_width(option)
			elif command == "7":
				self.get_node_method(option)
			elif command == "d":
				self.display_images()
			elif command == "r":
				self.path_finder = None
				start = time.time()
				self.run()
				end = time.time()
				print("total path finding time: " + str(end - start))
			elif command == "q":
				run = False
			else:
				print("command invalid")

	# circular, path length vs time spent
	def run(self):
		self.find_path_with_resolution(30, 3)
		self.find_path_with_resolution(20, 2)
		# self.find_path_with_resolution(10, 1)
		# self.find_path_with_resolution(5, 1)

	def find_path_with_resolution(self, resolution, boundary_distance):
		self.user_settings.cell_width = resolution
		self.path_finder = PathFinder(self.user_settings, previous = self.path_finder)

		start = time.time()
		self.path = self.path_finder.find_path()
		end = time.time()

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


	










