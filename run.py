import cv2
import time
import sys

from topographic_map import TopographicMap
from path_finding import PathFinder, Grid
from path_finding import UserSettings
from cropped_image import CroppedImage

# TODO: user settings
	# add interface to get user settings

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

def run(name):
	user_settings = UserSettings.initialized_from_filename(name)

	cv2.imshow("image", user_settings.cropped_img.cv_image)
	user_settings.cropped_img.image_masks.show_masks()

	start = time.time()
	contours = user_settings.cropped_img.contours
	end = time.time()

	print("contour extraction time: " + str(end - start))
	cv2.imshow("contours", contours)

	path_finder = PathFinder(user_settings)

	image = cv2.cvtColor(user_settings.cropped_img.contours, cv2.COLOR_GRAY2BGR)
	# grid_img = path_finder.grid.add_grid_to_image(image, 1)
	density_img = path_finder.grid.add_density_to_image(image)
	cv2.imshow("density", density_img)

	# grid = Grid(user_settings.cropped_img, 10)

	# heat_img = grid.add_heat_map_to_image(image)

	# cv2.imshow("grid", grid_img)
	# cv2.imshow("heat", heat_img)

	start = time.time()
	path = path_finder.find_path()
	end = time.time()

	print("path finding time: " + str(end - start))

	if path is None:
		print("no path found")
	else:
		path_img = path_finder.draw_path(path)
		cv2.imshow('path', path_img)

	print("done")
	cv2.waitKey(0)
	cv2.destroyAllWindows()

class UserInterface:
	def __init__(self):
		self.filename = None

	def start(self):
		self.user_settings = UserSettings()
		self.set_user_settings()
		self.interface_loop()

	def interface_loop(self):
		run = True

		while run:
			print(str(self.user_settings) + "\n")

			command = input("Enter # to change a setting, r to run path finding, s to show images, q to quit\n")
			
			if command == "1":
				self.filename = self.get_filename()
			elif command == "2":
				self.user_settings.find_start_end_points()
			elif command == "3":
				self.user_settings.avoid_water = not self.user_settings.avoid_water
			elif command == "4":
				self.user_settings.avoid_forrest = not self.user_settings.avoid_forrest
			elif command == "5":
				self.user_settings.max_grade = self.get_max_grade()
			elif command == "6":
				self.user_settings.cell_width = self.get_cell_width()
			elif command == "r":
				self.run()
			elif command == "q":
				run = False
			else:
				print("command invalid")

	def run(self):
		path_finder = PathFinder(self.user_settings)

		start = time.time()
		path = path_finder.find_path()
		end = time.time()

		print("path finding time: " + str(end - start))

		if path is None:
			print("no path found")
		else:
			path_img = path_finder.draw_path(path)
			cv2.imshow("contours", self.user_settings.cropped_img.contours)
			# cv2.imshow("path" + str(time.time()), path_img)
			image = cv2.cvtColor(self.user_settings.cropped_img.contours, cv2.COLOR_GRAY2BGR)
			grid_img = path_finder.grid.add_grid_to_image(path_img, 1)
			cv2.imshow("grid" + str(time.time()), grid_img)

	def set_user_settings(self):
		self.filename = self.get_filename()
		self.user_settings.set_topo_map(self.filename)
		self.user_settings.find_cropped_image()

	def get_filename(self):
		if self.filename is None and len(sys.argv) > 1:
			filename = sys.argv[1]
		else:
			filename = input("Enter the topographic map filename: ")

		return filename
	
	def get_max_grade(self):
		has_new_grade = False
		grade = None

		while has_new_grade == False:

			grade = input("Enter your maximum grade: ")

			if grade.isdigit() and int(grade) <= 100:
				has_new_grade = True
			else:
				print("Please enter a grade between 0 and 100")

		return int(grade)

	def get_cell_width(self):
		has_new_width = False
		width = None

		while has_new_width == False:

			width = input("Enter your grid cell width: ")

			if width.isdigit() and int(width) <= 100:
				has_new_width = True
			else:
				print("Please enter a width greater than 0")

		return int(width)

if __name__ == '__main__':
	user_interface = UserInterface()
	user_interface.start()

	# run("MountStickney.jpg")
	# run("SanLuisObispo.jpg")
	# run("Snoqualmie.jpg")


	










