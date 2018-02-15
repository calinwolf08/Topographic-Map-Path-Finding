import cv2

from topographic_map import TopographicMap
from path_finding import PathFinder
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

# TODO: improve contour extraction
	# check perf on different resize values ------------------------------------!!!
	# fine tune masks
	# adjust approach to closing gaps, find better endpoints

# TODO: improve path finding
	# generate successors from nearest density point----------------------------!!!
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

def run(name):
	user_settings = UserSettings.initialized_from_filename(name)

	cv2.imshow("contours", user_settings.cropped_img.contours)
	cv2.imshow("water", user_settings.cropped_img.image_masks.blue_mask)

	path_finder = PathFinder(user_settings)

	image = cv2.cvtColor(path_finder.user_settings.cropped_img.contours, cv2.COLOR_GRAY2BGR)
	grid_img = path_finder.grid.add_grid_to_image(image, 2)
	density_img = path_finder.grid.add_density_to_image(grid_img)

	cv2.imshow("grid", grid_img)
	cv2.imshow("density", density_img)

	path = path_finder.find_path()

	if path is None:
		print("no path found")
	else:
		path_img = path_finder.draw_path(path)
		path_img = cv2.resize(path_img, None, fx=1/2, fy=1/2, 
			interpolation = cv2.INTER_LINEAR)
		cv2.imshow('path', path_img)

	print("done")
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	# run("MountStickney.jpg")
	run("SanLuisObispo.jpg")
	# run("Snoqualmie.jpg")


	










