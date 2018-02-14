import cv2

from topographic_map import TopographicMap
from path_finding import PathFinder
from cropped_image import CroppedImage

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
	# fine tune masks
	# adjust approach to closing gaps, find better endpoints

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

def run(name):
	topo_map = TopographicMap(name)
	# print(topo_map.image_data.feet_per_pixel)
	# print(topo_map.image_data.contour_interval_dist)
	# cv2.imshow("sub_image", topo_map.image_data.sub_image)
	path_finder = PathFinder(topo_map, 50)
	
	cropped_img = path_finder.cropped_img

	# # cropped_img.image_masks.show_masks()
	# cv2.imshow('cropped_img', cropped_img.cv_image)
	cv2.imshow('contours', cropped_img.contours)

	path_node = path_finder.calculate_path()
	path = path_finder.draw_path(cropped_img.cv_image, path_node)
	# temp = cv2.cvtColor(cropped_img.contours, cv2.COLOR_GRAY2BGR)
	# path2 = path_finder.draw_path(temp, path_node)

	if path is None:
		print("no path found")
	else:
		cv2.imshow('path', path)
	# cv2.imshow('path2', path2)

	print("done")
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	run("MountStickney.jpg")
	# run("SanLuisObispo.jpg")
	# run("Snoqualmie.jpg")


	










