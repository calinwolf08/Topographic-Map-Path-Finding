import numpy as np
import cv2
import math
import sys, os
import random
from PIL import Image
import pytesseract

def isolate_red(cropped_img, hsv):
	# get red areas of hsv
	low_red = np.array([0, 50, 200])
	high_red = np.array([10, 255, 255])
	red_mask = cv2.inRange(hsv, low_red, high_red)
	red_img = cv2.bitwise_and(cropped_img, cropped_img, mask=red_mask)

	# find lines on red areas
	red_edges = cv2.Canny(red_img, 50, 115)
	lines = cv2.HoughLines(red_edges, 1, np.pi/180, 50)

	# connect lines if horizontal
#TODO: 	get avg angle of lines and draw within range of that avg instead of flat
#		to can make the mask smaller
	red_lines = cv2.bitwise_xor(cropped_img, cropped_img)
	
	#row col channels
	w, h, d = red_lines.shape

	if lines is not None:
		for line in lines:
			for rho,theta in line:
			    a = np.cos(theta)
			    b = np.sin(theta)
			    x0 = a*rho
			    y0 = b*rho
			    x1 = int(x0 + w*(-b))
			    y1 = int(y0 + w*(a))
			    x2 = int(x0 - w*(-b))
			    y2 = int(y0 - w*(a))

			    rise = y2 - y1
			    run = x2 - x1
			    theta = np.arctan(rise / run) if run != 0 else 90

			    if abs(theta) < np.pi / 1000:
			    	cv2.line(red_lines,(x1,y1),(x2,y2),(255,255,255),15)

	# create mask for lines
	red_lines_mask = cv2.bitwise_not(red_lines)
	low_black = np.array([0, 0, 0])
	high_black = np.array([1, 1, 1])
	red_lines_mask = cv2.inRange(red_lines_mask, low_black, high_black)

	red_lines_mask = cv2.bitwise_not(red_lines_mask)

	return red_lines_mask

def isolate_blue(cropped_img, hsv):
	# get red areas of hsv
	low_blue = np.array([50, 35, 100])
	high_blue = np.array([100, 150, 255])
	blue_mask = cv2.inRange(hsv, low_blue, high_blue)
	
	blue_contours = cv2.bitwise_xor(cropped_img, cropped_img)
	img, contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(blue_contours, contours, -1, (255,255,255), cv2.FILLED)

	non_blue_mask = cv2.bitwise_not(blue_contours)

	low_black = np.array([0, 0, 0])
	high_black = np.array([1, 1, 1])
	non_blue_mask = cv2.inRange(non_blue_mask, low_black, high_black)

	return cv2.bitwise_not(non_blue_mask)

def reduce_image_contours(mask, minArea):
	img2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	contours = list(filter(lambda c: cv2.contourArea(c) > minArea, contours))
	
	reduced = cv2.bitwise_xor(mask, mask)
	cv2.drawContours(reduced, contours, -1, (255,255,255), cv2.FILLED)
	
	return reduced

def isolate_black(cropped_img, hsv):
	# get red areas of hsv
	low_black = np.array([0, 0, 0])
	high_black = np.array([100, 100, 150])
	black_mask = cv2.inRange(hsv, low_black, high_black)

	black_contours = cv2.bitwise_xor(cropped_img, cropped_img)
	img, contours, hierarchy = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(black_contours, contours, -1, (255,255,255), cv2.FILLED)

	kernel = np.ones((5,5), np.uint8)
	dilated = cv2.dilate(black_contours, kernel, iterations=1)

	non_black_mask = cv2.bitwise_not(dilated)
	non_black_mask = cv2.inRange(non_black_mask, low_black, high_black)

	###########

	black_reduced = reduce_image_contours(non_black_mask, 75)
	black_reduced_color = cv2.bitwise_and(cropped_img, cropped_img, mask=black_reduced)
	
	(connected_mask, connected_color, contours_img) = connect_contours_loop(
		cropped_img, black_reduced, black_reduced_color, dist=30, maxIters=10)

	black_reduced = reduce_image_contours(connected_mask, 1000)

	kernel = np.ones((5,5), np.uint8)
	dilated = cv2.dilate(black_reduced, kernel, iterations=1)
	
	###########

	black_mask = cv2.bitwise_not(dilated)
	img = cv2.bitwise_and(cropped_img, cropped_img, mask = black_mask)

	return black_mask

def distToPt(pt1, pt2):
	return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

def getContourExtremes(contours, allExtremes = False):
	ret = []

	for c in contours:
		l = tuple(c[c[:,:,0].argmin()][0])
		r = tuple(c[c[:,:,0].argmax()][0])
		t = tuple(c[c[:,:,1].argmin()][0])
		b = tuple(c[c[:,:,1].argmax()][0])

		if allExtremes:
			ret.append((t, b))
			ret.append((l, r))
		else: 
			if distToPt(l, r) > distToPt(t, b):
				ret.append((l, r))
			else:
				ret.append((t, b))

	return ret

def connectPoints(p1, p2, points, dist, img, epsilon, slope = 0, testSlope = False, 
	lineThickness = 4):
	points.sort(key=lambda x: distToPt(p1, x))

	if len(points) > 2 and points[1] == p2:
		p = points[2]
	else:
		p = points[1]

	if distToPt(p, p1) < dist:
		if testSlope is False:
			cv2.line(img, p1, p, (0,0,255), lineThickness)

def connectExtremes(extremes, img, dist, epsilon, slope = 0, testSlope = False, 
	lineThickness = 4):
	copy = img.copy()

	points = list(map(lambda x: x[0], extremes))
	points += list(map(lambda x: x[1], extremes))

	w, h, d = img.shape
	z = 15
	points = list(filter(lambda x: x[0] > z and x[0] < w-z and x[1] > z and x[1] < h-z, points))

	for e in extremes:
		# first point in e
		x = e[0][0]
		y = e[0][1]

		if x > z and x < w-z and y > z and y < h-z:
			connectPoints(e[0], e[1], points, dist, copy, epsilon, slope, testSlope, lineThickness)
		
		# second point in e
		x = e[1][0]
		y = e[1][1]

		if x > z and x < w-z and y > z and y < h-z:
			connectPoints(e[1], e[0], points, dist, copy, epsilon, slope, testSlope, lineThickness)

	low_black = np.array([0, 0, 0]) #0,0,0
	high_black = np.array([1, 1, 1]) #255,255,85
	mask = cv2.inRange(copy, low_black, high_black)

	bw = cv2.bitwise_not(mask)

	return (copy, bw)

def getContours(img, mask):
	img2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	copy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	copy = cv2.bitwise_and(copy, copy)
	copy = cv2.cvtColor(copy, cv2.COLOR_GRAY2BGR)

	for i in range(len(contours)):
		r = random.randint(0, 255)
		g = random.randint(0, 255)
		b = random.randint(0, 255)
		cv2.drawContours(copy, contours, i, (r,g,b), cv2.FILLED)
	
	return (copy, contours)

def connect_contours(img, mask, dist, lineThickness = 4, allExtremes = False):
	(contours_img, contours) = getContours(img, mask)
	numContours = len(contours)
	
	extremes = getContourExtremes(contours, allExtremes)

	w, h, d = img.shape
	z = 10

	for e in extremes:
		cv2.circle(contours_img, e[0], 1, (255,0,0), 2)
		cv2.circle(contours_img, e[1], 1, (255,0,0), 2)
	
	color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

	(connected_img, connected_mask) = connectExtremes(
		extremes, color_mask, dist=dist, epsilon=0.5, testSlope = False, lineThickness = lineThickness)

	return (connected_mask, contours_img, numContours)

def connect_contours_loop(img, connected_mask, connected_color, dist, maxIters, 
	lineThickness = 4, allExtremes = False):
	prevNumContours = 0
	curNumContours = -1
	numIters = 0

	print("connecting contours, dist = " + str(dist))

	while prevNumContours != curNumContours and numIters < maxIters:
		(connected_mask, contours_img, numContours) = connect_contours(connected_color, connected_mask, dist, 
			lineThickness, allExtremes)
		print(str(numIters) + ": " + str(numContours))
		print("-------")

		connected_color = cv2.bitwise_and(img, img, mask=connected_mask)

		prevNumContours = curNumContours
		curNumContours = numContours
		numIters += 1

	return (connected_mask, connected_color, contours_img)

def setup_img(name):
	img = cv2.imread(name, 1)

	dist = get_countour_interval_dist(img)

	x1 = 300
	y1 = 600
	z = 200
	img = img[x1:x1+z, y1:y1+z]

	cropped_img = cv2.resize(img, None, fx=6, fy=6, interpolation = cv2.INTER_LINEAR)

	return (dist, img, cropped_img)

def get_masks(cropped_img):
	hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

	blue_mask = isolate_blue(cropped_img, hsv)
	black_mask = isolate_black(cropped_img, hsv)
	red_lines_mask = isolate_red(cropped_img, hsv)
	combined_mask = cv2.bitwise_and(blue_mask, black_mask)
	combined_mask = cv2.bitwise_and(combined_mask, red_lines_mask)
	
	return combined_mask

def prepare_for_contour_extraction(cropped_img, mask):
	kernel = np.ones((5,5), np.uint8)
	dilated = cv2.dilate(cropped_img, kernel, iterations=1)

	base_img = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
	base_img = cv2.fastNlMeansDenoising(base_img, None, 5, 7, 21)

	base_img = cv2.threshold(base_img,225,255,cv2.THRESH_BINARY_INV)[1]
	img_mask = cv2.bitwise_and(base_img, base_img, mask=mask)

	img_color = cv2.bitwise_and(cropped_img, cropped_img, mask=img_mask)

	return (base_img, img_mask, img_color)

def detect_numbers(img):
	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, img)

	results = pytesseract.image_to_string(Image.open(filename))
	os.remove(filename)
	print(results)

	return results

def run_connect_contours(cropped_img, connected_mask, connected_color, iters, 
	lineThickness = 4, minArea = 50, allExtremes = False):

	for (dist, maxIters) in iters:
		(connected_mask, connected_color, contours_img) = connect_contours_loop(
			cropped_img, connected_mask, connected_color, 
			dist=dist, maxIters=maxIters, lineThickness=lineThickness, allExtremes=allExtremes)

		connected_mask = reduce_image_contours(connected_mask, minArea)
		connected_color = cv2.bitwise_and(cropped_img, cropped_img, mask=connected_mask)

	return (connected_mask, connected_color, contours_img)

def skeletonize_mask(img):
	size = np.size(img)
	skel = np.zeros(img.shape,np.uint8)
	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	done = False
	 
	while( not done):
	    eroded = cv2.erode(img,element)
	    temp = cv2.dilate(eroded,element)
	    temp = cv2.subtract(img,temp)
	    skel = cv2.bitwise_or(skel,temp)
	    img = eroded.copy()
	 
	    zeros = size - cv2.countNonZero(img)
	    if zeros==size:
	        done = True

	return skel 

def extract_contours(cropped_img):
	# get masks to remove from image
	# print("get mask...")
	mask = get_masks(cropped_img)
	# cv2.imshow("mask", mask)

	# prepare img for extracting contours by dilating, denoising, and finding edges 
	# then apply mask and dilate edges
	# print("prepare...")
	(base_img, contours_mask, contours_color) = prepare_for_contour_extraction(cropped_img, mask)
	# cv2.imshow("base_img", base_img)
	# cv2.imshow("contours_mask", contours_mask)

	# detect any numbers
	detect_numbers(contours_mask)

	# detect contours and connect extreme points within certain distances
	# print("extract contours...")
	iters = [(10,10), (20,10), (30,10)]
	(connected_mask, connected_color, contours_img) = run_connect_contours(cropped_img, contours_mask, contours_color, iters)
	
	# cv2.imshow("contours_img", contours_img)
	# cv2.imshow("connected_mask", connected_mask)

	skel = skeletonize_mask(connected_mask)
	# cv2.imshow("skel", skel)
	skel = reduce_image_contours(skel, 1)
	# cv2.imshow("skel", skel)

	kernel = np.ones((2,2), np.uint8)
	dilated = cv2.dilate(skel, kernel, iterations=1)
	# cv2.imshow("dilated", dilated)

	skel_color = cv2.bitwise_and(cropped_img, cropped_img, mask=dilated)

	iters += [(40,10), (50,10), (60,10), (70,10)]
	(final, connected_color, final_color) = run_connect_contours(cropped_img, dilated, skel_color, iters, 
		lineThickness = 2, minArea = 2, allExtremes = False)

	# cv2.imshow("final_color", final_color)
	# cv2.imshow("final", final)
	return final, final_color

startX = -1 
startY = -1 
endX = -1 
endY = -1
image = None

def click_image(event, x, y, flags, param):
	# grab references to the global variables
	global startX, startY, endX, endY, image
	
	# if event != 0: print(event)

	if event == 1:
		if startX < 0:
			startX = x
			startY = y
			cv2.circle(image, (x,y), 5, (0,255,0), 2)
		elif endX < 0:
			endX = x
			endY = y
			cv2.circle(image, (x,y), 5, (0,0,255), 2)

def get_start_end_points(img):
	global image

	image = img.copy()

	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_image)
	cv2.imshow("image", image)

	while(startX < 0 or endX < 0):
		k = cv2.waitKey(20) & 0xFF
		cv2.imshow("image", image)

		if k == ord('q') or k == ord(' '):
			break

def convert_coord_to_grid(x1, y1, dim):
	x = int(x1 / dim)
	y = int(y1 / dim)

	return (x,y)

def convert_grid_to_coord(x1, y1, dim):
	x = (x1 * dim) + int(dim/2)
	y = y1 * dim + int(dim/2)

	return (x,y)

def add_grid(img, grid_dim, lineThickness):
	global startX, startY, endX, endY

	copy = img.copy()
	row, col, chan = copy.shape

	dx = int(col / grid_dim)
	dy = int(row / grid_dim)

	i = 0
	while i < row:
		cv2.line(copy,(0,i),(col,i),(255,0,0),lineThickness)
		i += dy

	j = 0
	while j < col:
		cv2.line(copy,(j,0),(j,row),(255,0,0),lineThickness)
		j += dx

	x = (int(startX / dx) * dx) + int(dx / 2)
	y = (int(startY / dy) * dy) + int(dy / 2)
	cv2.circle(copy, (x,y), int(dx / 2), (0,255,0), 2)

	x = (int(endX / dx) * dx) + int(dx / 2)
	y = (int(endY / dy) * dy) + int(dy / 2)
	cv2.circle(copy, (x,y), int(dx / 2), (0,0,255), 2)

	return copy

class Cell:
	def __init__(self, density, water_density = 0):
		self.density = density
		self.water_density = water_density

def get_img_densities(col, row, dx, dy, grid_dim, img):
	minY = row * dy
	maxY = minY + dy
	minX = col * dx
	maxX = minX + dx

	cur_img = img[minY:maxY, minX:maxX]

	non_zero = cv2.countNonZero(cur_img)

	return (non_zero / (dx * dy), 0)

def create_grid(img, grid_dim):
	rows, cols = img.shape

	dx = int(cols / grid_dim)
	dy = int(rows / grid_dim)

	grid = [[Cell(0,0) for x in range(grid_dim)] for y in range(grid_dim)]

	for c in range(grid_dim):
		for r in range(grid_dim):
			d, wd = get_img_densities(c, r, dx, dy, grid_dim, img)
			
			cell = grid[c][r]
			cell.density = d
			cell.water_density = wd

	return grid

class Node:
	def __init__(self, x, y, f=0.0, g=0.0, h=0.0, parent=None):
		self.x = x
		self.y = y
		self.f = f
		self.g = g
		self.h = h
		self.parent = parent

def generate_successors(cur, grid_dim):
	x0 = cur.x
	y0 = cur.y

	successors = []

	# successors to the left
	x = x0 - 1
	
	if x >= 0:
		lm = Node(x, y0, parent=cur)
		successors.append(lm)

		y = y0 - 1

		if y >= 0:
			lt = Node(x, y, parent=cur)
			successors.append(lt)

		y = y0 + 1

		if y < grid_dim:
			lb = Node(x, y, parent=cur)
			successors.append(lb)

	# successors to the right
	x = x0 + 1
	
	if x < grid_dim:
		rm = Node(x, y0, parent=cur)
		successors.append(rm)

		y = y0 - 1

		if y >= 0:
			rt = Node(x, y, parent=cur)
			successors.append(rt)

		y = y0 + 1

		if y < grid_dim:
			rb = Node(x, y, parent=cur)
			successors.append(rb)

	# top middle
	y = y0 + 1

	if y < grid_dim:
		mb = Node(x0, y, parent=cur)
		successors.append(mb)

	# bottom middle
	y = y0 - 1

	if y > 0:
		mt = Node(x0, y, parent=cur)
		successors.append(mt)

	return successors

def get_dist_between_nodes(start, end, grid, grid_dim, slope_multiplier):
	dx = end.x - start.x
	dy = end.y - start.y
	
	x = end.x
	y = end.y
	
	cur = grid[x][y]
	# print("--------------")
	# print(start.x)
	# print(start.y)

	# print(x)
	# print(y)

	# print(dx)
	# print(dy)

	#find coordinates of nearest density in direction start->end
	while cur.density == 0: 
		
		if x + dx >= 0 and x + dx < grid_dim and y + dy >= 0 and y + dy < grid_dim:
			x += dx
			y += dy
			cur = grid[x][y]
		else:
			break

	# distance to nearest density location
	dist_to_density = math.sqrt(math.pow(start.x-x, 2) + math.pow(start.y-y, 2))

	# estimated slope between start and end points
	slope = slope_multiplier / dist_to_density

	# flat distance between start and end points
	dist_to_point = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))

	# final full distance from start to end
	final_dist = math.sqrt(math.pow(dist_to_point, 2) + math.pow(slope, 2))

	return final_dist

def find_path(grid, grid_dim, sx, sy, ex, ey):
	global startX, startY
	open_list = [Node(sx, sy)]
	closed_list = []
	final = None

	print("start: " + str(sx) + ", " + str(sy))
	print("end: " + str(ex) + ", " + str(ey))
	count = 0
	slope_multiplier = 5

	while len(open_list) > 0:
		open_list.sort(key = lambda x: x.f, reverse=True)

		cur = open_list.pop()

		successors = generate_successors(cur, grid_dim)

		for s in successors:
			# if at enc point
			if s.x == ex and s.y == ey:
				return s

			# calculate heuristic
			s.g = cur.g + get_dist_between_nodes(cur, s, grid, grid_dim, slope_multiplier)
			s.h = math.sqrt(math.pow(ex-s.x, 2) + math.pow(ey-s.y, 2))
			s.f = s.g + s.h

			# if position already in open list with lower f then skip this one
			temp = list(filter(lambda x: x.x == s.x and x.y == s.y and x.f <= s.f, open_list))

			if len(temp) > 0:
				continue

			# if position already in open list with lower f then skip this one
			temp = list(filter(lambda x: x.x == s.x and x.y == s.y and x.f <= s.f, closed_list))

			if len(temp) > 0:
				continue

			open_list.append(s)

		closed_list.append(cur)
		count += 1

		# if count % 50 == 0:
		# 	print("visited: " + str(cur.x) + ", " + str(cur.y) + ": f=" + str(cur.f) + ", g=" + str(cur.g) + ", h=" + str(cur.h))

	print("no path found :(")
	sys.exit(0)

def show_density(grid, img, grid_dim):
	rows, cols, chan = img.shape
	dx = int(cols / grid_dim)
	dy = int(rows / grid_dim)

	for c in range(grid_dim):
		for r in range(grid_dim):
			x = (c * dx) + int(dx / 2)
			y = (r * dy) + int(dy / 2)

			cell = grid[c][r]

			if cell.density > 0:
				cv2.circle(img, (x,y), int(dx/2), (255,0,0), 2)

	return img

def draw_path(img, node, dim):
	path_img = img.copy()
	fromX, fromY = convert_grid_to_coord(node.x, node.y, dim)

	while node.parent is not None:
		parent = node.parent
		toX, toY = convert_grid_to_coord(parent.x, parent.y, dim)

		cv2.line(path_img,(fromX,fromY),(toX,toY),(0,0,255),2)

		node = parent
		fromX = toX
		fromY = toY

	return path_img

def get_countour_interval_dist(img):
	rows, cols, chan = img.shape

	z = 600

	img = img[rows-z:rows, int(0.25*cols):int(0.75*cols)]

	# cv2.imshow("blah", img)

	results = detect_numbers(img)
	print(results)
	results_list = results.split()

	candidates = []

	contour_idx = [i for i, x in enumerate(results_list) if x.upper() == "CONTOUR"]

	for i in contour_idx:
		if results_list[i+2].isnumeric():
			cand = (i, results_list[i+2])
			candidates.append(cand)

	interval_idx = [i for i, x in enumerate(results_list) if x.upper() == "INTERVAL"]

	for i in interval_idx:
		if results_list[i+1].isnumeric():
			cand = (i, results_list[i+1])
			candidates.append(cand)
	
	feet_idx = [i for i, x in enumerate(results_list) if x.upper() == "FEET"]

	for i in feet_idx:
		if results_list[i-1].isnumeric():
			cand = (i, results_list[i-1])
			candidates.append(cand)

	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	return candidates[0][1]

def run(name):
	# setup image to parse
	global startX, startY, endX, endY

	(dist, img, cropped_img) = setup_img(name)
	
	get_start_end_points(cropped_img)

	final, final_color = extract_contours(cropped_img)

	# final = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
	# cv2.circle(final, (startX,startY), 5, (0,255,0), 2)
	# cv2.circle(final, (endX,endY), 5, (0,0,255), 2)

	cv2.imshow('final_color', final_color)
	cv2.imshow('final', final)

	grid_dim = 100

	grid_img = add_grid(final_color, grid_dim = grid_dim, lineThickness = 1)
	cv2.imshow("grid_img", grid_img)

	grid = create_grid(final, grid_dim = grid_dim)

	final_bgr = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
	density_img = show_density(grid, final_bgr, grid_dim = grid_dim)

	cv2.imshow("density_img", density_img)
	
	rows, cols, chan = cropped_img.shape
	(sx, sy) = convert_coord_to_grid(startX, startY, int(rows / grid_dim))
	(ex, ey) = convert_coord_to_grid(endX, endY, int(rows / grid_dim))

	node = find_path(grid, grid_dim, sx, sy, ex, ey)
	
	path_img = draw_path(grid_img, node, int(rows / grid_dim))
	path_img2 = draw_path(cropped_img, node, int(rows / grid_dim))
	cv2.imshow("path_img", path_img)
	cv2.imshow("path_img2", path_img2)

	print("done")
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	# run("MountStickney.jpg")
	run("SanLuisObispo.jpg")
	# run("Snoqualmie.jpg")

	










