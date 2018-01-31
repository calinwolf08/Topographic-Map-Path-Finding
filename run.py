import numpy as np
import cv2
import math
import sys, os
import random
from PIL import Image
import pytesseract

from topographic_map import TopographicMap
from path_finding import PathFinder
from cropped_image import CroppedImage

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

		cv2.line(path_img,(fromX,fromY),(toX,toY),(0,255,255),2)

		node = parent
		fromX = toX
		fromY = toY

	return path_img

#TODO: create option to run algorithm in cropped parts or entire image at once
#TODO: compare run times for different sized cropped images

def run(name):
	topo_map = TopographicMap(name)
	path_finder = PathFinder(topo_map)

	path_finder.get_start_end_image()
	
	cropped_img = path_finder.cropped_img.cv_image

	path_finder.find_path()

	# final, final_color = extract_contours(cropped_img)

	# # final = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
	# # cv2.circle(final, (startX,startY), 5, (0,255,0), 2)
	# # cv2.circle(final, (endX,endY), 5, (0,0,255), 2)

	# cv2.imshow('final_color', final_color)
	# cv2.imshow('final', final)

	# grid_dim = 100

	# grid_img = add_grid(final_color, grid_dim = grid_dim, lineThickness = 1)
	# cv2.imshow("grid_img", grid_img)

	# grid = create_grid(final, grid_dim = grid_dim)

	# final_bgr = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
	# density_img = show_density(grid, final_bgr, grid_dim = grid_dim)

	# cv2.imshow("density_img", density_img)
	
	# rows, cols, chan = cropped_img.shape
	# (sx, sy) = convert_coord_to_grid(path_finder.startX, path_finder.startY, int(rows / grid_dim))
	# (ex, ey) = convert_coord_to_grid(path_finder.endX, path_finder.endY, int(rows / grid_dim))

	# node = find_path(grid, grid_dim, sx, sy, ex, ey)
	
	# path_img = draw_path(grid_img, node, int(rows / grid_dim))
	# path_img2 = draw_path(cropped_img, node, int(rows / grid_dim))
	# cv2.imshow("path_img", path_img)
	# cv2.imshow("path_img2", path_img2)

	print("done")
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	# run("MountStickney.jpg")
	run("SanLuisObispo.jpg")
	# run("Snoqualmie.jpg")

# prepare image itself
	# image class ????

# get contour interval from image
	# could be in image class
# get start end points from user

# create grid image
	# optional ?? just visual ??

# create grid
	# grid class 

# create density image
	# optional ?? just visual ??

# convert coords to grid coords

# find path

# draw path

	










