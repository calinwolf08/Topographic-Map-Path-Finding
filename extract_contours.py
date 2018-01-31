import cv2
import numpy as np
import random, math

from helper_functions import Helper

class ContourConnector:

	def __init__(self, contours_mask):
		self.contours_mask = contours_mask
		self.max_iterations = 10

		self.connected_contours_mask = contours_mask.copy()
		self.num_contours = -1

	def connect_contours_within_distance(self, distance):
		prev_num_contours = 0
		cur_num_contours = -1
		num_iterations = 0

		print("connecting contours, dist = " + str(distance))

		while prev_num_contours != cur_num_contours and num_iterations < self.max_iterations:
			self.num_contours = self.__connect_contours(distance)
			
			print(str(num_iterations) + ": " + str(self.num_contours))
			print("-------")

			prev_num_contours = cur_num_contours
			cur_num_contours = self.num_contours
			num_iterations += 1

		return self.connected_contours_mask

	def __connect_contours(self, distance):
		contours = self.__get_contours(self.connected_contours_mask)
		
		extremes = self.__get_contour_extremes(contours)
		self.__connect_extremes(extremes, distance)

		return len(contours)

	def __get_contours(self, mask):
		img2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# copy = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
		# copy = cv2.bitwise_and(copy, copy)
		# copy = cv2.cvtColor(copy, cv2.COLOR_GRAY2BGR)

		# for i in range(len(contours)):
		# 	r = random.randint(0, 255)
		# 	g = random.randint(0, 255)
		# 	b = random.randint(0, 255)
		# 	cv2.drawContours(copy, contours, i, (r,g,b), cv2.FILLED)
		
		# return (copy, contours)

		return contours

	def __get_contour_extremes(self, contours):
		ret = []

		for c in contours:
			l = tuple(c[c[:,:,0].argmin()][0])
			r = tuple(c[c[:,:,0].argmax()][0])
			t = tuple(c[c[:,:,1].argmin()][0])
			b = tuple(c[c[:,:,1].argmax()][0])

			if self.__dist_to_pt(l, r) > self.__dist_to_pt(t, b):
				ret.append((l, r))
			else:
				ret.append((t, b))

		return ret

	def __connect_extremes(self, extremes, distance):
		points = list(map(lambda x: x[0], extremes))
		points += list(map(lambda x: x[1], extremes))

		w, h = self.connected_contours_mask.shape
		border_thresh = 15
		points = list(filter(lambda x: x[0] > border_thresh and x[0] < w-border_thresh and 
			x[1] > border_thresh and x[1] < h-border_thresh, points))

		# self.connected_contours_mask = cv2.cvtColor(self.connected_contours_mask, cv2.COLOR_GRAY2BGR)

		for e in extremes:
			# first point in e
			x = e[0][0]
			y = e[0][1]

			if x > border_thresh and x < w-border_thresh and y > border_thresh and y < h-border_thresh:
				self.__connect_points(e[0], e[1], points, distance)
			
			# second point in e
			x = e[1][0]
			y = e[1][1]

			if x > border_thresh and x < w-border_thresh and y > border_thresh and y < h-border_thresh:
				self.__connect_points(e[1], e[0], points, distance)

		# self.connected_contours_mask = Helper.convert_image_to_mask(self.connected_contours_mask)

	def __connect_points(self, p1, p2, points, distance):
		lineThickness = 4
		epsilon=0.5
		points.sort(key=lambda x: self.__dist_to_pt(p1, x))

		if len(points) > 2 and points[1] == p2:
			p = points[2]
		else:
			p = points[1]

		if self.__dist_to_pt(p, p1) < distance:
			# cv2.line(self.connected_contours_mask, p1, p, (0,0,255), lineThickness)
			cv2.line(self.connected_contours_mask, p1, p, (255,255,255), lineThickness)

	def __dist_to_pt(self, pt1, pt2):
		return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

class ContourExtractor:
	def __init__(self, cropped_image):
		self.cv_image = cropped_image.cv_image
		self.cv_image_masks = cropped_image.cv_image_masks

		self.extracted_contours = None

	def extract_contours(self):
		distances = [10, 20, 30, 40, 50, 60, 70]

		min_contour_area = 50
		first_prepared_mask = self.__prepare_for_first_contour_connecting()
		self.__connect_contours_by_distances(first_prepared_mask, distances[:3], min_contour_area)
		
		min_contour_area = 2
		second_prepared_mask = self.__prepare_for_second_contour_connecting()
		self.__connect_contours_by_distances(second_prepared_mask, distances[3:], min_contour_area)

	def __prepare_for_first_contour_connecting(self):
		dilated_image = Helper.dilate_image(self.cv_image)
		dilated_mask = Helper.convert_image_to_mask(dilated_image)
		gray_denoised_image = cv2.fastNlMeansDenoising(dilated_mask, None, 5, 7, 21)
		threshold_image = cv2.threshold(gray_denoised_image,225,255,cv2.THRESH_BINARY_INV)[1]
		prepared_mask = cv2.bitwise_and(threshold_image, threshold_image, mask=self.cv_image_masks.topo_mask)

		# cv2.imshow("prep", prepared_mask)

		# prepared_image = cv2.bitwise_and(self.cv_image, self.cv_image, mask=prepared_mask)
		# cv2.imshow("prep2", prepared_image)

		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		return prepared_mask

	def __prepare_for_second_contour_connecting(self):
		skeleton_mask = self.__skeletonize_mask(self.extracted_contours)
		reduced_mask = self.reduce_image_contours(skeleton_mask, 1)
		dilated_mask = Helper.dilate_image(reduced_mask)
		# dilated_color = cv2.bitwise_and(self.cv_image, self.cv_image, mask=dilated_mask)

		return dilated_mask

	def __connect_contours_by_distances(self, mask, distances, min_contour_area):
		contour_connector = ContourConnector(mask)

		for distance in distances:
			contour_connector.connect_contours_within_distance(distance)
			self.reduce_image_contours(contour_connector.connected_contours_mask, min_contour_area)

		self.extracted_contours = contour_connector.connected_contours_mask

	def __skeletonize_mask(self, img):
		size = np.size(img)
		skel = np.zeros(img.shape,np.uint8)
		element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
		done = False
		 
		while (not done):
		    eroded = cv2.erode(img,element)
		    temp = cv2.dilate(eroded,element)
		    temp = cv2.subtract(img,temp)
		    skel = cv2.bitwise_or(skel,temp)
		    img = eroded.copy()
		 
		    zeros = size - cv2.countNonZero(img)
		    if zeros==size:
		        done = True

		return skel

	@staticmethod
	def reduce_image_contours(mask, minArea):
		img2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		contours = list(filter(lambda c: cv2.contourArea(c) > minArea, contours))
		
		reduced = cv2.bitwise_xor(mask, mask)
		cv2.drawContours(reduced, contours, -1, (255,255,255), cv2.FILLED)
		
		return reduced

###################
	# def connect_contours_loop(self, connected_mask, connected_color, dist, maxIters, 
	# 	lineThickness = 4, allExtremes = False):
	# 	prevNumContours = 0
	# 	curNumContours = -1
	# 	numIters = 0

	# 	print("connecting contours, dist = " + str(dist))

	# 	while prevNumContours != curNumContours and numIters < maxIters:
	# 		(connected_mask, contours_img, numContours) = self.connect_contours(
	# 			connected_color, connected_mask, dist, lineThickness, allExtremes)
	# 		print(str(numIters) + ": " + str(numContours))
	# 		print("-------")

	# 		connected_color = cv2.bitwise_and(self.cv_image, self.cv_image, mask=connected_mask)

	# 		prevNumContours = curNumContours
	# 		curNumContours = numContours
	# 		numIters += 1

	# 	return (connected_mask, connected_color, contours_img)

	# def connect_contours(self, mask, dist, lineThickness = 4, allExtremes = False):
	# 	(contours_img, contours) = self.get_contours(mask)
	# 	numContours = len(contours)
		
	# 	extremes = self.get_contour_extremes(contours, allExtremes)

	# 	w, h, d = self.cv_image.shape
	# 	z = 10

	# 	for e in extremes:
	# 		cv2.circle(contours_img, e[0], 1, (255,0,0), 2)
	# 		cv2.circle(contours_img, e[1], 1, (255,0,0), 2)
		
	# 	color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

	# 	(connected_img, connected_mask) = self.connect_extremes(
	# 		extremes, color_mask, dist=dist, epsilon=0.5, testSlope = False, lineThickness = lineThickness)

	# 	return (connected_mask, contours_img, numContours)

	# def get_contours(self, mask):
	# 	img2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# 	copy = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
	# 	copy = cv2.bitwise_and(copy, copy)
	# 	copy = cv2.cvtColor(copy, cv2.COLOR_GRAY2BGR)

	# 	for i in range(len(contours)):
	# 		r = random.randint(0, 255)
	# 		g = random.randint(0, 255)
	# 		b = random.randint(0, 255)
	# 		cv2.drawContours(copy, contours, i, (r,g,b), cv2.FILLED)
		
	# 	return (copy, contours)

	# def get_contour_extremes(self, contours, allExtremes = False):
	# 	ret = []

	# 	for c in contours:
	# 		l = tuple(c[c[:,:,0].argmin()][0])
	# 		r = tuple(c[c[:,:,0].argmax()][0])
	# 		t = tuple(c[c[:,:,1].argmin()][0])
	# 		b = tuple(c[c[:,:,1].argmax()][0])

	# 		if allExtremes:
	# 			ret.append((t, b))
	# 			ret.append((l, r))
	# 		else: 
	# 			if ContourExtractor.dist_to_pt(l, r) > ContourExtractor.dist_to_pt(t, b):
	# 				ret.append((l, r))
	# 			else:
	# 				ret.append((t, b))

	# 	return ret

	# def connect_extremes(self, extremes, img, dist, epsilon, slope = 0, testSlope = False, 
	# 	lineThickness = 4):
	# 	copy = img.copy()

	# 	points = list(map(lambda x: x[0], extremes))
	# 	points += list(map(lambda x: x[1], extremes))

	# 	w, h, d = img.shape
	# 	z = 15
	# 	points = list(filter(lambda x: x[0] > z and x[0] < w-z and x[1] > z and x[1] < h-z, points))

	# 	for e in extremes:
	# 		# first point in e
	# 		x = e[0][0]
	# 		y = e[0][1]

	# 		if x > z and x < w-z and y > z and y < h-z:
	# 			self.connect_points(e[0], e[1], points, dist, copy, epsilon, slope, testSlope, lineThickness)
			
	# 		# second point in e
	# 		x = e[1][0]
	# 		y = e[1][1]

	# 		if x > z and x < w-z and y > z and y < h-z:
	# 			self.connect_points(e[1], e[0], points, dist, copy, epsilon, slope, testSlope, lineThickness)

	# 	low_black = np.array([0, 0, 0]) #0,0,0
	# 	high_black = np.array([1, 1, 1]) #255,255,85
	# 	mask = cv2.inRange(copy, low_black, high_black)

	# 	bw = cv2.bitwise_not(mask)

	# 	return (copy, bw)

	# def connect_points(self, p1, p2, points, dist, img, epsilon, slope = 0, testSlope = False, 
	# 	lineThickness = 4):
	# 	points.sort(key=lambda x: ContourExtractor.dist_to_pt(p1, x))

	# 	if len(points) > 2 and points[1] == p2:
	# 		p = points[2]
	# 	else:
	# 		p = points[1]

	# 	if self.dist_to_pt(p, p1) < dist:
	# 		if testSlope is False:
	# 			cv2.line(img, p1, p, (0,0,255), lineThickness)

	# def dist_to_pt(self, pt1, pt2):
	# 	return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
 


