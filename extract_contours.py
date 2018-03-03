import cv2
import numpy as np
import random, math, time

from helper_functions import Helper

class ContourConnector:
	max_iterations = 10

	def __init__(self, contours_mask):
		self.contours_mask = contours_mask
		self.connected_contours_mask = contours_mask.copy()
		self.num_contours = -1

	def connect_contours_within_distance(self, distance):
		prev_num_contours = 0
		cur_num_contours = -1
		num_iterations = 0

		print("connecting contours, dist = " + str(distance))

		while prev_num_contours != cur_num_contours and num_iterations < ContourConnector.max_iterations:
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
			# first extreme point in e
			x = e[0][0]
			y = e[0][1]

			if x > border_thresh and x < w-border_thresh and y > border_thresh and y < h-border_thresh:
				self.__connect_points(e[0], e[1], points, distance)
			
			# second extreme point in e
			x = e[1][0]
			y = e[1][1]

			if x > border_thresh and x < w-border_thresh and y > border_thresh and y < h-border_thresh:
				self.__connect_points(e[1], e[0], points, distance)

		# self.connected_contours_mask = Helper.convert_image_to_mask(self.connected_contours_mask)

	def __connect_points(self, p1, p2, points, distance):
		points.sort(key=lambda x: self.__dist_to_pt(p1, x))

		if len(points) > 2 and points[1] == p2:
			p = points[2]
		else:
			p = points[1]

		if self.__dist_to_pt(p, p1) < distance:
			# cv2.line(self.connected_contours_mask, p1, p, (0,0,255), lineThickness)
			cv2.line(self.connected_contours_mask, p1, p, (255,255,255), 1)

	def __dist_to_pt(self, pt1, pt2):
		return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

class ContourExtractor:
	def __init__(self, cropped_image):
		self.cv_image = cropped_image.cv_image
		self.image_masks = cropped_image.image_masks

		self.extracted_contours = self.__extract_contours()

	def __extract_contours(self):
		distances = list(map(lambda x: x * Helper.resize_factor, [4, 6, 8, 10, 12, 14]))
		# distances = list(map(lambda x: x * Helper.resize_factor, [4]))

		min_contour_area = 8 * Helper.resize_factor
		first_prepared_mask = self.__prepare_for_first_contour_connecting()

		temp = first_prepared_mask.copy()
		temp = self.__prepare_for_second_contour_connecting(temp)

		return temp
		
		# first_connected_mask = self.__connect_contours_by_distances(first_prepared_mask, distances[:3], min_contour_area)		
		# min_contour_area = 2
		# second_prepared_mask = self.__prepare_for_second_contour_connecting(first_connected_mask)
		# second_connected_mask = self.__connect_contours_by_distances(second_prepared_mask, distances[3:], min_contour_area)

		# return second_connected_mask

	def __prepare_for_first_contour_connecting(self):
		dilated_image = Helper.dilate_image(self.cv_image)
		dilated_mask = Helper.convert_image_to_mask(dilated_image)
		gray_denoised_image = cv2.fastNlMeansDenoising(dilated_mask, None, 5, 7, 21)
		threshold_image = cv2.threshold(gray_denoised_image,225,255,cv2.THRESH_BINARY_INV)[1]
		prepared_mask = cv2.bitwise_and(threshold_image, threshold_image, mask=self.image_masks.topo_mask)

		return prepared_mask

	def __prepare_for_second_contour_connecting(self, mask):
		skeleton_mask = self.__skeletonize_mask(mask)
		reduced_mask = Helper.reduce_image_contours(skeleton_mask, 1)
		dilated_mask = Helper.dilate_image(reduced_mask)

		return dilated_mask

	def __connect_contours_by_distances(self, mask, distances, min_contour_area):
		contour_connector = ContourConnector(mask)

		for distance in distances:
			contour_connector.connect_contours_within_distance(distance)
			Helper.reduce_image_contours(contour_connector.connected_contours_mask, min_contour_area)

		return contour_connector.connected_contours_mask

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
 


