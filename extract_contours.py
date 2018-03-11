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

		# temp = self.__test_extract_contours(temp)

		return temp
		
		# first_connected_mask = self.__connect_contours_by_distances(first_prepared_mask, distances[:3], min_contour_area)		
		# min_contour_area = 2
		# second_prepared_mask = self.__prepare_for_second_contour_connecting(first_connected_mask)
		# second_connected_mask = self.__connect_contours_by_distances(second_prepared_mask, distances[3:], min_contour_area)

		# return second_connected_mask

	def __test_extract_contours(self, image):
		non_zero_points = cv2.findNonZero(image)
		end_point_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

		for point in non_zero_points:
			# if count % 1000 == 0:
			# 	print(count)
			# count += 1
			pt = (point[0][0], point[0][1])

			if self.__is_end_point(pt, image):
				end_point_image[pt[1]][pt[0]] = (0,0,255)
				# cv2.circle(end_point_image, (pt[0], pt[1]), 1, (0,0,255), -1)

		temp = end_point_image[:200, :200]
		temp = cv2.resize(temp, None, fx=5, fy=5, interpolation = cv2.INTER_LINEAR)
		cv2.imshow("temp", temp)

		temp2 = image[:200, :200]
		temp2 = cv2.resize(temp2, None, fx=5, fy=5, interpolation = cv2.INTER_LINEAR)
		cv2.imshow("temp2", temp2)

		cv2.imshow("end_point_image", end_point_image)
		end_point_image = cv2.cvtColor(end_point_image, cv2.COLOR_BGR2GRAY)

		return image

	def __is_end_point(self, pt, image):
		#don't check points near end
		w, h = image.shape
		if pt[0] < 2 or pt[0] > w - 2 or pt[1] < 2 or pt[1] > h - 2:
			return 0

		non_zero_neighbors = 0
		prev = False
		found = False
		first = False

		# top left
		tl = (pt[0]-1, pt[1]-1)
		if image[tl[1]][tl[0]] == 255:	# if pt is non zero
			non_zero_neighbors += 1		# increment counter
			prev = True					# check prev bool
			first = True				# indicate tl is non zero for ml at the end

		# top middle
		tm = (pt[0], pt[1]-1)
		if image[tm[1]][tm[0]] == 255:	# if pt is non zero
			if prev:					# if prev pt was also non zero 
				found = True			# 2 non zero neighbors found
			
			non_zero_neighbors += 1
			prev = True
		else:
			prev = False

		# top right
		tr = (pt[0]+1, pt[1]-1)
		if image[tr[1]][tr[0]] == 255:
			if found:					# if 2 non zero neighbors already found	
				return False

			if non_zero_neighbors == 1:	# non zero neighbor has been found
				if prev:
					found = True		# other neighbor was adjacent so 2 non zero neighbors found
				else:
					return False 		# other neighbor was NOT adjacent

			non_zero_neighbors += 1
			prev = True
		else:
			prev = False

		# mid right
		mr = (pt[0]+1, pt[1])
		if image[mr[1]][mr[0]] == 255:
			if found:
				return False

			if non_zero_neighbors == 1:
				if prev:
					found = True	
				else:
					return False 

			non_zero_neighbors += 1
			prev = True
		else:
			prev = False

		# bottom right
		br = (pt[0]+1, pt[1]+1)
		if image[br[1]][br[0]] == 255:
			if found:
				return False

			if non_zero_neighbors == 1:
				if prev:
					found = True	
				else:
					return False 

			non_zero_neighbors += 1
			prev = True
		else:
			prev = False

		# bottom mid
		bm = (pt[0], pt[1]+1)
		if image[bm[1]][bm[0]] == 255:
			if found:
				return False

			if non_zero_neighbors == 1:
				if prev:
					found = True	
				else:
					return False 

			non_zero_neighbors += 1
			prev = True
		else:
			prev = False

		# bottom left
		bl = (pt[0]-1, pt[1]+1)
		if image[bl[1]][bl[0]] == 255:
			if found:
				return False

			if non_zero_neighbors == 1:
				if prev:
					found = True	
				else:
					return False 

			non_zero_neighbors += 1
			prev = True
		else:
			prev = False

		# mid left
		ml = (pt[0]-1, pt[1])
		if image[ml[1]][ml[0]] == 255:
			if found:
				return False

			if non_zero_neighbors == 1:
				if prev or first:
					found = True	
				else:
					return False 

			non_zero_neighbors += 1
		else:
			prev = False

		return found or non_zero_neighbors == 1

	def __is_end_point2(self, point, non_zero_points):
		pt = (point[0][0], point[0][1])

		neighbors = [
			(pt[0]-1, pt[1]-1),
			(pt[0], pt[1]-1),
			(pt[0]+1, pt[1]-1),
			(pt[0]-1, pt[1]),
			(pt[0]+1, pt[1]),
			(pt[0]-1, pt[1]+1),
			(pt[0], pt[1]+1),
			(pt[0]+1, pt[1]+1),
		]

		non_zero_neighbors = list(filter(lambda x: x in non_zero_points, neighbors))

		if len(non_zero_neighbors) == 1:
			return True
		elif len(non_zero_neighbors) == 2:
			p1 = non_zero_neighbors[0]
			p2 = non_zero_neighbors[1]

			distance = math.sqrt(math.pow(p1[0]-p2[0], 2) + math.pow(p1[1]-p2[1], 2))

			if distance == 1:
				return True

		return False

	def __prepare_for_first_contour_connecting(self):
		dilated_image = Helper.dilate_image(self.cv_image)
		dilated_mask = Helper.convert_image_to_mask(dilated_image)
		gray_denoised_image = cv2.fastNlMeansDenoising(dilated_mask, None, 5, 7, 21)
		# threshold_image = cv2.threshold(gray_denoised_image,225,255,cv2.THRESH_BINARY_INV)[1]
		threshold_image = cv2.adaptiveThreshold(gray_denoised_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
		cv2.imshow("blah", threshold_image)

		prepared_mask = cv2.bitwise_and(threshold_image, threshold_image, mask=self.image_masks.topo_mask)

		return prepared_mask

	def __prepare_for_second_contour_connecting(self, mask):
		skeleton_mask = self.__skeletonize_mask(mask)
		# reduced_mask = Helper.reduce_image_contours(skeleton_mask, 1)
		# dilated_mask = Helper.dilate_image(reduced_mask)
		dilated_mask = Helper.dilate_image(skeleton_mask)
		cv2.imshow("blah2", dilated_mask)

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
 


