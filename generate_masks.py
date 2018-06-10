import cv2
import numpy as np

from extract_contours import ContourExtractor, ContourConnector
from helper_functions import Helper

import time

class ImageMasks:
	def __init__(self):
		self.topo_mask = None
		self.blue_mask = None
		self.black_mask = None
		self.red_mask = None
		self.green_mask = None
		self.steps = []

	def is_generated(self):
		is_generated = True

		if self.topo_mask == None or self.blue_mask == None or self.black_mask == None or self.red_mask == None:
			is_generated = False

		return is_generated

	def show_masks(self):
		# cv2.imshow('topo_mask', self.topo_mask)
		# cv2.imshow('blue_mask', self.blue_mask)
		# cv2.imshow('black_mask', self.black_mask)
		cv2.imshow('red_mask', self.red_mask)
		# cv2.imshow('green_mask', self.green_mask)

class MaskGenerator:
	low_blue = np.array([50, 35, 100], dtype=np.uint8)
	high_blue = np.array([100, 150, 255], dtype=np.uint8)

	low_black = np.array([0, 0, 0], dtype=np.uint8)
	high_black = np.array([100, 75, 150], dtype=np.uint8)

	low_red = np.array([0, 150, 150], dtype=np.uint8)
	high_red = np.array([255, 255, 255], dtype=np.uint8)

	low_green = np.array([20, 35, 100], dtype=np.uint8)
	high_green = np.array([50, 150, 255], dtype=np.uint8)

	def __init__(self, cv_image):
		self.bgr_image = cv_image
		self.hsv_image = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2HSV)
		self.__temp_image = cv2.bitwise_xor(self.bgr_image, self.bgr_image)
		self.dilate_array = (2 * Helper.resize_factor, 2 * Helper.resize_factor)
		self.__get_image_masks()

	def __get_image_masks(self):
		self.image_masks = ImageMasks()
		self.image_masks.blue_mask = self.__generate_blue_mask()
		self.image_masks.black_mask = self.__generate_black_mask()
		self.image_masks.red_mask = self.__generate_red_mask()
		self.image_masks.green_mask = self.__generate_green_mask()
		self.image_masks.topo_mask = self.__generate_topo_mask(self.image_masks)

	def __generate_blue_mask(self):
		blue_range = self.__get_image_in_range_from_hsv(MaskGenerator.low_blue, MaskGenerator.high_blue)
		self.image_masks.steps.append(("blue_range", blue_range))
		filled_blue_contours = self.__get_filled_contours_from_image(blue_range)
		self.image_masks.steps.append(("blue_filled", filled_blue_contours))
		blue_mask = Helper.convert_image_to_mask(filled_blue_contours)
		dilated = Helper.dilate_image(blue_mask, array=(2,2))
		self.image_masks.steps.append(("blue_dilated", dilated))
		blue_mask = Helper.reduce_image_contours(dilated, 15, line_thickness = cv2.FILLED)
		
		return blue_mask

	def __generate_black_mask(self):
		black_range = self.__get_image_in_range_from_hsv(MaskGenerator.low_black, MaskGenerator.high_black)
		filled_contours = self.__get_filled_contours_from_image(black_range)
		contours_mask = Helper.convert_image_to_mask(filled_contours)
		dilated = Helper.dilate_image(contours_mask, array=self.dilate_array)
		black_mask = Helper.reduce_image_contours(dilated, 6, line_thickness = cv2.FILLED)

		return black_mask

	def __generate_red_mask(self):
		red_range = self.__get_image_in_range_from_hsv(MaskGenerator.low_red, MaskGenerator.high_red)
		# cv2.imshow("red range", red_range)
		filled_contours = self.__get_filled_contours_from_image(red_range)
		# cv2.imshow("filled contours", filled_contours)
		contours_mask = Helper.convert_image_to_mask(filled_contours)
		# cv2.imshow("contours mask", contours_mask)
		dilated = Helper.dilate_image(contours_mask, array=self.dilate_array)
		# cv2.imshow("dilated", dilated)
		reduced = Helper.reduce_image_contours(dilated, 6, line_thickness = cv2.FILLED)
		# cv2.imshow("red range", red_range)

		return reduced

	def __generate_green_mask(self):
		green_range = self.__get_image_in_range_from_hsv(MaskGenerator.low_green, MaskGenerator.high_green)
		filled_green_contours = self.__get_filled_contours_from_image(green_range)
		green_mask = Helper.convert_image_to_mask(filled_green_contours)
		green_mask_reduced = Helper.reduce_image_contours(green_mask, 200, line_thickness = cv2.FILLED)

		return green_mask_reduced

	def __generate_general_color_lines_mask(self, low_range, high_range):
		color_range = self.__get_image_in_range_from_hsv(low_range, high_range)
		filled_contours = self.__get_filled_contours_from_image(color_range)
		contour_mask = Helper.convert_image_to_contour_mask(filled_contours)
		dilated = Helper.dilate_image(contour_mask, array=(2,2))
		mask = Helper.reduce_image_contours(dilated, 6, line_thickness = cv2.FILLED)

		return mask

	def __get_image_in_range_from_hsv(self, low, high):
		image_in_range = cv2.inRange(self.hsv_image, low, high)
		
		return image_in_range

	def __add_color_to_image(self, image):
		colored_image = cv2.bitwise_and(self.bgr_image, self.bgr_image, mask=image)

		return colored_image
#2242,152,2190,1612 d r p
#21170,315,180,2002 r
	def __get_filled_contours_from_image(self, image):
		img, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		if len(contours) < Helper.MAX_CONTOURS:
			r, c = image.shape
			temp = self.__temp_image[:r, :c]
			start = time.time()
			filled_contours = cv2.drawContours(temp, contours, -1, (255,255,255), cv2.FILLED)
			end = time.time()
			# print('filled: ' + str(len(contours)) + ": " + str(end-start) + ": " + str(filled_contours.shape))
			self.__clear_temp_image()
			# print('indiv: ' + str(filled_contours.shape))
		else:
			r, c = image.shape

			tl = image[:int(r/2), :int(c/2)]
			tl = self.__get_filled_contours_from_image(tl)

			tr = image[:int(r/2), int(c/2):]
			tr = self.__get_filled_contours_from_image(tr)

			bl = image[int(r/2):, :int(c/2)]
			bl = self.__get_filled_contours_from_image(bl)

			br = image[int(r/2):, int(c/2):]
			br = self.__get_filled_contours_from_image(br)

			top = np.concatenate((tl, tr), axis=1)
			bottom = np.concatenate((bl, br), axis=1)
			filled_contours = np.concatenate((top, bottom), axis=0)

		return filled_contours

	def __get_lines_image(self, lines):
		height, width, chanels = self.__temp_image.shape

		if lines is not None:
			for line in lines:
				for rho,theta in line:
				    a = np.cos(theta)
				    b = np.sin(theta)
				    x0 = a*rho
				    y0 = b*rho
				    x1 = int(x0 + width*(-b))
				    y1 = int(y0 + width*(a))
				    x2 = int(x0 - width*(-b))
				    y2 = int(y0 - width*(a))

				    rise = y2 - y1
				    run = x2 - x1
				    theta = np.arctan(rise / run) if run != 0 else 90

				    if abs(theta) < np.pi / 1000:
				    	cv2.line(self.__temp_image,(x1,y1),(x2,y2),(255,255,255),15)

		lines_image = self.__temp_image.copy()
		self.__clear_temp_image()

		return lines_image

	def __generate_topo_mask(self, image_masks):
		non_blue_mask = cv2.bitwise_not(image_masks.blue_mask)
		non_black_mask = cv2.bitwise_not(image_masks.black_mask)
		non_red_mask = cv2.bitwise_not(image_masks.red_mask)

		# print(non_blue_mask.shape)
		# print(non_black_mask.shape)

		temp = cv2.bitwise_and(non_blue_mask, non_black_mask)
		combined_mask = cv2.bitwise_and(temp, non_red_mask)

		self.image_masks.steps.append(("combined_mask", combined_mask))

		return combined_mask

	def __clear_temp_image(self):
		self.__temp_image = cv2.bitwise_xor(self.__temp_image, self.__temp_image)
