import cv2
import numpy as np

from extract_contours import ContourExtractor, ContourConnector
from helper_functions import Helper

class ImageMasks:
	def __init__(self):
		self.topo_mask = None
		self.blue_mask = None
		self.black_mask = None
		self.red_mask = None
		self.green_mask = None

	def is_generated(self):
		is_generated = True

		if self.topo_mask == None or self.blue_mask == None or self.black_mask == None or self.red_mask == None:
			is_generated = False

		return is_generated

	def show_masks(self):
		cv2.imshow('topo_mask', self.topo_mask)
		cv2.imshow('blue_mask', self.blue_mask)
		cv2.imshow('black_mask', self.black_mask)
		cv2.imshow('red_mask', self.red_mask)
		cv2.imshow('green_mask', self.green_mask)

class MaskGenerator:
	low_blue = np.array([50, 35, 100])
	high_blue = np.array([100, 150, 255])

	low_black = np.array([0, 0, 0])
	high_black = np.array([100, 100, 150])

	low_red = np.array([0, 75, 210])
	high_red = np.array([10, 255, 255])

	low_green = np.array([20, 35, 100])
	high_green = np.array([50, 150, 255])

	def __init__(self, cv_image):
		self.bgr_image = cv_image
		self.hsv_image = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2HSV)
		self.__temp_image = cv2.bitwise_xor(self.bgr_image, self.bgr_image)
		self.image_masks = self.__get_image_masks()

	def __get_image_masks(self):
		image_masks = ImageMasks()

		image_masks.blue_mask = self.__generate_blue_mask()
		image_masks.black_mask = self.__generate_black_mask()
		image_masks.red_mask = self.__generate_red_mask()
		image_masks.green_mask = self.__generate_green_mask()
		image_masks.topo_mask = self.__generate_topo_mask(image_masks)

		return image_masks

	def __generate_blue_mask(self):
		blue_range = self.__get_image_in_range_from_hsv(MaskGenerator.low_blue, MaskGenerator.high_blue)
		filled_blue_contours = self.__get_filled_contours_from_image(blue_range)
		blue_mask = Helper.convert_image_to_mask(filled_blue_contours)
		
		return blue_mask

	def __generate_black_mask(self):
		black_mask = self.__generate_general_color_lines_mask(MaskGenerator.low_black, MaskGenerator.high_black)

		return black_mask

	def __generate_red_mask(self):
		red_mask = self.__generate_general_color_lines_mask(MaskGenerator.low_red, MaskGenerator.high_red)

		return red_mask

	def __generate_green_mask(self):
		green_range = self.__get_image_in_range_from_hsv(MaskGenerator.low_green, MaskGenerator.high_green)
		filled_green_contours = self.__get_filled_contours_from_image(green_range)
		green_mask = Helper.convert_image_to_mask(filled_green_contours)
		green_mask_reduced = Helper.reduce_image_contours(green_mask, 200)

		return green_mask_reduced

	def __generate_general_color_lines_mask(self, low_range, high_range):
		color_range = self.__get_image_in_range_from_hsv(low_range, high_range)

		filled_contours = self.__get_filled_contours_from_image(color_range)
		dilated_contours = Helper.dilate_image(filled_contours)

		contours_mask = Helper.convert_image_to_mask(dilated_contours)
		
		contours_mask_reduced = Helper.reduce_image_contours(contours_mask, 75)
		contours_mask_reduced_color = self.__add_color_to_image(contours_mask_reduced)
		
		contour_connector = ContourConnector(contours_mask_reduced)
		contour_connector.connect_contours_within_distance(30)
		connected_mask = contour_connector.connected_contours_mask

		contours_connected_reduced = Helper.reduce_image_contours(connected_mask, 1000)
		mask = Helper.dilate_image(contours_connected_reduced)

		return mask

	def __get_image_in_range_from_hsv(self, low, high):
		image_in_range = cv2.inRange(self.hsv_image, low, high)
		
		return image_in_range

	def __add_color_to_image(self, image):
		colored_image = cv2.bitwise_and(self.bgr_image, self.bgr_image, mask=image)

		return colored_image

	def __get_filled_contours_from_image(self, image):
		img, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		filled_contours = cv2.drawContours(self.__temp_image, contours, -1, (255,255,255), cv2.FILLED)
		self.__clear_temp_image()

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

		temp = cv2.bitwise_and(non_blue_mask, non_black_mask)
		combined_mask = cv2.bitwise_and(temp, non_red_mask)

		return combined_mask

	def __clear_temp_image(self):
		self.__temp_image = cv2.bitwise_xor(self.__temp_image, self.__temp_image)






