import cv2
import numpy as np

from extract_contours import ContourExtractor, ContourConnector
from helper_functions import Helper

class MaskGenerator:
	__low_blue = np.array([50, 35, 100])
	__high_blue = np.array([100, 150, 255])

	__low_black = np.array([0, 0, 0])
	__high_black = np.array([100, 100, 150])

	__low_red = np.array([0, 50, 200])
	__high_red = np.array([10, 255, 255])

	def __init__(self, cv_image):
		self.bgr_image = cv_image

		self.hsv_image = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2HSV)
		self.temp_image = cv2.bitwise_xor(self.bgr_image, self.bgr_image)
		
		self.image_masks = MaskGenerator.ImageMasks()

	class ImageMasks:
		def __init__(self):
			self.topo_mask = None
			self.blue_mask = None
			self.black_mask = None
			self.red_lines_mask = None

		def is_generated(self):
			is_generated = True

			if self.topo_mask == None or self.blue_mask == None or self.black_mask == None or self.red_lines_mask == None:
				is_generated = False

			return is_generated

	def get_image_masks(self):
		if not self.image_masks.is_generated():
			self.generate_masks()

		return self.image_masks

	def generate_masks(self):
		self.image_masks.blue_mask = self.generate_blue_mask()
		self.image_masks.black_mask = self.generate_black_mask()
		self.image_masks.red_lines_mask = self.generate_red_lines_mask()
		self.image_masks.topo_mask = self.__generate_topo_mask()

	def generate_blue_mask(self):
		blue_range = self.__get_image_in_range_from_hsv(MaskGenerator.__low_blue, MaskGenerator.__high_blue)
		filled_blue_contours = self.__get_filled_contours_from_image(blue_range)
		blue_mask = Helper.convert_image_to_mask(filled_blue_contours)
		
		return blue_mask

	def generate_black_mask(self):
		black_range = self.__get_image_in_range_from_hsv(MaskGenerator.__low_black, MaskGenerator.__high_black)
		filled_black_contours = self.__get_filled_contours_from_image(black_range)
		dilated_black_contours = Helper.dilate_image(filled_black_contours)

		black_contours_mask = Helper.convert_image_to_mask(dilated_black_contours)
		
		black_contours_mask_reduced = ContourExtractor.reduce_image_contours(black_contours_mask, 75)
		black_contours_mask_reduced_color = self.__add_color_to_image(black_contours_mask_reduced)
		
		contour_connector = ContourConnector(black_contours_mask_reduced)
		contour_connector.connect_contours_within_distance(30)
		connected_mask = contour_connector.connected_contours_mask

		black_contours_connected_reduced = ContourExtractor.reduce_image_contours(connected_mask, 1000)
		black_mask = Helper.dilate_image(black_contours_connected_reduced)

		return black_mask

	def generate_red_lines_mask(self):
		red_range = self.__get_image_in_range_from_hsv(MaskGenerator.__low_red, MaskGenerator.__high_red)
		red_range_color = self.__add_color_to_image(red_range)
		lines = Helper.get_horizontal_lines(red_range_color)
		lines_image = self.__get_lines_image(lines)
		red_lines_mask = Helper.convert_image_to_mask(lines_image)

		return red_lines_mask

	def __get_image_in_range_from_hsv(self, low, high):
		image_in_range = cv2.inRange(self.hsv_image, low, high)
		
		return image_in_range

	def __add_color_to_image(self, image):
		colored_image = cv2.bitwise_and(self.bgr_image, self.bgr_image, mask=image)

		return colored_image

	def __get_filled_contours_from_image(self, image):
		img, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		filled_contours = cv2.drawContours(self.temp_image, contours, -1, (255,255,255), cv2.FILLED)
		self.__clear_temp_image()

		return filled_contours

	def __get_lines_image(self, lines):
		height, width, chanels = self.temp_image.shape

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
				    	cv2.line(self.temp_image,(x1,y1),(x2,y2),(255,255,255),15)

		lines_image = self.temp_image.copy()
		self.__clear_temp_image()

		return lines_image

	def __generate_topo_mask(self):
		non_blue_mask = cv2.bitwise_not(self.image_masks.blue_mask)
		non_black_mask = cv2.bitwise_not(self.image_masks.black_mask)
		non_red_mask = cv2.bitwise_not(self.image_masks.red_lines_mask)

		temp = cv2.bitwise_and(non_blue_mask, non_black_mask)
		combined_mask = cv2.bitwise_and(temp, non_red_mask)

		return combined_mask

	def __clear_temp_image(self):
		self.temp_image = cv2.bitwise_xor(self.temp_image, self.temp_image)






