import cv2
import numpy as np

from extract_contours import ContourExtractor

class MaskGenerator:
	low_blue = np.array([50, 35, 100])
	high_blue = np.array([100, 150, 255])

	low_black = np.array([0, 0, 0])
	high_black = np.array([100, 100, 150])

	low_red = np.array([0, 50, 200])
	high_red = np.array([10, 255, 255])

	def __init__(self, image):
		self.bgr_image = image

		self.hsv_image = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2HSV)
		self.temp_image = cv2.bitwise_xor(self.bgr_image, self.bgr_image)
		
		self.blue_mask = None
		self.black_mask = None
		self.red_lines_mask = None
		self.topo_mask = None

	def get_topo_mask():
		if self.topo_mask == None:
			self.generate_masks()

		return self.topo_mask

	def generate_masks(self):
		self.blue_mask = self.generate_blue_mask()
		self.black_mask = self.generate_black_mask()
		self.red_lines_mask = self.generate_red_lines_mask()
		self.topo_mask = self.__combine_masks()

	def generate_blue_mask(self):
		blue_range = self.__get_image_in_range_from_hsv(MaskGenerator.low_blue, MaskGenerator.high_blue)
		filled_blue_contours = self.__get_filled_contours_from_image(blue_range)
		blue_mask = MaskGenerator.convert_image_to_mask(filled_blue_contours)
		
		return blue_mask

	def generate_black_mask(self):
		black_range = self.__get_image_in_range_from_hsv(MaskGenerator.low_black, MaskGenerator.high_black)
		filled_black_contours = self.__get_filled_contours_from_image(black_range)
		dilated_black_contours = MaskGenerator.dilate_image(filled_black_contours)

		black_contours_mask = MaskGenerator.convert_image_to_mask(dilated_black_contours)
		
		black_contours_mask_reduced = ContourExtractor.reduce_image_contours(black_contours_mask, 75)
		black_contours_mask_reduced_color = self.__add_color_to_image(black_contours_mask_reduced)
		
		(connected_mask, connected_color, contours_img) = ContourExtractor.connect_contours_loop(
			self.bgr_image, black_contours_mask_reduced, black_contours_mask_reduced_color, dist=30, maxIters=10)

		black_contours_connected_reduced = ContourExtractor.reduce_image_contours(connected_mask, 1000)
		black_mask = MaskGenerator.dilate_image(black_contours_connected_reduced)

		return black_mask

	def generate_red_lines_mask(self):
		red_range = self.__get_image_in_range_from_hsv(MaskGenerator.low_red, MaskGenerator.high_red)
		red_range_color = self.__add_color_to_image(red_range)
		lines = MaskGenerator.get_horizontal_lines(red_range_color)
		lines_image = self.__get_lines_image(lines)
		red_lines_mask = MaskGenerator.convert_image_to_mask(lines_image)

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

	def __combine_masks(self):
		non_blue_mask = cv2.bitwise_not(self.blue_mask)
		non_black_mask = cv2.bitwise_not(self.black_mask)
		non_red_mask = cv2.bitwise_not(self.red_lines_mask)

		temp = cv2.bitwise_and(non_blue_mask, non_black_mask)
		combined_mask = cv2.bitwise_and(temp, non_red_mask)

		return combined_mask

	def __clear_temp_image(self):
		self.temp_image = cv2.bitwise_xor(self.temp_image, self.temp_image)

	@staticmethod
	def convert_image_to_mask(image):
		mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		return mask

	@staticmethod
	def convert_image_to_inverted_mask(image):
		mask = MaskGenerator.convert_image_to_mask(image)
		inverted_mask = cv2.bitwise_not(mask)

		return inverted_mask

	@staticmethod
	def dilate_image(image):
		kernel = np.ones((5,5), np.uint8)
		dilated_image = cv2.dilate(image, kernel, iterations=1)

		return dilated_image

	@staticmethod
	def get_horizontal_lines(image):
		edges = cv2.Canny(image, 50, 115)
		lines = cv2.HoughLines(edges, 1, np.pi/180, 50)

		return lines






