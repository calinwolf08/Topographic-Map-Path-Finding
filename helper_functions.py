import cv2
import numpy as np
import math

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __str__(self):
		return "(" + str(self.x) + ", " + str(self.y) + ")"

	def __eq__(self, point):
		return self.x == point.x and self.y == point.y

	def __add__(self, point):
		return Point(self.x + point.x, self.y + point.y)

class Helper:
	resize_factor = 1

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
	# def dilate_image(image, array=(5,5)):
	def dilate_image(image, array=(1,1)):
		kernel = np.ones(array, np.uint8)
		dilated_image = cv2.dilate(image, kernel, iterations=1)

		return dilated_image

	@staticmethod
	def get_horizontal_lines(image):
		edges = cv2.Canny(image, 50, 115)
		lines = cv2.HoughLines(edges, 1, np.pi/180, 50)

		return lines

	@staticmethod
	def show_images_and_wait(images):
		x = 1

		for image in images:
			cv2.imshow(str(x), image)
			x += 1

		cv2.waitKey(0)
		cv2.destroyAllWindows()

	@staticmethod
	def reduce_image_contours(mask, minArea, line_thickness = 1):
		img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = list(filter(lambda c: cv2.contourArea(c) > minArea, contours))
		reduced = cv2.bitwise_xor(mask, mask)
		# cv2.drawContours(reduced, contours, -1, (255,255,255), cv2.FILLED)
		cv2.drawContours(reduced, contours, -1, (255,255,255), line_thickness)
		
		return reduced

	@staticmethod
	def convert_angle_to_grade(angle):
		theta = math.radians(angle)
		tangent = math.tan(theta)

		grade = tangent * 100

		return grade

	@staticmethod
	def convert_grade_to_angle(grade):
		x = grade / 100
		theta = math.atan(x)
		angle = math.degrees(theta)

		return angle
