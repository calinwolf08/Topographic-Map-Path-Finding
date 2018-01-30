import cv2
import numpy as np
import random, math

class ContourExtractor:

	def __init__(self, image):
		self.image = image

	@staticmethod
	def dist_to_pt(pt1, pt2):
		return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

	@staticmethod
	def get_contour_extremes(contours, allExtremes = False):
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
				if ContourExtractor.dist_to_pt(l, r) > ContourExtractor.dist_to_pt(t, b):
					ret.append((l, r))
				else:
					ret.append((t, b))

		return ret

	@staticmethod
	def connect_points(p1, p2, points, dist, img, epsilon, slope = 0, testSlope = False, 
		lineThickness = 4):
		points.sort(key=lambda x: ContourExtractor.dist_to_pt(p1, x))

		if len(points) > 2 and points[1] == p2:
			p = points[2]
		else:
			p = points[1]

		if ContourExtractor.dist_to_pt(p, p1) < dist:
			if testSlope is False:
				cv2.line(img, p1, p, (0,0,255), lineThickness)

	@staticmethod
	def connect_extremes(extremes, img, dist, epsilon, slope = 0, testSlope = False, 
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
				ContourExtractor.connect_points(e[0], e[1], points, dist, copy, epsilon, slope, testSlope, lineThickness)
			
			# second point in e
			x = e[1][0]
			y = e[1][1]

			if x > z and x < w-z and y > z and y < h-z:
				ContourExtractor.connect_points(e[1], e[0], points, dist, copy, epsilon, slope, testSlope, lineThickness)

		low_black = np.array([0, 0, 0]) #0,0,0
		high_black = np.array([1, 1, 1]) #255,255,85
		mask = cv2.inRange(copy, low_black, high_black)

		bw = cv2.bitwise_not(mask)

		return (copy, bw)

	@staticmethod
	def connect_contours(img, mask, dist, lineThickness = 4, allExtremes = False):
		(contours_img, contours) = ContourExtractor.get_contours(img, mask)
		numContours = len(contours)
		
		extremes = ContourExtractor.get_contour_extremes(contours, allExtremes)

		w, h, d = img.shape
		z = 10

		for e in extremes:
			cv2.circle(contours_img, e[0], 1, (255,0,0), 2)
			cv2.circle(contours_img, e[1], 1, (255,0,0), 2)
		
		color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

		(connected_img, connected_mask) = ContourExtractor.connect_extremes(
			extremes, color_mask, dist=dist, epsilon=0.5, testSlope = False, lineThickness = lineThickness)

		return (connected_mask, contours_img, numContours)

	@staticmethod
	def get_contours(img, mask):
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

	@staticmethod
	def connect_contours_loop(img, connected_mask, connected_color, dist, maxIters, 
		lineThickness = 4, allExtremes = False):
		prevNumContours = 0
		curNumContours = -1
		numIters = 0

		print("connecting contours, dist = " + str(dist))

		while prevNumContours != curNumContours and numIters < maxIters:
			(connected_mask, contours_img, numContours) = ContourExtractor.connect_contours(
				connected_color, connected_mask, dist, lineThickness, allExtremes)
			print(str(numIters) + ": " + str(numContours))
			print("-------")

			connected_color = cv2.bitwise_and(img, img, mask=connected_mask)

			prevNumContours = curNumContours
			curNumContours = numContours
			numIters += 1

		return (connected_mask, connected_color, contours_img)

	@staticmethod
	def reduce_image_contours(mask, minArea):
		img2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		contours = list(filter(lambda c: cv2.contourArea(c) > minArea, contours))
		
		reduced = cv2.bitwise_xor(mask, mask)
		cv2.drawContours(reduced, contours, -1, (255,255,255), cv2.FILLED)
		
		return reduced

	@staticmethod
	def skeletonize_mask(img):
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
	def run_connect_contours(cropped_img, connected_mask, connected_color, iters, 
		lineThickness = 4, minArea = 50, allExtremes = False):

		for (dist, maxIters) in iters:
			(connected_mask, connected_color, contours_img) = ContourExtractor.connect_contours_loop(
				cropped_img, connected_mask, connected_color, 
				dist=dist, maxIters=maxIters, lineThickness=lineThickness, allExtremes=allExtremes)

			connected_mask = ContourExtractor.reduce_image_contours(connected_mask, minArea)
			connected_color = cv2.bitwise_and(cropped_img, cropped_img, mask=connected_mask)

		return (connected_mask, connected_color, contours_img)

	@staticmethod
	def prepare_for_contour_extraction(img, mask):
		kernel = np.ones((5,5), np.uint8)
		dilated = cv2.dilate(img, kernel, iterations=1)

		base_img = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
		base_img = cv2.fastNlMeansDenoising(base_img, None, 5, 7, 21)

		base_img = cv2.threshold(base_img,225,255,cv2.THRESH_BINARY_INV)[1]
		img_mask = cv2.bitwise_and(base_img, base_img, mask=mask)

		img_color = cv2.bitwise_and(img, img, mask=img_mask)

		return (img_mask, img_color)
