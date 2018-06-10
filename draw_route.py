import cv2
import time
import os
import numpy as np
from helper_functions import Point
from cropped_image import CroppedImage
from topographic_map import TopographicMap

def find_cropped_image(start, end, topo_map, padding = 500):
    # calculate padding needed for each point  
    yPad = padding
    xPad = padding

    # crop image around start and end points with padding
    minY = max(min(start.y, end.y) - yPad, 0)
    maxY = min(max(start.y, end.y) + yPad, topo_map.height)

    minX = max(min(start.x, end.x) - xPad, 0)
    maxX = min(max(start.x, end.x) + xPad, topo_map.width)

    img = topo_map.image[minY : maxY, minX : maxX]

    # calculate start/end points for cropped image
    # width/height of cropped image
    width = maxX - minX
    height = maxY - minY

    # ratio of start/end points to cropped image size
    sxFactor = ((start.x - minX) / width)
    syFactor = ((start.y - minY) / height)
    exFactor = ((end.x - minX) / width)
    eyFactor = ((end.y - minY) / height)

    # init cropped_img for extracting contours, etc.		
    cropped_img = CroppedImage(img)

    # use ratios to find scaled start/end points 
    cropped_img.start = Point(int(sxFactor * width), int(syFactor * height))
    cropped_img.end = Point(int(exFactor * width), int(eyFactor * height))

    return cropped_img

def draw_image(self, img_name, img):
    h = img.shape[0]
    w = img.shape[1]

    max_h = 500
    num = int(h / max_h)

    for i in range(num - 1):
        start = i*max_h
        end = start + max_h
        cv2.imshow(str(i) + '_' + img_name + str(time.time()), img[start:end,:])

    start = (num-1) * max_h
    cv2.imshow(str(num-1) + '_' + img_name + str(time.time()), img[start:,:])

temp_img = None

class DrawRoute:
    def __init__(self, image):
        self.image = image
        self.cur_start_point = None
        self.cur_end_point = None
        self.points = []
    
    def draw_route_in_pieces(self):
        h = self.image.shape[0]
        w = self.image.shape[1]

        max_h = 750
        num = int(h / max_h)

        temp_image = self.image.copy()
        points = []

        for i in range(num - 1):
            start = i*max_h
            end = start + max_h

            img, ps = self.draw_route(temp_image[start:end,:])

            for p in ps:
                points.append(Point(p.x, p.y+start))
            
        start = (num-1) * max_h

        img, ps = self.draw_route(temp_image[start:,:])

        for p in ps:
            points.append(Point(p.x, p.y+start))

        first = True
        prev = None
        for p in points:
            cv2.circle(temp_image, (p.x,p.y), 5, (255,0,0), 2)
            if first:
                first = False
            else:
                cv2.line(temp_image, (prev.x, prev.y),(p.x, p.y),(255,0,0),2)
            prev = p

        return temp_image, points

    def draw_route(self, image):
        self.temp_img = image.copy()
        self.cur_start_point = None
        self.cur_end_point = None
        self.points = []

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.__click_image)
        cv2.imshow("image", self.temp_img)

        while(True):
            k = cv2.waitKey(1000) & 0xFF
            cv2.imshow("image", self.temp_img)

            if k == ord('q') or k == ord(' '):
                break
        
        cv2.destroyAllWindows()
        return self.temp_img, self.points

    def __click_image(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            if self.cur_start_point is None:
                self.cur_start_point = Point(x,y)
                self.points.append(self.cur_start_point)
                cv2.circle(self.temp_img, (x,y), 5, (255,0,0), 2)
            else:
                self.cur_end_point = Point(x,y)
                self.points.append(self.cur_end_point)
                cv2.line(self.temp_img,\
                    (self.cur_start_point.x, self.cur_start_point.y),\
                    (self.cur_end_point.x, self.cur_end_point.y),\
                    (255,0,0),2)
                self.cur_start_point = self.cur_end_point
                cv2.circle(self.temp_img, (x,y), 5, (0,0,255), 2)

def get_drawn_route(start, end, topo_map):
    cropped_image = find_cropped_image(start, end, topo_map)
    route_image, points = DrawRoute(cropped_image.cv_image).draw_route_in_pieces()

    return route_image, points

def points_to_path_string(start, end):
		return str(start.x) + "_" + str(start.y) + "to" + str(end.x) + "_" + str(end.y)

def draw_save_route(start, end, topo_map):
    route_image, points = get_drawn_route(start, end, topo_map)

    path = "images/"+topo_map.filename[5:-4] + "/" + points_to_path_string(start, end) + "/drawn"
	
    if not os.path.exists(path):
        os.makedirs(path)
    
    num = len([x for x in os.listdir(path)]) + 1
    cv2.imwrite(path + '/drawn' + str(int(num/2)) + '.png', route_image)
    out = open(path + "/data" + str(int(num/2)) + '.csv', 'w')

    t = input("enter completion time:")
    m = input("enter mile length:")
    n = input("enter initial:")

    out.write("x,y," + str(t) + ',' + str(m) + ',' + str(n) + "\n")

    for p in points:
        out.write(str(p.x) + "," + str(p.y) + "\n")

    out.close()

if __name__ == "__main__":
    topo_map = TopographicMap("maps/SanLuisObispo.jpg")
    start = Point(550, 435)
    end = Point(1873, 1081)

    get_drawn_route(start, end, topo_map)
