from turtle import right
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
# from combined_thresh import combined_thresh
# from perspective_transform import perspective_transform

# feel free to adjust the parameters in the code if necessary

def line_fit(binary_warped):
	"""
	Find and fit lane lines
	"""
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	# Create an output image to draw on and visualize the result
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	# leftx_base = np.argmax(histogram[300:midpoint]) + 100
	# rightx_base = np.argmax(histogram[midpoint:-300]) + midpoint
	leftx_base = np.argmax(histogram[0:midpoint])
	rightx_base = np.argmax(histogram[midpoint:-1]) + midpoint

	# print('---')
	# print('Left Base:', leftx_base / histogram.shape[0])
	# print('Right Base:', rightx_base / histogram.shape[0])
	# print('---')

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	left_poly_x = []
	left_poly_y = []
	right_poly_x = []
	right_poly_y = []

	imgW = out_img.shape[1]
	imgH = out_img.shape[0]
	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		##TO DO
		leftline_LX = leftx_current - margin - 1
		leftline_LY = imgH - window_height * (window + 1) - 1
		leftline_RX = leftx_current + margin - 1
		leftline_RY = imgH - window_height * window - 1

		rightline_LX = rightx_current - margin - 1
		rightline_LY = imgH - window_height * (window + 1) - 1
		rightline_RX = rightx_current + margin - 1
		rightline_RY = imgH - window_height * window - 1

		left_poly_x.append(leftx_current)
		left_poly_y.append(leftline_RY)
		right_poly_x.append(rightx_current)
		right_poly_y.append(rightline_RY)

		####
		# Draw the windows on the visualization image using cv2.rectangle()
		##TO DO
		cv2.rectangle(out_img, (leftline_LX, leftline_LY), (leftline_RX, leftline_RY), (0, 255, 0), 1)
		cv2.rectangle(out_img, (rightline_LX, rightline_LY), (rightline_RX, rightline_RY), (0, 255, 0), 1)
		####
		# Identify the nonzero pixels in x and y within the window
		##TO DO
		left_idx = []
		right_idx = []
		for i in range(len(nonzerox)):
			if leftline_LX <= nonzerox[i] and nonzerox[i] <= leftline_RX and leftline_LY <= nonzeroy[i] and nonzeroy[i] <= leftline_RY:
				left_idx.append(i)
			if rightline_LX <= nonzerox[i] and nonzerox[i] <= rightline_RX and rightline_LY <= nonzeroy[i] and nonzeroy[i] <= rightline_RY:
				right_idx.append(i)
		####
		# Append these indices to the lists
		##TO DO
		if left_idx:
			left_lane_inds.append(np.asarray(left_idx))
		if right_idx:
			right_lane_inds.append(np.asarray(right_idx))

		####
		# If you found > minpix pixels, recenter next window on their mean position
		##TO DO
		if (len(left_idx) > minpix):
			leftx_current = int(np.mean(nonzerox[left_idx]))
		if (len(right_idx) > minpix):
			rightx_current = int(np.mean(nonzerox[right_idx]))

	# cv2.imwrite("test_images/23.png", out_img)
	# Concatenate the arrays of indices
	if len(left_lane_inds) != 0:
		left_lane_inds = np.asarray(np.asarray(left_lane_inds, dtype=object))
		left_lane_inds = np.concatenate(left_lane_inds).tolist()
	if len(right_lane_inds) != 0:
		right_lane_inds = np.asarray(right_lane_inds, dtype=object)
		right_lane_inds = np.concatenate(right_lane_inds).tolist()
	# if left_lane_inds.shape[0] == 0 or right_lane_inds.shape[0] == 0:
	# 	return None

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each using np.polyfit()
	# If there isn't a good fit, meaning any of leftx, lefty, rightx, and righty are empty,
	# the second order polynomial is unable to be sovled.
	# Thus, it is unable to detect edges.
	try:
		left_fit = np.polyfit(left_poly_y, left_poly_x, 2)
		right_fit = np.polyfit(right_poly_y, right_poly_x, 2)

		# left_curverad, right_curverad = cal_radius(left_poly_y, left_poly_x, right_poly_y, right_poly_x)
	####
	except TypeError:
		print("Unable to detect lanes")
		return None

	# Return a dict of relevant variables
	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['out_img'] = out_img
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds
	# ret['left_curverad'] = left_curverad
	# ret['right_curverad'] = right_curverad

	return ret

def cal_radius(imgH, imgW, left_fit, right_fit):
	# 图像中像素个数与实际中距离的比率
	# 沿车行进的方向长度大概覆盖了30米，按照美国高速公路的标准，宽度为3.7米（经验值）
	ym_per_pix = 30 / imgH  # y方向像素个数与距离的比例
	xm_per_pix = 100 / imgW  # x方向像素个数与距离的比例		需要修改这个3.7

	ploty=np.linspace(0, imgH-1, num=imgH)
	y_eval = np.max(ploty)
	leftx = left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
	rightx = right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]

	left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	xoffset = (left_fit_cr[2]+right_fit_cr[2])/2-imgW*xm_per_pix/2

	return left_curverad, right_curverad, xoffset


# # 1. 定义函数计算图像的中心点位置
# def cal_line__center(img):
#     undistort_img = img_undistort(img, mtx, dist)
#     rigin_pipline_img = pipeline(undistort_img)
#     transform_img = img_perspect_transform(rigin_pipline_img, M)
#     left_fit, right_fit = cal_line_param(transform_img)
#     y_max = img.shape[0]
#     left_x = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
#     right_x = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]

# def cal_center_departure(img, left_fit, right_fit):
 
#     # 计算中心点
#     y_max = img.shape[0]
#     left_x = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
#     right_x = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
#     xm_per_pix = 3.7 / 700

# 	center_depart = ((left_x + right_x) / 2 - lane_center) * xm_per_pix

def tune_fit(binary_warped, left_fit, right_fit):
	"""
	Given a previously fit line, quickly try to find the line based on previous lines
	"""
	# Assume you now have a new warped binary image
	# from the next frame of video (also called "binary_warped")
	# It's now much easier to find line pixels!
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 100
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# If we don't find enough relevant points, return all None (this means error)
	min_inds = 10
	if lefty.shape[0] < min_inds or righty.shape[0] < min_inds:
		return None

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Return a dict of relevant variables
	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds

	return ret


def viz1(binary_warped, ret, save_file=None):
	"""
	Visualize each sliding window location and predicted lane lines, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	out_img = ret['out_img']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	if save_file is None:
		plt.show()
	else:
		plt.savefig(save_file)
	plt.gcf().clear()


def bird_fit(binary_warped, ret, save_file=None):
	"""
	Visualize the predicted lane lines with margin, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Create an image to draw on and an image to show the selection window
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	margin = 100  # NOTE: Keep this in sync with *_fit()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	plt.imshow(result)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)

	# print(left_line_pts.shape)
	# left_line_cloud = list()
	# right_line_cloud = list()
	# for pixel in left_line_pts:
	# 	left_line_cloud.append(point_cloud.get_value(pixel[0], pixel[1]))
	# for pixel in right_line_pts:
	# 	right_line_cloud.append(point_cloud.get_value(pixel[0], pixel[1]))
	# print('-------')
	# print(left_line_cloud[0])

	# cv2.imshow('bird',result)
	# cv2.imwrite('bird_from_cv2.png', result)

	# if save_file is None:
	# 	plt.show()
	# else:
	# 	plt.savefig(save_file)
	# plt.gcf().clear()

	return result

def getXYZ(u, v, depth):
	fx, fy = 527.16845703125, 527.16845703125
	cx, cy = 616.2457275390625, 359.22479248046875

	z = depth
	x = ((u - cx) * z) / fx
	y = ((v - cy) * z) / fy

	return x, y, z


def final_viz(state, undist, depth, left_fit, right_fit, m_inv):
	"""
	Final lane line prediction visualized and overlayed on top of original image
	"""
	# Generate x and y values for plotting
	ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Create an image to draw the lines on
	#warp_zero = np.zeros_like(warped).astype(np.uint8)
	#color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	color_warp = np.zeros((720, 1280, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left_raw = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right_raw = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

	pts_left = list()
	pts_right = list()
	for i in pts_left_raw[0]:
		if 0 <= i[0] and i[0] < 1280 and 0 <= i[1] and i[1] < 720:
			pts_left.append(i)
	for i in pts_right_raw[0]:
		if 0 <= i[0] and i[0] < 1280 and 0 <= i[1] and i[1] < 720:
			pts_right.append(i)
	pts_left = np.array([pts_left])
	pts_right = np.array([pts_right])

	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	# 1280, 720
	# cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
	# if state == 'Go Ahead' or state == 'Turn Right':
	# 	cv2.polylines(color_warp, np.int_(pts_left), False, (255, 0, 0), 20)
	# if state == 'Go Ahead' or state == 'Turn Left':
	# 	cv2.polylines(color_warp, np.int_(pts_right), False, (0, 0, 255), 20)
	
	if state == 'Go Ahead':
		cv2.arrowedLine(color_warp, (640, 540), (640, 180), (0, 255, 0), 20, tipLength=0.3)
	elif state == 'Turn Left':
		cv2.arrowedLine(color_warp, (960, 200), (320, 200), (0, 255, 0), 20, tipLength=0.3)
	elif state == 'Turn Right':
		cv2.arrowedLine(color_warp, (320, 200), (960, 200), (0, 255, 0), 20, tipLength=0.3)
	else:
		cv2.circle(color_warp, (640, 200), 100, (0, 255, 0), 20)

	# cv2.arrowedLine(color_warp, (1280, 540), (1280, 180), (0, 0, 255), 20)
	# cv2.arrowedLine(color_warp, (640, 540), (640, 180), (0, 0, 255), 20)
	# cv2.arrowedLine(color_warp, (0, 540), (0, 180), (0, 0, 255), 20)

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
	if state == 'Go Ahead' or state == 'Turn Right':
		left_points = np.int_(cv2.perspectiveTransform(pts_left, m_inv))
		# left_points = list()
		# for i in left_points_raw[0]:
		# 	if i[0] >= 0 and i[1] >= 0:
		# 		left_points.append(i)
		cv2.polylines(newwarp, np.array([left_points]), False, (255, 0, 0), 10)
	# print(np.int_(cv2.perspectiveTransform(pts_right, m_inv)))
	if state == 'Go Ahead' or state == 'Turn Left':
		right_points = np.int_(cv2.perspectiveTransform(pts_right, m_inv))
		# right_points = list()
		# for i in right_points_raw[0]:
		# 	if i[0] >= 0 and i[1] >= 0:
		# 		right_points.append(i)
		cv2.polylines(newwarp, np.array([right_points]), False, (0, 0, 255), 10)
		# print(np.array([right_points]))
	# Combine the result with the original image
	# Convert arrays to 8 bit for later cv to ros image transfer
	undist = np.array(undist, dtype=np.uint8)
	newwarp = np.array(newwarp, dtype=np.uint8)
	result = cv2.addWeighted(undist, 1, newwarp, 1, 0)

	return result
