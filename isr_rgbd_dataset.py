import os
import sys
import numpy as np
import cv2
import random
import argparse



def raw_images_training_test_split(config):
	# Load raw images path
	rgb_training_raw_images_path = []
	rgb_testing_raw_images_path = []

	img_id = list(range(0,10001))
	for idx in range(len(img_id)):
		if idx % 4 != 0:  # skip every 4 frames
			continue

		rgb_img_path = config.isr_rgb_images_path + "rgb_image_" + str(idx) + ".png"

		if (idx <= 4000 or (idx >= 7684 and idx <= 9000)): 	# Training raw images
			rgb_training_raw_images_path.append(rgb_img_path)
		else:												# Testing raw images
			rgb_testing_raw_images_path.append(rgb_img_path)

	if config.raw_images_shuffle:
		random.shuffle(rgb_training_raw_images_path)
		random.shuffle(rgb_testing_raw_images_path)

	return rgb_training_raw_images_path, rgb_testing_raw_images_path


def get_object_centric_images(config, rgb_raw_images_path):
	rgb_object_images = []
	depth_object_images = []
	objec_image_label = []
	objects_per_class = np.zeros(10, dtype = int)

	for idx, rgb_raw_image_path in enumerate(rgb_raw_images_path):
		depth_raw_image_path = rgb_raw_image_path.replace("RGB", "Depth").replace("rgb", "d")
		image_label_path = rgb_raw_image_path.replace("RGB", "labels").replace(".png",".txt")

		# Load raw rgb-d pair image
		rgb_raw_image = cv2.imread(rgb_raw_image_path, cv2.IMREAD_COLOR)
		depth_raw_image = cv2.imread(depth_raw_image_path, cv2.IMREAD_UNCHANGED)

		# Load label file
		f_label_file = open(image_label_path, 'r')
		bb_labels = f_label_file.readlines()
		f_label_file.close()

		# Get object-centric RGB-D pair images
		for _, label_string in enumerate(bb_labels):
			label_string.replace('\n','')
			object_data = label_string.split(' ')
			object_label = int(object_data[0])
			objects_per_class[object_label] = objects_per_class[object_label] + 1
			objec_image_label.append(object_label)

			# Scaled info
			x_object_center_scaled = float(object_data[1])
			y_object_center_scaled = float(object_data[2])
			object_width_scaled = float(object_data[3])
			object_height_scaled = float(object_data[4])

			# Original info
			x_object_center_original = float(x_object_center_scaled * rgb_raw_image.shape[1])
			y_object_center_original = float(y_object_center_scaled * rgb_raw_image.shape[0])
			object_width_original = float(object_width_scaled * rgb_raw_image.shape[1])
			object_height_original = float(object_height_scaled * rgb_raw_image.shape[0])

			# Original Bounding Box coordinates
			object_coordinate_x1 = int(x_object_center_original - object_width_original/2)
			object_coordinate_y1 = int(y_object_center_original - object_height_original/2)
			object_coordinate_x2 = int(x_object_center_original + object_width_original/2)
			object_coordinate_y2 = int(y_object_center_original + object_height_original/2)

			object_rgb_image = rgb_raw_image[object_coordinate_y1:object_coordinate_y2, object_coordinate_x1:object_coordinate_x2].copy()
			object_depth_image = depth_raw_image[object_coordinate_y1:object_coordinate_y2, object_coordinate_x1:object_coordinate_x2].copy()

			object_rgb_image = cv2.resize(object_rgb_image, (config.object_centric_image_size, config.object_centric_image_size), fx = 1, fy = 1, interpolation = cv2.INTER_LINEAR)
			object_depth_image = cv2.resize(object_depth_image, (config.object_centric_image_size, config.object_centric_image_size), fx = 1, fy = 1, interpolation = cv2.INTER_LINEAR)

			rgb_object_images.append(object_rgb_image)
			depth_object_images.append(object_depth_image)

	rgb_object_images = np.array(rgb_object_images)
	depth_object_images = np.array(depth_object_images)

	# Print the numbers of the dataset
	print("--------------------------------------------------------------------------------------")
	print('RGB Object images: ' + str(rgb_object_images.shape)) 
	print('Depth Object images: ' + str(depth_object_images.shape))
	print('Number of Labels: ' + str(len(objec_image_label)))
	print('Images per Class: ' + str(objects_per_class))
	print("--------------------------------------------------------------------------------------\n")

	return rgb_object_images, depth_object_images, objec_image_label



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--isr_class_names", type=str,
		default=os.path.join('isr_rgbd_dataset.names'))
	parser.add_argument("--isr_rgb_images_path", type=str,
		default=os.path.join('RGB',''))
	parser.add_argument("--raw_images_shuffle", default=True)
	parser.add_argument("--object_centric_image_size", type=int, default=224)
	opt = parser.parse_args()

	rgb_training_raw_imgs_path, rgb_testing_raw_imgs_path =  raw_images_training_test_split(opt)
	get_object_centric_images(opt, rgb_training_raw_imgs_path)