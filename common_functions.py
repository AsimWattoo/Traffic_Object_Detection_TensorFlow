import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

def upscale_points(image, coordinates, startFromZero= False):
    if startFromZero:
        x = int(coordinates[0] * image.shape[0])
        y = int(coordinates[1] * image.shape[1])
        width = int(coordinates[2] * image.shape[0])
        height = int(coordinates[3] * image.shape[1])
        return (y, x, height, width)
    else:
        x = int(coordinates[1] * image.shape[0])
        y = int(coordinates[2] * image.shape[1])
        width = int(coordinates[3] * image.shape[0])
        height = int(coordinates[4] * image.shape[1])
        return (int(coordinates[0]), y, x, height, width)

def draw_boundary(image, color, coordinates, startFromZero= False, upScale= True):
    if upScale:
        if startFromZero:
            (x, y, width, height) = upscale_points(image, coordinates, startFromZero)
        else:
            (object_class, x, y, width, height) = upscale_points(image, coordinates, startFromZero)
    else:
        if startFromZero:
            x, y, width, height = coordinates
        else:
            object_class, x, y, width, height =  coordinates
    image[x-width:x+width, y-height:y-height+5, :] = color
    image[x+width:x+width+5, y-height:y+height, :] = color
    image[x-width:x+width, y+height-5:y+height, :] = color
    image[x-width-5:x-width, y-height:y+height, :] = color
    if startFromZero:
        return image
    else:
        return image, (int(coordinates[0]), x, y, width, height)

def clip(value, prev_min, prev_max, new_min, new_max):
    return int((((value - prev_min) * (new_max - new_min)) / (prev_max - prev_min)) + new_min)

def load_image(folder, images_list, labels_list, index):
    image = plt.imread(f'{folder}/images/{images_list[index]}')
    image = cv2.resize(image, (640, 640))
    object_coordinates = []
    with open(f'{folder}/labels/{labels_list[index]}', 'r') as file:
        label = file.readlines()
    for lab in label:
        coordinates = [float(num) for num in lab.split(' ')]
        coords = upscale_points(image, coordinates)
        object_coordinates.append(coords)
    return image, object_coordinates

def generate_anchor_points(image, stride = 5):
    anchor_points = []
    for i in range(stride, image.shape[0], stride):
        for j in range(stride, image.shape[1], stride):
            anchor_points.append([i, j])
    return anchor_points

def add_anchor_points(image, points, point_size = 5): 
    new_image = image.copy()
    for point in points:
        new_image[point[0] - point_size : point[0] + point_size, point[1] - point_size : point[1] + point_size, :] = [255, 255, 0]
    return new_image

def generate_regions(anchor_points, region_ratios, region_scales):
    # Going for each anchor point
    regions = []
    for anchor in anchor_points:
        for ratio in region_ratios:
            for scale in region_scales:
                value = 1 * scale
                width = ratio[0] * value
                height = ratio[1] * value
                regions.append([anchor[0], anchor[1], width, height])
    return regions

def generate_non_overlapping_regions(image, region_ratios, region_scales):
    # Going for each anchor point
    regions = []
    for ratio in region_ratios:
        for scale in region_scales:
            value = 1 * scale
            width = ratio[0] * value
            height = ratio[1] * value
            x = width
            y = height
            row = 0
            col = 0
            has_x_reached_end = False
            has_y_reached_end = False
            while x < image.shape[0] and y < image.shape[1]:
                has_x_reached_end = False
                has_y_reached_end = False

                if col * width >= image.shape[0]:
                    col = 0
                    row += 1

                if width + col * width >= image.shape[0]:
                    x = image.shape[0] - width
                    has_x_reached_end = True

                if height + row * height >= image.shape[1]:
                    y = image.shape[1] - height
                    has_y_reached_end = True
                    
                if not has_x_reached_end and not has_y_reached_end:
                    x = width + col * 2 * width
                    y = height + row * 2 * height
                    regions.append([x, y, width, height])
                elif has_x_reached_end and has_y_reached_end:
                    regions.append([x, y, width, height])
                    break
                elif has_x_reached_end:
                    regions.append([x, y, width, height])
                else:
                    regions.append([x, y, width, height])
                col += 1

    return regions

def add_regions(image, regions, color=[0, 0, 0], thickness = 1):
    new_image = image.copy()
    # index = 0
    for region in regions:
        x = region[0]
        y = region[1]
        width = region[2]
        height = region[3]
        topleft = [0 if x - width < 0 else x - width, 0 if y - height < 0 else y - height]
        topright = [image.shape[0] if x + width > image.shape[0] else  x + width, 0 if y - height < 0 else y - height]
        bottomleft = [0 if x - width < 0 else x - width, image.shape[1] if y + height > image.shape[1] else y + height]
        bottomright = [image.shape[0] if x + width > image.shape[0] else x + width, image.shape[1] if y + height > image.shape[1] else y + height]
        # chosen_color = color[int(index / len(color))]
        # index += 1
        new_image[topleft[0] : topright[0], topleft[1]:topleft[1] + thickness, :] = color
        new_image[topright[0]:topright[0] + thickness, topright[1]:bottomright[1], :] = color
        new_image[bottomleft[0] : bottomright[0], bottomleft[1] - thickness : bottomleft[1], :] = color
        new_image[bottomleft[0] - thickness: bottomleft[0], topleft[1]:bottomleft[1], :] = color
    return new_image

def overlapArea(rect1, rect2):

   # Finding the length and width of the overlap area
   x = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
   y = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))
   area = x * y
   return area

def generate_masks(image, x, y, width, height):
    mask = np.zeros(image.shape)
    mask[x - width : x + width, y - height : y + height, :] = [1, 1, 1]
    return mask

def is_region_inside_image(image, x, y, width, height):
    if x - width < 0 or x + width > image.shape[0] or y - height < 0 or y + height > image.shape[1]:
        return False
    else:
        return True

def find_IOU(object_coords, target_region):
    a1 = (object_coords[3] * 2) * (object_coords[4] * 2)
    a2 = (target_region[2] * 2) * (target_region[3] * 2)
    rect1 = [object_coords[1] - object_coords[3], object_coords[2] - object_coords[4], object_coords[1] + object_coords[3], object_coords[2] + object_coords[4]]
    rect2 = [target_region[0] - target_region[2], target_region[1] - target_region[3], target_region[0] + target_region[2], target_region[1] + target_region[3]]
    intersection_area = overlapArea(rect1, rect2)
    total_area = a1 + a2 - intersection_area
    return intersection_area / total_area

def show_overlapping_area(region_mask, object_mask):
    main_mask = region_mask
    main_mask[main_mask == 1] = 0.5
    object_mask = object_mask
    object_mask[object_mask == 1] = 0.5
    combined_mask = main_mask + object_mask
    return combined_mask

def propose_regions(image, object_coordinates, anchor_point_stride = 20, region_ratios = [[1, 1], [2, 1], [3, 1]], region_scales = [16, 24, 32, 40], iou_limit = 0.5):

    anchor_points = generate_anchor_points(image, anchor_point_stride)
    regions = generate_regions(anchor_points, region_ratios, region_scales)
    region_targets = []
    for region in regions:
        if is_region_inside_image(image, region[0], region[1], region[2], region[3]):
            for object_coords in object_coordinates:
                iou = find_IOU(object_coords, region)
                if iou > iou_limit or (iou < 0.3 and iou > 0):
                    region_targets.append([1 if iou > iou_limit else 0, region[0], region[1], region[2], region[3]])

    # object_regions = []
    # for object_coords in object_coordinates:
    #     object_regions.append([1, object_coords[1], object_coords[2], object_coords[3], object_coords[4]])

    fine_regions = []
    non_fine_regions = []
    for i in range(0, len(region_targets)):
        if region_targets[i][0] == 1:
            fine_regions.append(region_targets[i])
        else:
            non_fine_regions.append(region_targets[i])

    if len(fine_regions) == 0:
        return []

    fine_regions = np.array(fine_regions, dtype=np.int32)
    # Deciding the non-fine regions to take
    non_fine_regions_len = 150
    if non_fine_regions_len > len(non_fine_regions):
        non_fine_regions_len = len(non_fine_regions)
    # Taking non fine regions equal to that of fine regions
    # object_regions = np.array(object_regions, dtype=np.int32)
    random_region_indices = np.random.choice(len(non_fine_regions), non_fine_regions_len)
    non_fine_regions = np.array(non_fine_regions, dtype=np.int32)
    non_fine_regions = non_fine_regions[random_region_indices]
    region_targets = np.append(fine_regions, non_fine_regions, axis=0)
    region_targets = shuffle(region_targets)
    return region_targets

def propose_train_regions(image, object_coordinates, anchor_point_stride = 20, region_ratios = [[1, 1], [2, 1], [3, 1]], region_scales = [16, 24, 32, 40], iou_limit = 0.5):
    # anchor_points = generate_anchor_points(image, anchor_point_stride)
    # regions = generate_regions(anchor_points, region_ratios, region_scales)
    # region_targets = []
    # for region in regions:
    #     if is_region_inside_image(image, region[0], region[1], region[2], region[3]):
    #         for object_coords in object_coordinates:
    #             iou = find_IOU(object_coords, region)
    #             if iou > iou_limit or (iou < 0.3 and iou > 0):
    #                 region_targets.append([1 if iou > iou_limit else 0, region[0], region[1], region[2], region[3]])

    object_regions = []
    for object_coords in object_coordinates:
        object_regions.append([object_coords[0], object_coords[1], object_coords[2], object_coords[3], object_coords[4]])

    # fine_regions = []
    # non_fine_regions = []
    # for i in range(0, len(region_targets)):
    #     if region_targets[i][0] == 1:
    #         fine_regions.append(region_targets[i])
    #     else:
    #         non_fine_regions.append(region_targets[i])

    # fine_regions = np.array(fine_regions, dtype=np.int32)
    # # Deciding the non-fine regions to take
    # non_fine_regions_len = 150
    # if non_fine_regions_len > len(non_fine_regions):
    #     non_fine_regions_len = len(non_fine_regions)
    # # Taking non fine regions equal to that of fine regions
    # # object_regions = np.array(object_regions, dtype=np.int32)
    # random_region_indices = np.random.choice(len(non_fine_regions), non_fine_regions_len)
    # non_fine_regions = np.array(non_fine_regions, dtype=np.int32)
    object_regions = np.array(object_regions, dtype=np.int32)
    # non_fine_regions = non_fine_regions[random_region_indices]
    # region_targets = np.append(object_regions, non_fine_regions, axis=0)
    return object_regions

def resize_region(image, x, y, width, height, new_size = (64, 64)):
    cropped_image = image[x - width: x + width, y - height : y + height, :]
    cropped_image = cv2.resize(cropped_image, new_size)
    return cropped_image

def detect_object_regions(image, rpn):
    region_ratios = [[1, 1], [2, 1]]
    region_scales = [64, 96, 128]
    anchor_point_stride = 30
    anchor_points = generate_anchor_points(image, anchor_point_stride)
    regions = generate_regions(anchor_points, region_ratios, region_scales)
    final_regions = []
    region_images = []
    for region in regions:
        if not is_region_inside_image(image, region[0], region[1], region[2], region[3]):
            continue

        final_regions.append(region)
        region_images.append(resize_region(image, region[0], region[1], region[2], region[3], (128, 128)))
    
    final_regions = np.array(final_regions)
    region_images = np.array(region_images)
    region_proposals = rpn.predict(region_images)
    region_cofidence = 0.7
    region_proposals = np.reshape(region_proposals, (-1,))
    target_regions = final_regions[region_proposals > region_cofidence]
    print(f'Proposed Regions: {len(target_regions)} out of {len(final_regions)}')
    target_region_images = region_images[region_proposals > region_cofidence]
    return target_regions, target_region_images

def detect_objects(image, rpn, classifier, region_ratios, region_scales):
    anchor_point_stride = 15
    anchor_points = generate_anchor_points(image, anchor_point_stride)
    regions = generate_regions(anchor_points, region_ratios, region_scales)
    print(f'Anchor Points: {len(anchor_points)} - Regions: {len(regions)}')
    final_regions = []
    region_images = []
    for region in regions:
        if not is_region_inside_image(image, region[0], region[1], region[2], region[3]):
            continue

        final_regions.append(region)
        region_images.append(resize_region(image, region[0], region[1], region[2], region[3], (128, 128)))
    
    final_regions = np.array(final_regions)
    region_images = np.array(region_images)
    region_proposals = rpn.predict(region_images)
    region_cofidence = 0.7
    region_proposals = np.reshape(region_proposals, (-1,))
    target_regions = final_regions[region_proposals > region_cofidence]
    print(f'Proposed Regions: {len(target_regions)} out of {len(final_regions)}')
    target_region_images = region_images[region_proposals > region_cofidence]
    predicted_classes = classifier.predict(target_region_images)
    proposed_regions = []
    predicted_classes = np.argmax(predicted_classes, axis= 1)
    proposed_region_classes = []
    # Performing non-maximum suppression
    for i in range(len(target_regions)):
        has_overlapping_region = False
        for j in range(i + 1, len(target_regions)):
            r1 = target_regions[i]
            r2 = target_regions[j][:]
            r2 = np.insert(r2, 0, 0)
            iou = find_IOU(r2, r1)
            if iou > 0.2:
                has_overlapping_region = True
                break
        if not has_overlapping_region:
            proposed_regions.append(r1)
            proposed_region_classes.append(predicted_classes[i])
    print(f'Final Regions: {len(proposed_regions)}')
    return proposed_regions, proposed_region_classes