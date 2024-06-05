import cv2
import numpy as np


def draw_rectangle_with_text(img, pt1, pt2, color, thickness=2,
                             text=None, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8,
                             text_color=(255, 255, 255), text_thickness=2):

    cv2.rectangle(img, pt1, pt2, color, thickness)

    if text:
        draw_multiline_text(img, text, pt1, font_face, font_scale, text_color,
                            text_thickness, bg_color=color)


def draw_multiline_text(img, text, org, font_face=cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale=0.8, font_color=(255, 255, 255),
                        thickness=2, bg_color=(0, 165, 255)):
    if '\n' in text:
        uv_top_left = np.array([org[0], org[1]+10], dtype=float)
        for line in text.split('\n'):
            (text_width, text_height) = cv2.getTextSize(
                line, font_face, fontScale=font_scale, thickness=thickness)[0]
            uv_bottom_left_i = uv_top_left + [0, text_height]
            org = tuple(uv_bottom_left_i.astype(int))

            box_coords = (org, (org[0] + text_width + 2, org[1] - text_height - 6))
            cv2.rectangle(img, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
            cv2.putText(img, line, org, font_face, font_scale, font_color, thickness)
            uv_top_left += [0, text_height * 1.5]
    else:
        (text_width, text_height) = cv2.getTextSize(
            text, font_face, fontScale=font_scale, thickness=thickness)[0]
        cv2.rectangle(img, org, (org[0] + text_width + 2, org[1] +
                                 text_height + 10), bg_color, cv2.FILLED)
        cv2.putText(img, text, (org[0], org[1] + text_height + 3),
                    font_face, 0.8, font_color, thickness)


def select_roi(cap, resize_factor=3):
    # num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # select_frame_number = int(num_frames * 0.5)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, select_frame_number-1)
    width, height = cv2.get(cv2.CAP_PROP_FRAME_WIDTH), cv2.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ret, frame = cap.read()
    # roi = (585, 161, 854, 672) # For jack trim video
    if not ret:
        # print("No video found!")
        return
    frame = cv2.resize(frame, (width // resize_factor, height // resize_factor))  # x/= 3, y/= 3
    roi = cv2.selectROI('Select ROI', frame, False)
    # print(roi)
    roi = [resize_factor * x for x in roi]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return roi


def is_box_in_roi(box, roi):
    return is_point_in_roi(box[:2], roi) and is_point_in_roi(box[2:], roi)


def is_point_in_roi(pt, rect):
    return rect[0] < pt[0] < rect[0]+rect[2] and rect[1] < pt[1] < rect[1]+rect[3]


def moving_average(arr, window_length: int = 7):
    return list(np.convolve(arr, np.ones(window_length), 'valid') / window_length)[-1]


def is_centroid_in_roi(pt, rect):
    if len(pt) == 2:
        return rect[0] < pt[0] < rect[2] and rect[1] < pt[1] < rect[3]
    elif len(pt) == 4:
        # Find centroid
        x1, y1, x2, y2 = pt
        cX = int((x1 + x2) / 2.0)
        cY = int((y1 + y2) / 2.0)
        pt = (cX, cY)
        return rect[0] < pt[0] < rect[2] and rect[1] < pt[1] < rect[3]
    else:
        return False