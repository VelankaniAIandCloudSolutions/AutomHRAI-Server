from flask import Flask, render_template, Response
import cv2
import numpy as np
import pandas as pd
import os
import time

def read_components(excel_path):
    # Read the Excel file
    df = pd.read_excel(excel_path)
    return df

def highlight_components(image, components):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 0, 0)
    font_thickness = 1
    for index, row in components.iterrows():
        x, y, w, h = int(row['X_Point']), int(row['Y_Point']), int(row['W']), int(row['H'])
        component_name = row['Comp. No']
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, component_name, (x, y - 10), font, font_scale, font_color, font_thickness)

def save_image(folder_path, image, prefix, count):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filename = os.path.join(folder_path, f"{prefix}_{count}.jpg")
    cv2.imwrite(filename, image)
    return count + 1


app = Flask(__name__)

def generate_frames(template_path, excel_path):
    df = pd.read_excel(excel_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 0, 0)
    font_thickness = 1
    template = cv2.imread(template_path, 0)
    template_color = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    height, width = template.shape

    components = read_components(excel_path)
    highlight_components(template_color, components)

    sift = cv2.SIFT_create()
    keypoints_template, descriptors_template = sift.detectAndCompute(template, None)

    index_params = dict(algorithm = 1, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    cap = cv2.VideoCapture(0)
    defect_count, pass_count, frame_count, count = 1, 1, 1, 1
    save_this_frame = False

    init_time = time.time()  # Track the last time a frame was saved
    save_interval = 10  # Interval in seconds to save the frame

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_with_components = frame.copy()

            keypoints_frame, descriptors_frame = sift.detectAndCompute(gray_frame, None)
            matches = flann.knnMatch(descriptors_template, descriptors_frame, k=2)

            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
            if len(good_matches) > 10:
                src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                dst = cv2.perspectiveTransform(np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2), M)
                frame_with_components = cv2.polylines(frame_with_components, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

                similarity_threshold = 0.8
                for index, row in components.iterrows():
                    x, y, w, h = int(row['X_Point']), int(row['Y_Point']), int(row['W']), int(row['H'])
                    pts = np.float32([[x, y], [x, y+h], [x+w, y+h], [x+w, y]]).reshape(-1, 1, 2)
                    dst_pts = cv2.perspectiveTransform(pts, M)

                    template_patch = cv2.getRectSubPix(template_color, (w, h), (x + w/2, y + h/2))
                    mask = np.zeros_like(frame, dtype=np.uint8)
                    cv2.fillPoly(mask, [np.int32(dst_pts)], (255, 255, 255))
                    live_patch = cv2.bitwise_and(frame_with_components, mask)
                    live_patch = cv2.getRectSubPix(live_patch, (w, h), (np.mean(dst_pts[:, 0, 0]), np.mean(dst_pts[:, 0, 1])))

                    live_patch_gray = cv2.cvtColor(live_patch, cv2.COLOR_BGR2GRAY)
                    template_patch_gray = cv2.cvtColor(template_patch, cv2.COLOR_BGR2GRAY)
                    result = cv2.matchTemplate(live_patch_gray, template_patch_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)

                    color = (0, 255, 0) if max_val > similarity_threshold else (0, 0, 255)
                    frame_with_components = cv2.polylines(frame_with_components, [np.int32(dst_pts)], True, color, 2, cv2.LINE_AA)
                    component_name = row['Comp. No']
                    label_position = (int(dst_pts[0][0][0]), int(dst_pts[0][0][1]) - 10)
                    cv2.putText(frame_with_components, component_name, label_position, font, font_scale, font_color, font_thickness)


                    # key = cv2.waitKey(1) & 0xFF
                    # if key == ord('q'):
                    #     print('Quitting...........')
                    #     break
                    # if key == ord('s'):
                    #     print('Saving the frame')
                    #     save_this_frame = True

                    current_time = time.time()
                    if current_time - last_saved >= save_interval and current_time - last_saved <= save_interval:
                        iterate = 1
                        print('Saving the frame')
                        save_this_frame = True

                    if save_this_frame:
                        prefix = component_name
                        if color == (0, 255, 0):
                            folder_path = r"G:\Hardware\Solder Defects\Detects\pass"
                            df['Status'].loc[index] = 'pass'
                            pass_count = save_image(folder_path, live_patch, prefix, pass_count)
                        else:
                            folder_path = r"G:\Hardware\Solder Defects\Detects\defect"
                            df['Status'].loc[index] = 'fail'
                            defect_count = save_image(folder_path, live_patch, prefix, defect_count)
                        iterate+= 1
                        if iterate == (len(row['X_Point'])+1):
                            break

                if save_this_frame:
                    frame_count = save_image(r"G:\Hardware\Solder Defects\Detects\images", frame_with_components,  "full_frame", frame_count)
                    count = save_image(r"G:\Hardware\Solder Defects\Detects\images", template_color,  "template", count)
                    print('Image saved')
                    df.to_excel("final_result.xlsx")
                    save_this_frame = False

            # Resize both images to the smallest height among them and ensure same type
            min_height = min(template_color.shape[0], frame_with_components.shape[0])
            template_color_resized = cv2.resize(template_color, (template_color.shape[1], min_height), interpolation=cv2.INTER_AREA)
            frame_with_components_resized = cv2.resize(frame_with_components, (frame_with_components.shape[1], min_height), interpolation=cv2.INTER_AREA)

            combined_image = cv2.hconcat([template_color_resized, frame_with_components_resized])

            ret, buffer = cv2.imencode('.jpg', combined_image)
            frame = buffer.tobytes()

            # Yield the frame in a format suitable for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(r"G:\Hardware\Solder Defects\Dataset\template_images\opencv_frame_0.png", r"G:\Hardware\Solder Defects\Dataset\location.xlsx"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
