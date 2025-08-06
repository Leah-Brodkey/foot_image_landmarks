#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Combined: Two or One foot ###


# Set Variables

IMAGE_PATH = r"your_image.png"  # default image when none supplied
BLUR_KERNEL = (5, 5)  # gaussian blur kernel size - how much to smooth it out to decrease noise
THRESHOLD_VALUE = 150  # brightness cutoff for converting grayscale to black-and-white (0–255)
MIN_CONTOUR_AREA = 5_000  # min area of contours considered in pixels^2
MAX_CONTOUR_AREA = 800_000  # max area of contours considered in pixels^2
MIN_ASPECT_RATIO = 1.0  # height/Width ratio to ensure foot is a shape at least as tall as it is wide
EXPECTED_TOES = 5  # how many toes should be there
BIGTOE_PCT = 0.27  # percent of total foot width dedicated to searching for the big toe
PINKY_PCT = 0.08  # percent of total foot width dedicated to searching for the pinky toe
TOE_REGION_PCT = 0.35  # limits vertical search for toes to the top (% of the foot contour)
TOP_CUTOFF_PCT = 0.30  # discards mid-toe candidates that are too low vertically (% of foot height)
TOE_SLICES = 5  # number of vertical segments used to search for middle toes
MIN_PEAK_SPACING_PCT = 0.06  # min horizontal spacing between detected toe peaks (% of foot height)
CEN_OFFSET_PCT = 0.06  # offset below toe tip for center point (proportional to foot height)
MT1_OFFSET_PCT = 0.25  # vertical offset for M1 landmark from big toe (proportional to foot height)
MT2_OFFSET_PCT = 0.25  # vertical offset for M2 landmark from pinky (proportional to foot height)
MT2_SHIFT_PCT = 0.3  # percent of horizontal foot width to shift M2 toward big toe
EDGE_MARGIN_PCT = 0.05  # % margin from left/right image edges where contours are ignored

# Imports

from pathlib import Path  # file path handling
import cv2  # image processing
import numpy as np  # numerical operations

# Thresholding Function

def threshold_image(image, blur_kernel, thresh):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # color to greyscale
    gray_blur = cv2.GaussianBlur(gray, blur_kernel, 0)  # reduce noise via blurring
    _, gry = cv2.threshold(gray_blur, thresh, 255, cv2.THRESH_BINARY)  # create black and white silhouette image
    return gry  # return binary mask

# Contour Filtering Function

def find_foot_contours(binary, min_area, max_area, min_ar):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    feet = []
    img_h, img_w = binary.shape
    x_margin = EDGE_MARGIN_PCT * img_w

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        ar = h / w if w else 0

        if not (min_area < area < max_area and ar >= min_ar):
            continue

        # Edge exclusion logic
        if x < x_margin or (x + w) > (img_w - x_margin):
            continue

        feet.append(cnt)

    feet.sort(key=cv2.contourArea, reverse=True)
    return feet


"""Find toe tips on a single foot."""
def find_toe_peaks(pts, y_min, height, *, big_left=True):
    toe_band = pts[pts[:, 1] <= y_min + TOE_REGION_PCT * height]  # cut to top band only
    if toe_band.size == 0:
        return np.empty((0, 2))  # nothing found

    x_min, x_max = toe_band[:, 0].min(), toe_band[:, 0].max()  # horizontal limits
    foot_w = x_max - x_min  # full width of foot

    # divide into zones
    if big_left:  # LEFT foot (big toe on left side)
        big_zone   = toe_band[toe_band[:, 0] <= x_min + BIGTOE_PCT * foot_w]
        pinky_zone = toe_band[toe_band[:, 0] >= x_max - PINKY_PCT * foot_w]
        mid_mask = (toe_band[:, 0] > x_min + BIGTOE_PCT * foot_w) & (toe_band[:, 0] < x_max - PINKY_PCT * foot_w)
    else:  # RIGHT foot (big toe on right side)
        big_zone   = toe_band[toe_band[:, 0] >= x_max - BIGTOE_PCT * foot_w]
        pinky_zone = toe_band[toe_band[:, 0] <= x_min + PINKY_PCT * foot_w]
        mid_mask = (toe_band[:, 0] < x_max - BIGTOE_PCT * foot_w) & (toe_band[:, 0] > x_min + PINKY_PCT * foot_w)

    peaks = []  # init list
    if big_zone.size:
        peaks.append(big_zone[np.argmin(big_zone[:, 1])])  # highest point = toe tip
    if pinky_zone.size:
        peaks.append(pinky_zone[np.argmin(pinky_zone[:, 1])])

    # mid toes
    mid_pts = toe_band[mid_mask]
    if mid_pts.size:
        slice_w = (mid_pts[:, 0].max() - mid_pts[:, 0].min()) / TOE_SLICES  # how wide each vertical strip is
        top_line = y_min + TOP_CUTOFF_PCT * height  # max allowed toe height
        cand = []  # middle toe tip candidates
        for i in range(TOE_SLICES):
            xs = mid_pts[:, 0].min() + i * slice_w
            xe = xs + slice_w
            sl = mid_pts[(mid_pts[:, 0] >= xs) & (mid_pts[:, 0] < xe)]  # vertical slice
            if sl.size:
                p = sl[np.argmin(sl[:, 1])]  # highest point in slice
                if p[1] <= top_line:
                    cand.append(p)  # valid candidate

        min_px = MIN_PEAK_SPACING_PCT * height  # how far apart toe tips must be
        for c in sorted(cand, key=lambda p: p[0]):  # go left to right
            if all(np.linalg.norm(c - p) > min_px for p in peaks):  # not too close to existing tip
                peaks.append(c)
            if len(peaks) == EXPECTED_TOES:
                break  # done

    ordered = sorted(peaks, key=lambda p: p[0])[:EXPECTED_TOES]  # left to right
    if not big_left:
        ordered = ordered[::-1]  # so toes[0] always = big toe
    return np.array(ordered)

"""Draw toe tips, toe center points, and metatarsal markers on image."""
def draw_annotations(canvas, toes, height, prefix=""):
    for i, pt in enumerate(toes):  # label toe tips
        p = tuple(pt.astype(int))
        cv2.circle(canvas, p, 4, (0, 0, 255), -1)  # red dot
        cv2.putText(canvas, f"{prefix}T{i+1}", p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if len(toes) < 3:
        return  # not enough points to do more

    off = int(CEN_OFFSET_PCT * height)  # offset center points below toe tips

    if len(toes) < 5:
        print(f"Only {len(toes)} toes found for foot '{prefix}'. Skipping center points.")
        return

    # index map
    big, mid, pinky = 0, 2, 4
    cen_big   = toes[big]   + np.array([0, off])
    cen_mid   = toes[mid]   + np.array([0, off])
    cen_pinky = toes[pinky] + np.array([0, off])
    mt1 = toes[big] + np.array([0, MT1_OFFSET_PCT * height])  # down from big toe
    foot_w = toes[pinky][0] - toes[big][0]  # horizontal foot width
    shift = -MT2_SHIFT_PCT * foot_w if foot_w > 0 else MT2_SHIFT_PCT * abs(foot_w)  # shift left or right depending on orientation
    mt2 = toes[pinky] + np.array([shift, MT2_OFFSET_PCT * height])  # down and toward center

    # log coords
    print(f"\nXY coordinates for foot '{prefix}' (pixels):")
    print(f"  Center  Big   → ({int(cen_big[0])}, {int(cen_big[1])})")
    print(f"  Center  Mid   → ({int(cen_mid[0])}, {int(cen_mid[1])})")
    print(f"  Center  Pinky → ({int(cen_pinky[0])}, {int(cen_pinky[1])})")
    print(f"  M1          → ({int(mt1[0])}, {int(mt1[1])})")
    print(f"  M2          → ({int(mt2[0])}, {int(mt2[1])})")

    for tag, pt in [("Big", cen_big), ("Mid", cen_mid), ("Pinky", cen_pinky)]:
        cv2.circle(canvas, tuple(pt.astype(int)), 3, (200, 0, 200), -1)  # purple dot
        cv2.putText(canvas, tag, tuple(pt.astype(int) + np.array([4, -4])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 200), 1)

    for tag, pt in [("M1", mt1), ("M2", mt2)]:
        cv2.circle(canvas, tuple(pt.astype(int)), 3, (255, 255, 0), -1)  # yellow dot
        cv2.putText(canvas, tag, tuple(pt.astype(int) + np.array([4, -4])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

"""Single-foot toe/landmark detection and display."""
def annotate_image(img_path=IMAGE_PATH, show_preview=False):
    import matplotlib.pyplot as plt  # for showing image

    img_path = Path(img_path)  # normalize path
    print(f"Loading {img_path}…")
    src = cv2.imread(str(img_path))  # read image
    if src is None:
        raise FileNotFoundError(img_path)

    gry = threshold_image(src, BLUR_KERNEL, THRESHOLD_VALUE)  # make binary
    feet = find_foot_contours(gry, MIN_CONTOUR_AREA, MAX_CONTOUR_AREA, MIN_ASPECT_RATIO)
    if not feet:
        print("No plausible foot found; returning mask only.")
        canvas = cv2.cvtColor(gry, cv2.COLOR_GRAY2BGR)
        if show_preview:
            plt.figure(figsize=(8, 10))
            plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            plt.title("No Foot Detected – Binary Mask")
            plt.axis("off")
            plt.show()
        return canvas


    pts = feet[0][:, 0, :]  # unwrap contour array
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    h = y_max - y_min

    peaks = find_toe_peaks(pts, y_min, h, big_left=True)  # assume left foot by default
    canvas = cv2.cvtColor(gry, cv2.COLOR_GRAY2BGR)
    draw_annotations(canvas, peaks, h)

    if show_preview:
        plt.figure(figsize=(8, 10))
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.title("Toe Tips and Center Points")
        plt.axis("off")
        plt.show()

    return canvas

"""Dual-foot toe/landmark detection and display."""
def annotate_two_feet(img_path=IMAGE_PATH, show_preview=False):
    import matplotlib.pyplot as plt

    img_path = Path(img_path)
    src = cv2.imread(str(img_path))
    if src is None:
        raise FileNotFoundError(img_path)

    gry = threshold_image(src, BLUR_KERNEL, THRESHOLD_VALUE)
    feet = find_foot_contours(gry, MIN_CONTOUR_AREA, MAX_CONTOUR_AREA, MIN_ASPECT_RATIO)
    if len(feet) < 2:
        print("Less than two feet detected; falling back to single annotate.")
        result = annotate_image(img_path, show_preview)
        if result is None:
            gry = threshold_image(src, BLUR_KERNEL, THRESHOLD_VALUE)
            canvas = cv2.cvtColor(gry, cv2.COLOR_GRAY2BGR)
            if show_preview:
                plt.figure(figsize=(8, 10))
                plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                plt.title("No Feet Detected – Binary Mask")
                plt.axis("off")
                plt.show()
            return canvas
        return result


    canvas = cv2.cvtColor(gry, cv2.COLOR_GRAY2BGR)

    foot_pts = [cnt[:, 0, :] for cnt in feet[:2]]
    centroids = [pts.mean(axis=0) for pts in foot_pts]
    left_idx, right_idx = (0, 1) if centroids[0][0] < centroids[1][0] else (1, 0)

    # LEFT side of image → anatomical RIGHT foot
    pts_l = foot_pts[left_idx]
    y_min_l, y_max_l = pts_l[:, 1].min(), pts_l[:, 1].max()
    h_l = y_max_l - y_min_l
    peaks_l = find_toe_peaks(pts_l, y_min_l, h_l, big_left=False)
    draw_annotations(canvas, peaks_l, h_l, prefix="R")

    # RIGHT side of image → anatomical LEFT foot
    pts_r = foot_pts[right_idx]
    y_min_r, y_max_r = pts_r[:, 1].min(), pts_r[:, 1].max()
    h_r = y_max_r - y_min_r
    peaks_r = find_toe_peaks(pts_r, y_min_r, h_r, big_left=True)
    draw_annotations(canvas, peaks_r, h_r, prefix="L")

    if show_preview:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.title("Two-Foot Toe Tips and Center Points")
        plt.axis("off")
        plt.show()

    return canvas


# In[ ]:




