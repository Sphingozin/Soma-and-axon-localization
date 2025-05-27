import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
import openpyxl
from tkinter import filedialog
from skimage.draw import disk
from skimage.filters import threshold_otsu, gaussian
from PIL import Image
import os
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.morphology import label, remove_small_objects, skeletonize
from skimage.measure import regionprops, label as sk_label
from matplotlib.widgets import Slider
from skimage.util import img_as_float
import matplotlib
matplotlib.use('TkAgg')

# === Load Reference and Target Images ===
root = tk.Tk()
root.withdraw()
ref_path = filedialog.askopenfilename(title="Select the REFERENCE image", filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")])
if not ref_path:
    print("No reference image selected. Exiting...")
    exit()

tgt_path = filedialog.askopenfilename(title="Select the TARGET image", filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")])
if not tgt_path:
    print("No target image selected. Exiting...")
    exit()

def load_image(path):
    img = Image.open(path)
    img = np.array(img)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    return img_as_float(img)

ref_image = load_image(ref_path)
tgt_image = load_image(tgt_path)

# === Threshold slider for reference mask ===
ref_mask_holder = {}

def update_mask(val, img, ax_mask, mask_display, ax_title, mask_holder):
    threshold = val
    binary_mask = img > threshold
    binary_mask = remove_small_objects(binary_mask, min_size=2)
    mask_holder['mask'] = binary_mask
    mask_display.set_data(binary_mask)
    ax_title.set_title(f"Cell Mask (Threshold: {threshold:.2f})")
    plt.draw()

fig, (ax_image_ref, ax_mask_ref) = plt.subplots(1, 2, figsize=(12, 6))
fig.subplots_adjust(bottom=0.15)

ax_image_ref.imshow(ref_image, cmap='gray')
ax_image_ref.set_title("Reference Image")
ax_image_ref.axis('off')
initial_threshold_ref = np.mean(ref_image)
binary_mask_ref = ref_image > initial_threshold_ref
binary_mask_ref = remove_small_objects(binary_mask_ref, min_size=2)
ref_mask_holder['mask'] = binary_mask_ref
mask_display_ref = ax_mask_ref.imshow(binary_mask_ref, cmap='gray')
ax_mask_ref.set_title(f"Cell Mask (Threshold: {initial_threshold_ref:.2f})")
ax_mask_ref.axis('off')

ax_slider_ref = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_ref = Slider(ax_slider_ref, 'Threshold (Reference)', np.min(ref_image), np.max(ref_image), valinit=initial_threshold_ref, valstep=0.01)
slider_ref.on_changed(lambda val: update_mask(val, ref_image, ax_mask_ref, mask_display_ref, ax_mask_ref, ref_mask_holder))

plt.show()

# === Soma Detection by Edge Points (Manual) ===
def define_soma_circle(image):
    print("Click on the soma edge to define it. Press 'f' to fit the circle, 'Backspace' to remove last point.")
    points = []
    markers = []

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title("Click Soma Edge → Press 'f' to Fit")

    def onclick(event):
        if event.inaxes == ax and event.button == 1 and event.xdata and event.ydata:
            x, y = event.xdata, event.ydata
            points.append((x, y))
            marker, = ax.plot(x, y, 'ro')
            markers.append(marker)
            fig.canvas.draw_idle()

    def onkey(event):
        if event.key == 'backspace':
            if points:
                points.pop()
                marker = markers.pop()
                marker.remove()
                fig.canvas.draw_idle()
        elif event.key == 'f':
            fig.canvas.flush_events()
            plt.close()

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

    if len(points) < 3:
        print("Not enough points to fit a circle.")
        return None, None

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    A = np.c_[2*x, 2*y, np.ones(len(x))]
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0], c[1]
    r = np.sqrt(c[2] + cx**2 + cy**2)
    return (cy, cx), r

soma_center, soma_radius = define_soma_circle(ref_image)
if soma_center is None:
    print("Soma center not defined.")
    exit()

# === Draw Soma and Rings ===
image_height, image_width = ref_image.shape
max_radius = np.sqrt((image_height / 2) ** 2 + (image_width / 2) ** 2)
num_rings = 15
ring_radii = np.linspace(soma_radius, max_radius, num_rings + 1)

fig, ax = plt.subplots()
ax.imshow(ref_image, cmap='gray')
circle = plt.Circle((soma_center[1], soma_center[0]), soma_radius, color='cyan', fill=False, linewidth=2)
ax.add_patch(circle)
for r in ring_radii:
    circle = plt.Circle((soma_center[1], soma_center[0]), r, color='red', fill=False, linewidth=1)
    ax.add_patch(circle)
plt.title("Soma and Concentric Rings")
plt.axis("off")
plt.show()

# === Intensity Calculation ===
def calculate_mean_intensity(image, mask):
    if np.count_nonzero(mask) == 0:
        return 0.0
    return np.mean(image[mask])

def get_ring_intensities(image, center, radii, structure_mask):
    ring_intensities = []
    for i in range(len(radii)-1):
        inner_r = radii[i]
        outer_r = radii[i + 1]
        mask_outer = np.zeros_like(image, dtype=bool)
        rr_outer, cc_outer = disk(center, outer_r, shape=image.shape)
        mask_outer[rr_outer, cc_outer] = True
        mask_inner = np.zeros_like(image, dtype=bool)
        rr_inner, cc_inner = disk(center, inner_r, shape=image.shape)
        mask_inner[rr_inner, cc_inner] = True
        ring_mask = mask_outer & ~mask_inner
        ring_mask &= structure_mask
        ring_intensities.append(calculate_mean_intensity(image, ring_mask))
    return ring_intensities

ref_mask = ref_mask_holder['mask']
ref_rings = get_ring_intensities(ref_image, soma_center, ring_radii, ref_mask)
tgt_rings = get_ring_intensities(tgt_image, soma_center, ring_radii, ref_mask)

# === Axon Detection (Target Mask via GUI) ===
print("Now select an image to define the AXON mask (typically the TARGET image)...")
axon_img_path = filedialog.askopenfilename(title="Select the image for AXON detection", filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")])
if not axon_img_path:
    print("No axon image selected. Skipping axon detection...")
    axon_ref_rings = [0]*len(ring_radii[:-1])
    axon_tgt_rings = [0]*len(ring_radii[:-1])
else:
    axon_image = load_image(axon_img_path)
    axon_mask_holder = {}

    def update_axon_mask(val):
        threshold = val
        binary_mask = axon_image > threshold
        binary_mask = remove_small_objects(binary_mask, min_size=2)
        axon_mask_holder['mask'] = binary_mask
        mask_display.set_data(binary_mask)
        ax.set_title(f"Axon Mask (Threshold: {threshold:.2f})")
        plt.draw()

    fig, (ax_img, ax) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(bottom=0.15)

    ax_img.imshow(axon_image, cmap='gray')
    ax_img.set_title("Image for Axon Mask")
    ax_img.axis('off')
    initial_threshold = np.mean(axon_image)
    init_mask = axon_image > initial_threshold
    init_mask = remove_small_objects(init_mask, min_size=2)
    axon_mask_holder['mask'] = init_mask
    mask_display = ax.imshow(init_mask, cmap='gray')
    ax.set_title(f"Axon Mask (Threshold: {initial_threshold:.2f})")
    ax.axis('off')

    ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Threshold (Axon)', np.min(axon_image), np.max(axon_image), valinit=initial_threshold, valstep=0.01)
    slider.on_changed(update_axon_mask)
    plt.show()

    axon_mask = axon_mask_holder['mask']
    axon_ref_rings = get_ring_intensities(ref_image, soma_center, ring_radii, axon_mask)
    axon_tgt_rings = get_ring_intensities(tgt_image, soma_center, ring_radii, axon_mask)

# === Save to Excel ===
def save_ring_intensities_to_excel(ref_vals, tgt_vals, axon_ref_vals, axon_tgt_vals, ring_radii, ref_path, soma_radius=0):
    inner_radii = list(ring_radii[:-1])
    outer_radii = list(ring_radii[1:])
    ring_labels = ['Ring ' + str(i+1) for i in range(len(ref_vals))]

    data = {
        'Ring': ring_labels,
        'Inner Radius': inner_radii,
        'Outer Radius': outer_radii,
        'Ref Intensity': ref_vals,
        'Target Intensity': tgt_vals,
        'Target / Reference Ratio': [t / r if r != 0 else 0 for r, t in zip(ref_vals, tgt_vals)],
        'Axon Ref Intensity': axon_ref_vals,
        'Axon Target Intensity': axon_tgt_vals,
        'Axon Target / Ref Ratio': [t / r if r != 0 else 0 for r, t in zip(axon_ref_vals, axon_tgt_vals)]
    }

    df = pd.DataFrame(data)
    output_dir = os.path.dirname(ref_path)
    filename = "ring_intensities.xlsx"
    excel_path = os.path.join(output_dir, filename)

    try:
        df.to_excel(excel_path, index=False)
        print(f"Excel file saved successfully to {excel_path}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")

save_ring_intensities_to_excel(ref_rings, tgt_rings, axon_ref_rings, axon_tgt_rings, ring_radii, ref_path, soma_radius)
