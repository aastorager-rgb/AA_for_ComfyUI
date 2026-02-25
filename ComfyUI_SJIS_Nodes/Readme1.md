# ComfyUI Shift-JIS Variable AA Nodes

This custom node pack provides a top-tier pipeline for converting images into Variable Width Shift-JIS based ASCII Art (AA), widely used in Japanese internet culture.

Unlike Western tone-based ASCII art that simply replaces pixel brightness with Unicode characters on a 1:1 basis, this node perfectly mimics the human craftsman's approach (Line extraction -> Skeletonization -> Semantic part assembly) using AI and computer vision algorithms.

## Key Features

* **GPU Parallel Rendering (PyTorch Conv2D):** Replaces slow CPU loops by binding hundreds of font templates into a single tensor and scanning the entire canvas at once using `F.conv2d`. It reduces operation time from several minutes to just 2-5 seconds while keeping VRAM usage comfortably below 1GB.
* **Y-Tolerance & Soft Shift Penalty:** Characters are not strictly confined to a single row; they can move flexibly up and down (`y_tolerance`) to closely match the original line art. However, a penalty (`y_shift_penalty`) is applied the further they move from the original baseline to prevent the arrangement from becoming too scattered.
* **Pixel-Perfect ROI & Distance Blending:** Supports perfect pixel-level segmentation masks instead of rectangular bounding boxes.  Specifically, the center of the mask strongly applies the dedicated eye/pupil weights, which naturally blend (Lerp) into the surrounding hair (Base) weights towards the edges, expressing seamless details without unnatural transitions.
* **Soft Spatial Constraints:** The eye region is meticulously divided into upper/lower and left/center/right quadrants. By adopting a soft penalty approach, even if a character does not perfectly fall into its designated area, placement is allowed if the shape (Phase) matches perfectly, enabling the depiction of pupils with complex Kanji combinations like '弋歹'.

---

## Node Configuration & Parameter Guide

This pipeline operates with three nodes connected sequentially.

### 1. Shift_JIS Line Extractor

Extracts the 'lines' from the original image that will become the framework of the ASCII art.

* **method:** * `Segmentation (K-means)`: Optimized for anime/illustration images. Extracts lines by dividing color boundaries.
* `DoG`, `Canny`, `Simple`: Used for photorealistic images or images with distinct contrast.


* **line_thickness:** The thickness of the line. This value is dynamically scaled in proportion to `text_lines` (resolution), maintaining a consistent thickness even if the resolution is changed.

### 2. Shift_JIS Thinning (Skeletonize)

Shaves down the extracted thick lines into a 1-pixel thick skeleton so that ASCII characters can be placed on them. (Utilizes the Guo-Hall algorithm)

* **anti_loop_fill:** An area threshold value to pre-fill fine holes inside intersecting pixels (hair ends, ribbons, etc.) during the thinning process, preventing circular holes (Loops) from forming.

### 3. Shift_JIS Variable AA Generator (Main Rendering Engine)

Analyzes the extracted skeleton image and the original image (for tone) to assemble the optimal Shift-JIS characters as if typing on a typewriter.

#### [BASE] Parameters (General areas like hair, outlines, background)

* **y_tolerance:** The allowable number of pixels a character can move up and down to adhere to the skeleton (Recommended: 3~4).
* **y_shift_penalty:** Point deduction applied when moving away from the baseline. Determines the compromise between skeleton conformity and character alignment (Recommended: 0.1 ~ 0.5).
* **Phase / Density / Missing Penalty:** Scores deducted when the character's angle is misaligned, ink spills over, or fails to cover the skeleton completely.
* **Frequency Bonus:** Extra points given to commonly used, simple characters (-, _, /, etc.).

#### [ROI] Parameters (Focused depiction areas like eyes, mouth)

Activates when a pixel mask (Segm) is connected to the `opt_roi_mask` port or when `Enable Face/Eye ROI` is turned on. (Impact Pack's SAM + Segs to Mask combination is recommended). This area receives independent weights separate from the Base to preserve pupils that are easily lost.

* **Original Tone Match:** To restore the original size of the pupil that was lost to a thin line during the thinning process, a massive bonus is given if the dark tone of the original image matches the ink volume of the character. Increasing this value helps dense Kanji (●, 示, 歹, etc.) settle well in the pupil area.

---

## System Operation (How it works)

1. **Pre-caching:** Loads the specified `.csv` dictionary (character list) and `.ttf` font, drawing them directly on a virtual canvas. Calculates the shape, mask, angle (Phase), and trigonometric (Cos/Sin) tensors of the drawn characters, grouping them by width and caching them permanently in GPU VRAM.
2. **Dynamic Sliding Window (GPU Conv2D):**  Utilizes PyTorch's `F.conv2d` to avoid exhaustive search operations. Convolves dozens of character mask tensors with the canvas tensor at once, deriving overlap scores for the entire canvas in a fraction of a second.
3. **Weight Interpolation (Distance Transform Blending):** When a mask is provided, the exact center is calculated as 1.0, and the extreme edges as 0.0. Based on this value, it smoothly interpolates (Lerp) between `BASE` parameters and `ROI` parameters so the eye boundaries blend naturally.
4. **Semantic Placement (Score-Priority & Anti-Spam):** Attempts to place characters on the canvas starting from the location with the highest score. When a character classified as an eye region (`EYE_IDIOMS`) is placed, it distributes a soft weight (+10.0) to surrounding empty spaces to encourage specific parts based on left/right/center properties. Concurrently, if a large character over 6 pixels is used, a strong penalty (-9999.0) prevents it from appearing twice in that same eye blob, inducing diverse eye shapes.
5. **Text Rendering:** Once all placements are complete, calculates the variable width of each character, fills in exact spaces ( , 　, .) at appropriate positions, and returns the final ASCII art text string and image.

---

## Recommended Workflow (ComfyUI)

To maximize details, especially in the 'Eyes' region during ASCII art conversion, the following workflow is recommended:

1. **Load Image** -> Input original image
2. **Impact Pack Bbox Detector** (using eye recognition model `Eyes.pt`) -> Extract rectangular Bbox
3. **SAM (Segment Anything Model)** -> Create a pixel mask (Segm) that extracts only the exact shape of the eyes within the Bbox
4. **Segs to Mask** -> Convert to a binary black-and-white mask image
5. Connect **Shift_JIS Line Extractor -> Shift_JIS Thinning -> Shift_JIS Variable AA Generator**
6. Connect the binary mask to the `opt_roi_mask` port of the Generator
7. Adjust the `[ROI] Original Tone Match` value in the Generator to determine the darkness of the pupils.