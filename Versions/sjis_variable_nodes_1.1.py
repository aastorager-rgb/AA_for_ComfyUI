import os
import numpy as np
import torch
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from numpy.lib.stride_tricks import sliding_window_view
import comfy.utils

# ==========================================
# 0. Global Typography Settings & Helpers
# ==========================================
FONT_SIZE = 16
LINE_SPACING = 2
ROW_HEIGHT = FONT_SIZE + LINE_SPACING

def get_resource_files(extension):
    node_dir = os.path.dirname(os.path.abspath(__file__))
    files = []
    for root, _, filenames in os.walk(node_dir):
        for f in filenames:
            if f.endswith(extension):
                rel_path = os.path.relpath(os.path.join(root, f), node_dir).replace("\\", "/")
                files.append(rel_path)
    if not files: files.append(f"example{extension}")
    return sorted(files)

CSV_FILES = get_resource_files(".csv")
TTF_FILES = get_resource_files(".ttf")
TXT_FILES = get_resource_files(".txt") 

def get_char_width(char, font, draw_ctx):
    if hasattr(draw_ctx, 'textlength'): return int(draw_ctx.textlength(char, font=font))
    else: return int(draw_ctx.textbbox((0,0), char, font=font)[2])

def thinning_guohall_numpy(image):
    img = image.copy()
    try: return cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    except:
        skel = np.zeros(img.shape, np.uint8); element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(img, element); temp = cv2.subtract(img, cv2.dilate(eroded, element))
            skel = cv2.bitwise_or(skel, temp); img = eroded.copy()
            if cv2.countNonZero(img) == 0: break
        return skel

def calculate_orientation_map(img):
    img_f = img.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(img_f, (3, 3), 0.7)
    gx = cv2.Scharr(blurred, cv2.CV_32F, 1, 0); gy = cv2.Scharr(blurred, cv2.CV_32F, 0, 1)
    vx = cv2.boxFilter(gx**2 - gy**2, -1, (5, 5)); vy = cv2.boxFilter(2 * gx * gy, -1, (5, 5))
    theta = 0.5 * np.arctan2(vy, vx)
    theta[vx >= 0] += (np.pi / 2)
    return np.cos(2 * theta).astype(np.float32), np.sin(2 * theta).astype(np.float32), theta

# ==========================================
# 1. Semantic Categories
# ==========================================
DENSE_KANJI = set(['瀟', '憎', '浄', '占', '李', '斗', '狄', '灘', '濾', '鼎', '撼'])

EYE_UP_LEFT = set(list("だ灯衍行仍了乍仡乞云伝芸茫忙它佗俐仗なてｨｪｵｴﾃﾇﾏﾓfr"))
EYE_UP_CENTER = set(list("不示宍亦兀亢万迩尓禾乏弌弍弐泛夾赱符≡女乍气旡まみてテチｪｫｭｮｰｴｵｻﾁﾆﾓﾕﾖェュョヵ"))
EYE_UP_RIGHT = set(list("豺犾狄勿下卞抃圷圦坏心沁气汽斥拆仔竹刃刈付以雫爿なうか刈ﾊ､ｧｩｪｫｬｭｱｦｳｵｷｹﾁﾃﾇﾕﾖェュヵヶァ"))
EYE_LOW_LEFT = set(list("芍弋爪心父戈弌弍弐式汽辷込乂癶廴匕丈叱杙之比仆トヽヾゝゞﾞｬｾﾀﾊﾋﾏﾓ㌧､t｀ﾞ~"))
EYE_LOW_RIGHT = set(list("歹万久刋升刈乃汐沙少炒梦斗孑才必瓜欠次亥圦乂ノソルツ八ｧｨｩｫｱｦｳｶｸｹｻｼｽｾﾀﾁﾂﾃﾅﾇﾈﾉﾊﾑﾒﾔﾗﾘﾙﾚﾜﾝァヵ㌧㌢㌃㍗㌣㌻j'´~"))
EYE_IDIOMS_ALL = EYE_UP_LEFT | EYE_UP_CENTER | EYE_UP_RIGHT | EYE_LOW_LEFT | EYE_LOW_RIGHT

# ==========================================
# 2. Nodes Implementation (Extractor & Thinning)
# ==========================================
class SjisLineExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "text_lines": ("INT", {"default": 30, "min": 10, "max": 100}),
            "method": (["Segmentation (K-means)", "DoG (Soft Lines)", "Canny (Hard Edges)", "Simple (Grayscale)", "None"],),
            "threshold": ("INT", {"default": 127, "min": 0, "max": 255}),
            "segmentation_k": ("INT", {"default": 3, "min": 2, "max": 32}),
            "line_thickness": ("FLOAT", {"default": 2.5, "min": 0.1, "max": 10.0, "step": 0.1}),
            "invert_output": ("BOOLEAN", {"default": True}),
        }}
    RETURN_TYPES = ("IMAGE",); RETURN_NAMES = ("line_image",); FUNCTION = "extract_lines"; CATEGORY = "image/ascii"

    def extract_lines(self, image, text_lines, method, threshold, segmentation_k, line_thickness, invert_output):
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img).convert("RGB")
        gray = np.array(pil_img.convert('L'))
        target_h = text_lines * ROW_HEIGHT; target_w = int(target_h * (gray.shape[1] / gray.shape[0]))
        
        if method == "Segmentation (K-means)":
            img_res = cv2.resize(np.array(pil_img), (target_w, target_h), interpolation=cv2.INTER_AREA)
            Z = img_res.reshape((-1, 3)).astype(np.float32)
            _, labels, _ = cv2.kmeans(Z, segmentation_k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
            labels_2d = labels.reshape((target_h, target_w))
            edges = (labels_2d != np.roll(labels_2d, 1, axis=0)) | (labels_2d != np.roll(labels_2d, 1, axis=1))
            binary = edges.astype(np.uint8) * 255; binary[0, :] = 0; binary[:, 0] = 0
        else:
            gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
            if method == "DoG (Soft Lines)":
                img_blur = cv2.GaussianBlur(255-gray, (0, 0), sigmaX=max(0.5, line_thickness * 0.5))
                lines = cv2.divide(255-gray, 255-img_blur, scale=256)
                _, binary = cv2.threshold(lines, threshold, 255, cv2.THRESH_BINARY_INV); binary = 255 - binary
            elif method == "Canny (Hard Edges)": binary = cv2.Canny(gray, threshold // 2, threshold)
            elif method == "Simple (Grayscale)": _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            else: binary = 255 - gray
        
        scale = target_h / (40.0 * ROW_HEIGHT); act_t = max(1, int(round(line_thickness * scale)))
        if method != "None" and act_t > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (act_t, act_t))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            if act_t // 2 > 1: binary = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (act_t//2, act_t//2)), iterations=1)
            
        out = binary if invert_output else 255 - binary
        return (torch.from_numpy(np.array(Image.fromarray(out).convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0),)

class SjisThinning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "line_image": ("IMAGE",),
            "threshold": ("INT", {"default": 127, "min": 0, "max": 255}),
            "clean_strength": ("INT", {"default": 1, "min": 0, "max": 5}),
            "anti_loop_fill": ("INT", {"default": 50, "min": 0, "max": 500}),
        }}
    RETURN_TYPES = ("IMAGE",); RETURN_NAMES = ("thinned_image",); FUNCTION = "process_thinning"; CATEGORY = "image/ascii"

    def process_thinning(self, line_image, threshold, clean_strength, anti_loop_fill):
        img = (line_image[0].cpu().numpy() * 255).astype(np.uint8)
        img_arr = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if np.mean(img_arr) > 127: img_arr = 255 - img_arr
        _, binary = cv2.threshold(img_arr, threshold, 255, cv2.THRESH_BINARY)
        if anti_loop_fill > 0:
            cnts, hi = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hi is not None:
                for i, c in enumerate(cnts):
                    if hi[0][i][3] != -1 and cv2.contourArea(c) <= anti_loop_fill: cv2.drawContours(binary, [c], 0, 255, -1)
        if clean_strength > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_strength*2+1, clean_strength*2+1))
            binary = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
        skeleton = thinning_guohall_numpy(binary)
        return (torch.from_numpy(np.array(Image.fromarray(255 - skeleton).convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0),)

# ==========================================
# 4. Node 3: Hybrid Generator
# ==========================================

class SjisVariableWidthGenerator:
    _char_list_cache = None
    _char_data_cache = [] 
    _char_groups_cache = None
    _tone_chars_cache = None
    _device = None
    _current_font_path = None
    _current_csv_path = None
    _current_txt_path = None
    
    _fw_width = 16
    _hw_width = 8
    _dot_width = 4
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original_image": ("IMAGE",), 
            "thinned_image": ("IMAGE",),
            "char_list_path": (CSV_FILES,), 
            "font_path": (TTF_FILES,),      
            "text_lines": ("INT", {"default": 30, "min": 10, "max": 100}),
            "global_y_shift": ("INT", {"default": 0, "min": -16, "max": 16, "step": 1, "label": "▶ Image Y Shift (px)"}),  
            "letter_spacing": ("FLOAT", {"default": 0.0, "min": -2.9, "max": 10.0, "step": 0.1}),
            "placement_method": (["Score-Priority", "Sequential"],),          
            "y_tolerance": ("INT", {"default": 2, "min": 0, "max": 6}),
            "y_shift_penalty": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 10.0, "step": 0.05}),
            "phase_congruency_weight": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 5.0, "step": 0.1}),
            "density_penalty_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01}),
            "missing_penalty_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01}),
            "frequency_weight": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            "semantic_roi_bypass": ("BOOLEAN", {"default": True}),
            "roi_phase_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "roi_density_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "roi_missing_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "roi_frequency_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "roi_tone_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            
            "bg_tone_mode": (["1: Full Area Tone", "2: Fill Empty Spaces", "3: Line-art Only"], {"default": "3: Line-art Only"}),
            "bg_tone_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "bg_tone_contrast": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
            "bg_tone_brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.02}),
            "char_tone_path": (TXT_FILES, {"default": "char_tone.txt"}),
        },
        "optional": {
            "opt_roi_mask": ("MASK",),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("aa_image", "aa_text", "grid_image")
    FUNCTION = "generate_hybrid_ascii"
    CATEGORY = "image/ascii"

    def load_resources(self, char_list_path, font_path, char_tone_path):
        node_dir = os.path.dirname(os.path.abspath(__file__))
        full_csv = os.path.join(node_dir, char_list_path)
        full_font = os.path.join(node_dir, font_path)
        full_txt = os.path.join(node_dir, char_tone_path)
        
        try: font = ImageFont.truetype(full_font, FONT_SIZE)
        except: font = ImageFont.load_default()
        
        dummy_draw = ImageDraw.Draw(Image.new("L", (1,1)))
        self._fw_width = max(1, get_char_width('　', font, dummy_draw))
        self._hw_width = max(1, get_char_width(' ', font, dummy_draw))
        self._dot_width = max(1, get_char_width('.', font, dummy_draw))
        
        if SjisVariableWidthGenerator._current_txt_path != full_txt:
            tone_list = []
            max_t = 0.01
            if os.path.exists(full_txt):
                with open(full_txt, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.replace('\n', '').replace('\r', '') 
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            c = parts[0]
                            if c == "Chars" or "source:" in c: continue
                            try:
                                t = float(parts[1])
                                cw = get_char_width(c, font, dummy_draw)
                                if cw > 0:
                                    tone_list.append((t, c, cw))
                                    if t > max_t: max_t = t
                            except: pass
            SjisVariableWidthGenerator._tone_chars_cache = [(t/max_t, c, cw) for t, c, cw in tone_list]
            SjisVariableWidthGenerator._current_txt_path = full_txt
            
        if (self._char_list_cache is None or self._current_font_path != full_font or self._current_csv_path != full_csv): 
            chars = []
            freq_scores = []
            if os.path.exists(full_csv):
                lines = []
                try: 
                    with open(full_csv, 'r', encoding='cp932') as f:
                        lines = f.readlines()
                except: 
                    with open(full_csv, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                
                raw_chars = []
                raw_freqs = []
                if lines:
                    for line in lines[1:]:
                        line = line.rstrip('\n\r')
                        if not line: continue
                        parts = line.split(',')
                        
                        if len(parts) >= 3:
                            try:
                                freq = float(parts[-1])
                                char_str = ",".join(parts[1:-1])
                                if len(char_str) >= 2 and char_str.startswith('"') and char_str.endswith('"'):
                                    char_str = char_str[1:-1]
                                raw_chars.append(char_str)
                                raw_freqs.append(freq)
                            except: pass
                        elif len(parts) == 2:
                            try:
                                freq = float(parts[1])
                                char_str = parts[0]
                                if len(char_str) >= 2 and char_str.startswith('"') and char_str.endswith('"'):
                                    char_str = char_str[1:-1]
                                raw_chars.append(char_str)
                                raw_freqs.append(freq)
                            except:
                                raw_chars.append(parts[0])
                                raw_freqs.append(1.0)
                        elif len(parts) == 1:
                            raw_chars.append(parts[0])
                            raw_freqs.append(1.0)
                            
                if raw_chars:
                    chars = [c if c and c != 'nan' else ' ' for c in raw_chars]
                    max_f = max(raw_freqs) if raw_freqs else 1.0
                    freq_scores = [np.log1p(f) / np.log1p(max_f) for f in raw_freqs]
            
            if not chars: return None
            
            SjisVariableWidthGenerator._char_list_cache = chars
            SjisVariableWidthGenerator._current_font_path = full_font
            SjisVariableWidthGenerator._current_csv_path = full_csv
            SjisVariableWidthGenerator._char_data_cache = []            
            SjisVariableWidthGenerator._char_groups_cache = None 
            
            for i, c in enumerate(chars):
                w = get_char_width(c, font, dummy_draw)
                spatial_flags = 0
                if c in EYE_IDIOMS_ALL: spatial_flags |= 1
                if c in EYE_UP_LEFT or c in EYE_LOW_LEFT: spatial_flags |= 2
                if c in EYE_UP_CENTER: spatial_flags |= 4
                if c in EYE_UP_RIGHT or c in EYE_LOW_RIGHT: spatial_flags |= 8
                if c in EYE_UP_LEFT or c in EYE_UP_CENTER or c in EYE_UP_RIGHT: spatial_flags |= 16
                if c in EYE_LOW_LEFT or c in EYE_LOW_RIGHT: spatial_flags |= 32
                
                if c.strip() and w > 0:
                    img_char = Image.new("L", (w, ROW_HEIGHT), 0)
                    ImageDraw.Draw(img_char).text((0, 0), c, font=font, fill=255)
                    char_arr = np.array(img_char).astype(np.uint8)
                    c_cos, c_sin, c_theta = calculate_orientation_map(char_arr)
                    c_mask = (char_arr > 0).astype(np.float32)
                    SjisVariableWidthGenerator._char_data_cache.append({
                        'char': c, 'mask': c_mask, 'cos_strict': c_cos * c_mask, 'sin_strict': c_sin * c_mask,
                        'width': w, 'freq_score': freq_scores[i], 'ink': np.sum(c_mask), 'spatial_flags': spatial_flags
                    })
                else: 
                    SjisVariableWidthGenerator._char_data_cache.append({'char': c, 'mask': None, 'width': max(1, w), 'freq_score': freq_scores[i], 'ink': 0, 'spatial_flags': spatial_flags})

        if getattr(SjisVariableWidthGenerator, '_char_groups_cache', None) is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            char_groups = {}
            for idx, data in enumerate(SjisVariableWidthGenerator._char_data_cache):
                cw = data['width']
                if data['mask'] is None: continue
                if cw not in char_groups:
                    char_groups[cw] = {'indices': [], 'masks': [], 'cos_stricts': [], 'sin_stricts': [], 'inks': [], 'freqs': [], 'flags': [], 'is_dense': []}
                char_groups[cw]['indices'].append(idx)
                char_groups[cw]['masks'].append(torch.from_numpy(data['mask']).unsqueeze(0))
                char_groups[cw]['cos_stricts'].append(torch.from_numpy(data['cos_strict']).unsqueeze(0))
                char_groups[cw]['sin_stricts'].append(torch.from_numpy(data['sin_strict']).unsqueeze(0))
                char_groups[cw]['inks'].append(data['ink'])
                char_groups[cw]['freqs'].append(data['freq_score'])
                char_groups[cw]['flags'].append(data['spatial_flags'])
                char_groups[cw]['is_dense'].append(data['char'] in DENSE_KANJI)
            
            for cw in char_groups:
                for k in ['masks', 'cos_stricts', 'sin_stricts']:
                    char_groups[cw][k] = torch.stack(char_groups[cw][k]).to(device).to(torch.float32)
                for k in ['inks', 'freqs']:
                    char_groups[cw][k] = torch.tensor(char_groups[cw][k], dtype=torch.float32, device=device)
                char_groups[cw]['flags'] = torch.tensor(char_groups[cw]['flags'], dtype=torch.int32, device=device)
                char_groups[cw]['is_dense'] = torch.tensor(char_groups[cw]['is_dense'], dtype=torch.bool, device=device)
            SjisVariableWidthGenerator._char_groups_cache = char_groups
            SjisVariableWidthGenerator._device = device
            
        return SjisVariableWidthGenerator._char_data_cache, font

    def _get_gap_string(self, width, is_l):
        if width <= 0: return ""
        res = "　" * (int(round(width)) // self._fw_width)
        rem = int(round(width)) % self._fw_width
        while rem > 0:
            if rem >= self._hw_width and not (is_l and len(res)==0) and not (len(res)>0 and res[-1]==" "):
                res += " "
                rem -= self._hw_width
            elif rem >= self._dot_width:
                res += "."
                rem -= self._dot_width
            else:
                break
        return res

    def _solve_stripe_sequential(self, scores, char_data, w, spacing, last_ink_x, is_roi_mask):
        cur_x = 0.0
        line = ""
        placements = []
        is_l = True
        while int(round(cur_x)) <= last_ink_x and int(round(cur_x)) < w:
            sx = int(round(cur_x))
            best_idx = np.argmax(scores[sx, :])
            if scores[sx, best_idx] <= 0.0:
                gw = min(self._fw_width, w - sx)
                gs = self._get_gap_string(gw, is_l)
                if not gs: 
                    cur_x += max(1.0, float(gw))
                    continue
                line += gs
                for gc in gs:
                    gcw = self._fw_width if gc == '　' else (self._hw_width if gc == ' ' else self._dot_width)
                    ir = np.any(is_roi_mask[int(cur_x):min(w, int(cur_x+gcw))])
                    placements.append((gc, float(cur_x), gcw, ir))
                    cur_x += gcw
                is_l = False
                continue
            char_info = char_data[best_idx]
            cw = char_info['width']
            line += char_info['char']
            ir = np.any(is_roi_mask[sx:min(w, sx+cw)])
            placements.append((char_info['char'], float(cur_x), cw, ir))
            cur_x += max(1.0, float(cw) + spacing)
            is_l = False
        return line, placements

    def _solve_stripe_score_priority(self, score_matrix, char_data_list, w, spacing, last_ink_x, is_roi_mask):
        placements = []
        scores = score_matrix.copy()
        if last_ink_x < w:
            scores[last_ink_x:, :] = -99999.0
            
        l_i = [i for i, d in enumerate(char_data_list) if (d['spatial_flags'] & 2) > 0]
        c_i = [i for i, d in enumerate(char_data_list) if (d['spatial_flags'] & 4) > 0]
        r_i = [i for i, d in enumerate(char_data_list) if (d['spatial_flags'] & 8) > 0]
        
        max_it = w * 2
        it = 0
        while it < max_it:
            it += 1
            bx, bc = np.unravel_index(np.argmax(scores), scores.shape)
            if scores[bx, bc] <= 0.0:
                break
                
            char_info = char_data_list[bc]
            cw, c, cf = char_info['width'], char_info['char'], char_info['spatial_flags']
            ir = np.any(is_roi_mask[bx:min(w, bx+cw)]) and (cf & 1) > 0
            placements.append((c, float(bx), cw, ir))
            
            s_margin = 2 if (ir and (cf & 4) > 0) else 0
            eb = min(w, int(bx + cw + spacing + s_margin))
            sb = max(0, int(bx - spacing - s_margin))
            
            for i, d in enumerate(char_data_list):
                scores[max(0, sb - d['width'] + 1):eb, i] = -99999.0
                
            if ir:
                bs, be = bx, min(w-1, bx+cw-1)
                while bs > 0 and is_roi_mask[bs-1]: bs -= 1
                while be < w-1 and is_roi_mask[be+1]: be += 1
                
                if c in DENSE_KANJI:
                    scores[:, bc] = -100.0
                if cw > 6:
                    scores[bs:be+1, bc] = -9999.0
                
                bw = 10.0
                if (cf & 2) > 0:
                    target_range = scores[eb:be+1, c_i + r_i]
                    if target_range.size > 0: scores[eb:be+1, c_i + r_i] += bw
                elif (cf & 8) > 0:
                    target_range = scores[bs:sb, l_i + c_i]
                    if target_range.size > 0: scores[bs:sb, l_i + c_i] += bw
                elif (cf & 4) > 0:
                    l_range = scores[bs:sb, l_i]
                    if l_range.size > 0: scores[bs:sb, l_i] += bw
                    r_range = scores[eb:be+1, r_i]
                    if r_range.size > 0: scores[eb:be+1, r_i] += bw

        placements.sort(key=lambda x: x[1])
        line = ""
        cx = 0.0
        isl = True
        ap = []
        for c, x, cw, ir in placements:
            tx = max(cx, float(x))
            gp = tx - cx
            if gp >= self._dot_width:
                gs = self._get_gap_string(gp, isl)
                line += gs
                for gc in gs:
                    gcw = self._fw_width if gc == '　' else (self._hw_width if gc == ' ' else self._dot_width)
                    gc_ir = np.any(is_roi_mask[int(cx):min(w, int(cx+gcw))])
                    ap.append((gc, cx, gcw, gc_ir))
                    cx += gcw
            ap.append((c, cx, cw, ir))
            line += c
            cx += max(1.0, float(cw) + spacing)
            isl = False
            
        if cx < w - self._dot_width:
            gs = self._get_gap_string(w - cx, isl)
            line += gs
            for gc in gs:
                gcw = self._fw_width if gc == '　' else (self._hw_width if gc == ' ' else self._dot_width)
                gc_ir = np.any(is_roi_mask[int(cx):min(w, int(cx+gcw))])
                ap.append((gc, cx, gcw, gc_ir))
                cx += gcw
                
        return line, ap

    def solve_stripe_hybrid(self, row_img_bin, row_tone, row_cos, row_sin, row_roi, row_roi_weights, char_data_list, spacing, w_p, w_d, w_m, w_f, r_w_p, r_w_d, r_w_m, r_w_f, r_t_w, p_method, y_t, y_s_p, bg_mode, bg_weight):
        device = SjisVariableWidthGenerator._device
        groups = SjisVariableWidthGenerator._char_groups_cache
        H, W = row_img_bin.shape
        h = ROW_HEIGHT
        scores = np.full((W, len(char_data_list)), -99999.0, dtype=np.float32)
        
        ink_p = np.where(row_img_bin > 0)[1]
        force_full_width = (bg_mode.startswith("1") or bg_mode.startswith("2")) and bg_weight > 0
        last_x = W if force_full_width else (int(ink_p.max() + 5) if ink_p.size > 0 else 0)
        
        tone_map = (255.0 - row_tone.astype(np.float32)) / 255.0
        cy_s, cy_e = max(0, y_t), min(H, y_t + h)
        stripe_roi = np.bitwise_or.reduce(row_roi[cy_s:cy_e, :], axis=0)
        is_roi_mask = (stripe_roi & 1) > 0 
        
        if cv2.countNonZero(row_img_bin) == 0 and not np.any(is_roi_mask) and not force_full_width:
            return self._solve_stripe_sequential(scores, char_data_list, W, spacing, 0, is_roi_mask)

        with torch.no_grad():
            t_str = torch.from_numpy(row_img_bin).unsqueeze(0).unsqueeze(0).to(device)
            t_cos = torch.from_numpy(row_cos * row_img_bin).unsqueeze(0).unsqueeze(0).to(device)
            t_sin = torch.from_numpy(row_sin * row_img_bin).unsqueeze(0).unsqueeze(0).to(device)
            t_tne = torch.from_numpy(tone_map).unsqueeze(0).unsqueeze(0).to(device)
            t_roi = torch.from_numpy(row_roi).unsqueeze(0).unsqueeze(0).to(torch.int32).to(device)
            t_wgt = torch.from_numpy(row_roi_weights).unsqueeze(0).unsqueeze(0).to(device)
            
            Y_out = H - h + 1
            if Y_out <= 0: return self._solve_stripe_sequential(scores, char_data_list, W, spacing, 0, is_roi_mask)
            
            y_indices = torch.arange(Y_out, device=device).view(Y_out, 1)
            y_pen = (torch.abs(y_indices - y_t) * y_s_p).unsqueeze(0)
            
            for cw, g in groups.items():
                N = len(g['indices'])
                ones_k = torch.ones((1, 1, h, cw), dtype=torch.float32, device=device)
                
                ov2 = torch.nn.functional.conv2d(t_str, g['masks']).squeeze(0)
                t_ov2 = torch.nn.functional.conv2d(t_tne, g['masks']).squeeze(0)
                m_cos = torch.nn.functional.conv2d(t_cos, g['cos_stricts']).squeeze(0)
                m_sin = torch.nn.functional.conv2d(t_sin, g['sin_stricts']).squeeze(0)
                
                pha2 = (ov2 - (m_cos + m_sin)) * 0.5
                t_ink2 = torch.nn.functional.conv2d(t_str, ones_k).squeeze(0)
                
                b_list = [(torch.nn.functional.conv2d((t_roi & (1<<k)).float(), ones_k).squeeze(0) > 0) for k in range(6)]
                b1, b2, b4, b8, b16, b32 = b_list[0], b_list[1], b_list[2], b_list[3], b_list[4], b_list[5]
                
                blob_centrality = torch.nn.functional.conv2d(t_wgt, ones_k).squeeze(0) / (h * cw)
                
                c_req = g['flags'] & ~1
                m2 = ~((c_req & 2).bool().view(N,1,1)) | b2
                m4 = ~((c_req & 4).bool().view(N,1,1)) | b4
                m8 = ~((c_req & 8).bool().view(N,1,1)) | b8
                m16 = ~((c_req & 16).bool().view(N,1,1)) | b16
                m32 = ~((c_req & 32).bool().view(N,1,1)) | b32
                spatial_match = m2 & m4 & m8 & m16 & m32
                
                exc, mis = torch.relu(g['inks'].view(N, 1, 1) - ov2), torch.relu(t_ink2 - ov2)
                
                lerp_w = lambda r, b: b + (r - b) * blob_centrality
                cur_w_den = lerp_w(r_w_d, w_d)
                cur_w_mis = lerp_w(r_w_m, w_m)
                cur_w_pha = lerp_w(r_w_p, w_p)
                cur_w_frq = lerp_w(r_w_f, w_f)
                
                calc = ov2 - pha2*cur_w_pha - exc*cur_w_den - mis*cur_w_mis + g['freqs'].view(N,1,1)*cur_w_frq - y_pen
                
                is_eye, is_ctr = (g['flags'] & 1) > 0, (g['flags'] & 4) > 0
                val_s = b1 & spatial_match & ((ov2 > 0) | (is_ctr.view(N, 1, 1).bool() & (t_ov2 > 1.0)))
                t_bon = torch.where(is_ctr.view(N, 1, 1).bool(), t_ov2 * r_t_w, torch.zeros_like(t_ov2))
                
                calc = torch.where((is_eye.view(N, 1, 1).bool() & val_s.bool()).bool(), calc + 99999.0 + t_bon, calc)
                calc = torch.where((is_eye.view(N, 1, 1).bool() & ~val_s.bool() & b1.bool()).bool(), calc - 99999.0, calc)
                calc = torch.where(g['is_dense'].view(N, 1, 1).bool(), torch.where(is_eye.view(N,1,1).bool() ^ b1.bool(), calc - 99999.0, calc - 70.0), calc)
                if cw >= 7:
                    if cw >= 13:
                        calc = torch.where(b1.bool(), calc + 15.0, calc)
                    else:
                        calc = torch.where(b1.bool(), calc + 10.0, calc)                        
                
                if bg_mode.startswith("1") and bg_weight > 0:
                    is_valid_bg = (~b1.bool()) & (t_ov2 > 2.0)
                    calc = torch.where(is_valid_bg, calc + (t_ov2 * bg_weight), calc)
                    strict_mask = ((ov2 > 0) | (is_eye.view(N,1,1).bool() & val_s.bool()) | is_valid_bg).bool()
                    calc = torch.where(strict_mask, calc, torch.full_like(calc, -99999.0))
                else:
                    calc = torch.where(((ov2 > 0) | (is_eye.view(N,1,1).bool() & val_s.bool())).bool(), calc, torch.full_like(calc, -99999.0))
                
                b_np = torch.max(calc, dim=1)[0].cpu().numpy()
                for i, idx in enumerate(g['indices']):
                    vx = np.where(b_np[i] > -99990.0)[0]
                    if vx.size > 0: scores[vx, idx] = b_np[i][vx]
                    
        return (self._solve_stripe_score_priority if p_method == "Score-Priority" else self._solve_stripe_sequential)(scores, char_data_list, W, spacing, last_x, is_roi_mask)

    def generate_hybrid_ascii(self, original_image, thinned_image, char_list_path, font_path, text_lines, letter_spacing, placement_method, global_y_shift, y_tolerance, y_shift_penalty, phase_congruency_weight, density_penalty_weight, missing_penalty_weight, frequency_weight, semantic_roi_bypass, roi_phase_weight, roi_density_weight, roi_missing_weight, roi_frequency_weight, roi_tone_weight, bg_tone_mode, bg_tone_weight, bg_tone_contrast, bg_tone_brightness, char_tone_path, opt_roi_mask=None):
        print("[Shift_JIS AA] Progress 1/5: Reset to initial state and loading resources...")
        res = self.load_resources(char_list_path, font_path, char_tone_path)
        if not res: return (torch.zeros((1,64,64,3)), "Error", torch.zeros((1,64,64,3)))
        char_data_list, font = res
        
        # 1. 선화(Line-art) 처리 및 시프트
        img_arr = (thinned_image[0].cpu().numpy() * 255).astype(np.uint8)
        img_bin = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        if np.mean(img_bin) > 127: img_bin = 255 - img_bin 
            
        target_h = text_lines * ROW_HEIGHT
        target_w = int(target_h * (img_bin.shape[1]/img_bin.shape[0]))
        img_bin = thinning_guohall_numpy(cv2.resize(img_bin, (target_w, target_h), interpolation=cv2.INTER_NEAREST))
        
        if global_y_shift != 0:
            shifted_bin = np.zeros_like(img_bin) # 검은색(0, 잉크 없음) 패딩
            if global_y_shift > 0: shifted_bin[global_y_shift:, :] = img_bin[:-global_y_shift, :]
            else: shifted_bin[:global_y_shift, :] = img_bin[-global_y_shift:, :]
            img_bin = shifted_bin
            
        img_bin_f32 = (img_bin > 0).astype(np.float32)
        
        # 2. 원본 이미지(Tone) 처리 및 시프트
        ori_arr = (original_image[0].cpu().numpy() * 255).astype(np.uint8)
        img_tone_raw = cv2.resize(cv2.cvtColor(ori_arr, cv2.COLOR_RGB2GRAY), (target_w, target_h), interpolation=cv2.INTER_AREA)
        img_tone_f = img_tone_raw.astype(np.float32)
        img_tone_f = (img_tone_f - 127.5) * bg_tone_contrast + 127.5 + (bg_tone_brightness * 255.0)
        img_tone = np.clip(img_tone_f, 0, 255).astype(np.uint8)
        
        if global_y_shift != 0:
            shifted_tone = np.full_like(img_tone, 255) # 흰색(255, 공백 톤) 패딩
            if global_y_shift > 0: shifted_tone[global_y_shift:, :] = img_tone[:-global_y_shift, :]
            else: shifted_tone[:global_y_shift, :] = img_tone[-global_y_shift:, :]
            img_tone = shifted_tone
        
        img_cos, img_sin, _ = calculate_orientation_map(img_bin)
        roi_map = np.zeros((target_h, target_w), dtype=np.int32)
        roi_weight_map = np.zeros((target_h, target_w), dtype=np.float32)
        
        print("[Shift_JIS AA] Progress 2/5: Eye ROI mask detecting...")
        if semantic_roi_bypass and opt_roi_mask is not None:
            print("[Shift_JIS AA] Mask detected!")
            m_np = cv2.resize((opt_roi_mask[0].cpu().numpy() * 255).astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            
            # 3. ROI 마스크 시프트
            if global_y_shift != 0:
                shifted_m_np = np.zeros_like(m_np) # 검은색(0, 마스크 없음) 패딩
                if global_y_shift > 0: shifted_m_np[global_y_shift:, :] = m_np[:-global_y_shift, :]
                else: shifted_m_np[:global_y_shift, :] = m_np[-global_y_shift:, :]
                m_np = shifted_m_np
                
            _, mt, st, _ = cv2.connectedComponentsWithStats(cv2.threshold(m_np, 127, 255, cv2.THRESH_BINARY)[1], connectivity=8)
            for i in range(1, len(st)):
                x, y, w, h, _ = st[i]
                if w > 0 and h > 0:
                    blob_mask = (mt[y:y+h, x:x+w] == i)
                    roi_map[y:y+h, x:x+w][blob_mask] |= 5
                    dist = cv2.distanceTransform(blob_mask.astype(np.uint8), cv2.DIST_L2, 5)
                    if dist.max() > 0:
                        roi_weight_map[y:y+h, x:x+w][blob_mask] = dist[blob_mask] / dist.max()
                    hh, hw = h//2, w//2
                    roi_map[y:y+hh, x:x+w][blob_mask[0:hh, :]] |= 16
                    roi_map[y+hh:y+h, x:x+w][blob_mask[hh:h, :]] |= 32
                    roi_map[y:y+h, x:x+hw][blob_mask[:, 0:hw]] |= 2
                    roi_map[y:y+h, x+hw:x+w][blob_mask[:, hw:w]] |= 8
                    
        tasks = []
        m_y = y_tolerance
        if bg_tone_mode.startswith("1"):
            m_y = 0 
            
        for r in range(text_lines):
            ys = r * ROW_HEIGHT
            ye = min(ys + ROW_HEIGHT, target_h)
            sys, sye = max(0, ys-m_y), min(target_h, ye+m_y)
            pt, pb = max(0, -(ys-m_y)), max(0, (ys+ROW_HEIGHT+m_y)-target_h)
            tasks.append([
                np.pad(img_bin_f32[sys:sye, :], ((pt, pb), (0, 0)), 'constant'),
                np.pad(img_tone[sys:sye, :], ((pt, pb), (0, 0)), 'constant'),
                np.pad(img_cos[sys:sye, :], ((pt, pb), (0, 0)), 'constant'),
                np.pad(img_sin[sys:sye, :], ((pt, pb), (0, 0)), 'constant'),
                np.pad(roi_map[sys:sye, :], ((pt, pb), (0, 0)), 'constant'),
                np.pad(roi_weight_map[sys:sye, :], ((pt, pb), (0, 0)), 'constant')
            ])
            
        res_l = [""] * len(tasks)
        all_p = [[] for _ in range(len(tasks))]
        replaced_boxes = [[] for _ in range(len(tasks))]
        
        # --- 프로그레스 바 전체 스텝 수 계산 ---
        total_pbar_steps = len(tasks)
        if bg_tone_mode.startswith("2") and bg_tone_weight > 0:
            total_pbar_steps += len(tasks)
        pbar = comfy.utils.ProgressBar(total_pbar_steps)
        
        print(f"[Shift_JIS AA] Progress 3/5: Tone matching in image... ({len(tasks)} lines total)")
        for idx, t in enumerate(tasks):
            res_l[idx], all_p[idx] = self.solve_stripe_hybrid(t[0], t[1], t[2], t[3], t[4], t[5], char_data_list, letter_spacing, phase_congruency_weight, density_penalty_weight, missing_penalty_weight, frequency_weight, roi_phase_weight, roi_density_weight, roi_missing_weight, roi_frequency_weight, roi_tone_weight, placement_method, m_y, y_shift_penalty, bg_tone_mode, bg_tone_weight)
            pbar.update(1)
            if idx % 10 == 0: torch.cuda.empty_cache()
                
        if bg_tone_mode.startswith("2") and bg_tone_weight > 0 and SjisVariableWidthGenerator._tone_chars_cache:
            print(f"[Shift_JIS AA] Progress 4/5: Tone matching in empty space... ({len(tasks)} lines total)")
            tone_chars = SjisVariableWidthGenerator._tone_chars_cache
            REPLACEABLE_CHARS = set(['　', ' ', '.', ',', "'", '．', '，']) 
            
            def get_best_tone_string_dynamic(patch, bg_weight):
                target_width = patch.shape[1]
                if target_width == 0: return None
                
                dp = {0: (0.0, 0, "")}
                for w in range(1, target_width + 1):
                    best_cost = float('inf')
                    best_prev = 0
                    best_char = ""
                    
                    for tc_tone, tc_char, tc_width in tone_chars:
                        if tc_width <= 0 or w < tc_width: continue
                        if (w - tc_width) in dp:
                            char_patch = patch[:, w - tc_width : w]
                            local_tone = 0.0
                            if char_patch.size > 0:
                                local_tone = (1.0 - (np.mean(char_patch) / 255.0)) * bg_weight
                                
                            diff = abs(tc_tone - local_tone)
                            cost = dp[w - tc_width][0] + diff
                            
                            if cost < best_cost:
                                best_cost = cost
                                best_prev = w - tc_width
                                best_char = tc_char
                                
                    if best_cost < float('inf'):
                        dp[w] = (best_cost, best_prev, best_char)
                        
                if target_width in dp:
                    res_str = ""
                    curr = target_width
                    while curr > 0:
                        _, prev, ch = dp[curr]
                        res_str = ch + res_str
                        curr = prev
                    return res_str
                return None 

            for idx, t in enumerate(tasks):
                row_tone = t[1][m_y:m_y+ROW_HEIGHT, :]
                new_line = ""
                
                chunk_chars = ""
                chunk_w = 0.0
                chunk_start_x = 0.0
                
                def flush_chunk():
                    nonlocal new_line, chunk_chars, chunk_w, chunk_start_x
                    if chunk_w > 0:
                        ix, icw = int(round(chunk_start_x)), int(round(chunk_w))
                        patch = row_tone[:, ix:ix+icw]
                        
                        if patch.size > 0:
                            max_tone_in_patch = (1.0 - (np.min(patch) / 255.0)) * bg_tone_weight
                            if max_tone_in_patch > 0.05:
                                filled_str = get_best_tone_string_dynamic(patch, bg_tone_weight)
                                if filled_str is not None:
                                    new_line += filled_str
                                    replaced_boxes[idx].append((ix, icw))
                                else:
                                    new_line += chunk_chars 
                            else:
                                new_line += chunk_chars 
                        else:
                            new_line += chunk_chars
                        
                        chunk_chars = ""
                        chunk_w = 0.0

                for (char, x, cw, is_roi) in all_p[idx]:
                    if not is_roi and char in REPLACEABLE_CHARS:
                        if chunk_w == 0: chunk_start_x = x
                        chunk_chars += char
                        chunk_w += cw
                    else:
                        flush_chunk() 
                        new_line += char
                flush_chunk() 
                res_l[idx] = new_line
                pbar.update(1)
        else:
            print("[Shift_JIS AA] Progress 4/5: Tone matching skipped (Mode 3: Line-art Only)")
            
        print("[Shift_JIS AA] Progress 5/5: Final image rendering...")
        aa_text = "\n".join(res_l)
        dummy_draw = ImageDraw.Draw(Image.new("L", (1,1)))
        max_w = max([sum([get_char_width(c, font, dummy_draw) + letter_spacing for c in l]) for l in res_l] + [64.0])
        canvas = Image.new("RGB", (int(max_w), target_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for i, line in enumerate(res_l):
            x = 0.0
            y = i * ROW_HEIGHT
            for char in line:
                draw.text((int(round(x)), y), char, font=font, fill=(0,0,0))
                x += max(1.0, get_char_width(char, font, dummy_draw) + letter_spacing)
                
        grid_vis = cv2.cvtColor(255 - img_bin, cv2.COLOR_GRAY2RGB)
        
        overlay = grid_vis.copy()
        for r, boxes in enumerate(replaced_boxes):
            y = r * ROW_HEIGHT
            for (x, w) in boxes:
                cv2.rectangle(overlay, (x, y), (x + w, y + ROW_HEIGHT), (150, 210, 250), -1)
        cv2.addWeighted(overlay, 0.5, grid_vis, 0.5, 0, grid_vis)
        
        for r, pl in enumerate(all_p):
            y = r * ROW_HEIGHT
            cv2.line(grid_vis, (0, y), (target_w, y), (0, 0, 255), 1)
            for (_, x, cw, ir) in pl:
                cv2.rectangle(grid_vis, (int(round(x)), y), (int(round(x+cw)), y+ROW_HEIGHT), (0,255,0) if ir else (255,0,0), 2 if ir else 1)
        
        print("[Shift_JIS AA] All task completed!")
        return (torch.from_numpy(np.array(canvas).astype(np.float32)/255.0).unsqueeze(0), aa_text, torch.from_numpy(np.array(grid_vis).astype(np.float32)/255.0).unsqueeze(0))

NODE_CLASS_MAPPINGS = {"SjisLineExtractor": SjisLineExtractor, "SjisThinning": SjisThinning, "SjisVariableWidthGenerator": SjisVariableWidthGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"SjisLineExtractor": "Shift_JIS Line Extractor", "SjisThinning": "Shift_JIS Thinning (Skeletonize)", "SjisVariableWidthGenerator": "Shift_JIS Variable AA (Generator)"}