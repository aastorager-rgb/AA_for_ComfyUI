Here is the updated documentation, incorporating the line thinning/matching feature, node-specific parameter descriptions, and system requirements. It is provided in both English and Korean without any emojis.

---

# English Version

# ComfyUI Shift-JIS Art Generator Nodes (v1.0)

This project is a custom node set for ComfyUI that analyzes original images to automatically generate high-quality, variable-width Shift-JIS ASCII art, simulating the intricate work of human ASCII artists.

Moving beyond simple density (tone-based) mapping, it provides a hybrid pipeline that supports line-art extraction, skeletonization, topological morphology matching using Japanese Shift-JIS character templates, and Dynamic Programming (DP)-based background tone replacement.

## Requirements

To use these nodes effectively within a ComfyUI workflow, the following environments and models are required:

* **ComfyUI**: Core framework.
* **Python Libraries**: `torch`, `numpy`, `opencv-python` (`cv2`), `pandas`, `Pillow` (`PIL`).
* **ComfyUI Impact Pack** (Highly Recommended): Used for detecting faces and eyes to generate the ROI mask.
* **BBox Models**: `bbox/face_yolov8m.pt` or `bbox/eyes.pt` (or similar bounding box models) to supply the `opt_roi_mask` to the generator.

## Pipeline Structure

The pipeline consists of three sequential nodes:

1. **Shift_JIS Line Extractor**: Extracts the structural framework (edges) from the image.
2. **Shift_JIS Thinning**: Compresses the extracted line-art into a 1px-wide skeleton.
3. **Shift_JIS Variable AA (Generator)**: Analyzes the thinned image and the original image's tone to place the optimal Shift-JIS characters.

## Core Features

### 1. Line Thinning and Topological Matching

Instead of merely mapping dark pixels to dense characters, the system extracts the true topological skeleton (1px thick) of the image using Guo-Hall thinning. It then calculates orientation maps (gradients) and structurally matches them with the geometric masks of thousands of Shift-JIS characters. This ensures that the flow of hair, jawlines, and clothing wrinkles are perfectly traced by the characters.

### 2. Background Tone Application Modes (BG Tone Mode)

Utilizes the `char_tone.txt` dictionary to fill empty spaces lacking line-art with characters matching the original image's brightness.

* **Mode 1: Full Area Tone**: Matches characters to all areas exceeding a specified tone weight threshold, regardless of line-art presence.
* **Mode 2: Fill Empty Spaces (Dynamic Replacement)**: Completes the initial line-art based generation, then identifies continuous blocks of empty spaces or dots. It uses Dynamic Programming (DP) to calculate the actual local tone of the pixels in these blocks, seamlessly replacing them with the optimal combination of characters. Replaced areas are visually indicated by a light blue translucent box in the debugging grid.
* **Mode 3: Line-art Only**: Ignores background tones and places characters strictly along the extracted line-art trajectories.

### 3. Global Y-Axis Shift

If line breaks or character placements are slightly misaligned, you can shift the entire input line-art, tone map, and ROI mask along the Y-axis (-16px to +16px) to find the optimal hit point. New empty spaces created by this shift are treated as pure white/blank areas.

### 4. Semantic ROI & Feature Preservation

Specially manages complex and easily distorted core areas like eyes and faces by using an optional mask (`opt_roi_mask`). It relaxes density/phase penalties within the ROI and heavily increases the selection weight for dedicated Kanji and full-width characters (8px or wider) to preserve delicate expressions.

### 5. Robust Custom CSV Parser & Auto-Caching

Features a custom parsing algorithm that flawlessly reads Japanese text files (CP932, UTF-8) even if they contain stray commas or unclosed quotes. It automatically detects changes in the selected files, immediately purging and rebuilding the legacy tensor cache in memory to completely prevent dimension conflicts and character corruption.

### 6. Step-by-Step Progress Monitoring

Tracks the detailed 5-step generation process in real-time via the terminal console and the ComfyUI progress bar.

## I/O Specifications & Node Parameters

### Node 1: Shift_JIS Line Extractor

**Inputs**

* `image`: Original input image.
* `text_lines`: Target resolution measured in lines of text.
* `method`: Extraction algorithm (Segmentation, DoG, Canny, Simple, None).
* `threshold`: Binary threshold value (0-255).
* `segmentation_k`: Number of clusters for K-means method.
* `line_thickness`: Adjusts the thickness of the extracted lines before thinning.
* `invert_output`: Inverts the black/white output.
**Outputs**
* `line_image`: Extracted line-art image.

### Node 2: Shift_JIS Thinning (Skeletonize)

**Inputs**

* `line_image`: Output from the Extractor node.
* `threshold`: Binarization threshold for the input image.
* `clean_strength`: Morphological cleaning strength to remove noise.
* `anti_loop_fill`: Area threshold to fill unnecessarily closed loops before thinning.
**Outputs**
* `thinned_image`: 1px-wide skeletonized image.

### Node 3: Shift_JIS Variable AA (Generator)

**Inputs**

* `original_image`: The original image (used for tone analysis).
* `thinned_image`: The skeletal image passed from the Thinning node.
* `char_list_path` / `font_path` / `char_tone_path`: File paths for the CSV list, TTF font, and TXT tone dictionary.
* `text_lines`: Number of vertical text lines (must match Extractor).
* `letter_spacing`: Spacing between characters.
* `placement_method`: Algorithm for placing text (Score-Priority or Sequential).
* `global_y_shift`: Shifts the entire processing matrix up or down (px).
* `y_tolerance` / `y_shift_penalty`: Controls how much characters can vertically deviate to match lines.
* `phase_congruency_weight` / `density_penalty_weight` / `missing_penalty_weight` / `frequency_weight`: Core topological scoring weights.
* `semantic_roi_bypass`: Toggles the ROI mask logic on or off.
* `roi_*_weights`: Custom scoring weights applied strictly inside the ROI mask.
* `bg_tone_mode`: Selects the background application logic (Mode 1, 2, or 3).
* `bg_tone_weight` / `bg_tone_contrast` / `bg_tone_brightness`: Tone adjustment parameters for background replacement.
* *(Optional)* `opt_roi_mask`: A mask designating crucial areas (e.g., from an Impact Pack BBox detector).

**Outputs**

* `aa_image`: The final rendered ASCII art image.
* `aa_text`: Copy-pasteable text string data.
* `grid_image`: Debugging image visualizing scan lines, hit boxes, and DP tone replacement areas.

---

---

# Korean Version

# ComfyUI Shift-JIS Art Generator Nodes (v1.0)

이 프로젝트는 ComfyUI 환경에서 원본 이미지를 분석하여 실제 사람이 수작업으로 만든 것과 같은 고품질 가변 폭(Variable-width) Shift-JIS 아스키 아트를 자동으로 생성하는 커스텀 노드 세트입니다.

단순한 밀도(Tone-based) 매핑을 넘어, 선화(Line-art) 추출, 골격화(Skeletonize), 일본어 문자 템플릿(Shift-JIS)의 위상수학적 형태학(Morphology) 매칭, 그리고 동적 계획법(DP)을 활용한 배경 톤 치환까지 지원하는 하이브리드 파이프라인을 제공합니다.

## 요구 사항 (Requirements)

이 노드를 ComfyUI 워크플로우에서 정상적으로 구동하기 위해 다음 환경 및 모델이 필요합니다.

* **ComfyUI**: 기본 실행 프레임워크
* **Python 라이브러리**: `torch`, `numpy`, `opencv-python` (`cv2`), `pandas`, `Pillow` (`PIL`)
* **ComfyUI Impact Pack** (강력 권장): 얼굴 및 눈 영역을 인식하여 ROI 마스크를 생성하는 데 사용됩니다.
* **BBox 모델**: `bbox/face_yolov8m.pt` 또는 `bbox/eyes.pt` 등 (Generator 노드의 `opt_roi_mask`에 연결하여 사용)

## 파이프라인 구조

파이프라인은 다음 3단계의 노드로 구성됩니다.

1. **Shift_JIS Line Extractor**: 이미지에서 문자로 표현될 뼈대(에지)를 추출합니다.
2. **Shift_JIS Thinning**: 추출된 선화 이미지를 1px 두께의 골격으로 압축합니다.
3. **Shift_JIS Variable AA (Generator)**: 세선화된 이미지와 원본 이미지의 톤을 분석하여 문자를 매칭합니다.

## 핵심 주요 기능

### 1. 선화 세선화(Line Thinning) 및 위상수학적 매칭

어두운 픽셀을 단순히 밀도가 높은 문자로 치환하는 방식을 벗어나, Guo-Hall 세선화 알고리즘을 통해 이미지의 진정한 위상수학적 골격(1px 두께)을 추출합니다. 이후 추출된 뼈대의 방향성(Gradient)을 계산하고, 수천 개의 Shift-JIS 문자가 가진 기하학적 마스크와 구조적으로 매칭합니다. 이를 통해 머리카락의 흐름, 턱선, 옷 주름 등이 문자의 획을 따라 완벽하게 추적됩니다.

### 2. 배경 톤 애플리케이션 모드 (BG Tone Mode)

`char_tone.txt` 파일의 명암 사전을 활용하여, 선화가 없는 공간도 원본 이미지의 밝기에 맞춰 문자로 채워 넣습니다.

* **Mode 1: Full Area Tone**: 선화 유무와 관계없이 지정된 톤 가중치 임계값을 넘는 모든 영역에 문자를 매칭합니다.
* **Mode 2: Fill Empty Spaces (동적 치환)**: 선화를 기반으로 1차 생성을 완료한 뒤, 연속된 공백이나 점으로 이루어진 '빈 공간 덩어리'를 찾아냅니다. 동적 계획법(DP)을 사용하여 해당 덩어리의 실제 픽셀 부분 톤(Local Tone)을 계산하고, 가장 오차가 적은 문자의 조합으로 빈 공간을 덮어씌웁니다. 치환된 영역은 디버그 그리드에서 옅은 파란색으로 표시됩니다.
* **Mode 3: Line-art Only**: 배경 톤을 무시하고 오직 추출된 선화 궤적에만 문자를 배치합니다.

### 3. 전역 Y축 이동 (Global Y Shift)

줄바꿈이나 글자 매칭 위치가 미묘하게 어긋날 때, 입력된 선화, 톤 이미지, ROI 마스크 전체를 Y축으로 이동(-16px ~ +16px)시켜 최적의 타격점(Hit point)을 찾을 수 있습니다. 이동으로 인해 생기는 빈 공간은 깔끔한 공백으로 처리됩니다.

### 4. 시맨틱 ROI 및 얼굴/눈 보존

눈(Eyes)과 같이 형태가 복잡하고 뭉개지기 쉬운 핵심 영역을 마스킹(`opt_roi_mask`)하여 특별 관리합니다. ROI 영역 내에서는 밀도 및 위상 페널티를 완화하고, 전용 한자 및 전각 문자(폭 8px 이상)의 출현 가중치를 높여 정교한 표정을 살려냅니다.

### 5. 강력한 커스텀 CSV 파서 및 자동 캐싱

일본어 텍스트 파일(CP932, UTF-8) 내부에 콤마나 단일 따옴표가 섞여 있어도 파싱 에러 없이 완벽하게 읽어내는 커스텀 알고리즘이 적용되었습니다. 또한 사용할 파일이 변경되면 기존 텐서 캐시를 즉각 폭파하고 새로 빌드하여 차원 충돌 및 글자 깨짐 현상을 원천 차단합니다.

### 6. 진행 상황 모니터링

터미널 콘솔창 및 ComfyUI 프로그레스 바를 통해 총 5단계의 상세한 진행 상황을 실시간으로 추적할 수 있습니다.

## 노드 별 입출력 및 파라미터 안내

### Node 1: Shift_JIS Line Extractor

**입력 (Inputs)**

* `image`: 원본 이미지
* `text_lines`: 생성할 텍스트 줄 수 (해상도 결정)
* `method`: 외곽선 추출 알고리즘 (Segmentation, DoG, Canny, Simple, None)
* `threshold`: 이진화 임계값 (0-255)
* `segmentation_k`: K-means 분할 방식 사용 시 군집(Cluster) 개수
* `line_thickness`: 세선화 전 외곽선의 두께를 보강하여 안정성을 높임
* `invert_output`: 흑백 반전 여부
**출력 (Outputs)**
* `line_image`: 추출된 선화 이미지

### Node 2: Shift_JIS Thinning (Skeletonize)

**입력 (Inputs)**

* `line_image`: Extractor 노드의 결과물
* `threshold`: 입력 이미지 이진화 임계값
* `clean_strength`: 노이즈를 제거하는 형태학적(Morphological) 클리닝 강도
* `anti_loop_fill`: 불필요하게 닫힌 픽셀 루프를 칠해버릴 면적 임계값
**출력 (Outputs)**
* `thinned_image`: 1px 두께로 세선화된 이미지

### Node 3: Shift_JIS Variable AA (Generator)

**입력 (Inputs)**

* `original_image`: 원본 이미지 (배경 톤 분석용)
* `thinned_image`: Thinning 노드에서 출력된 뼈대 이미지
* `char_list_path` / `font_path` / `char_tone_path`: 각각 CSV 리스트, TTF 폰트, TXT 톤 사전 파일 경로
* `text_lines`: 텍스트 줄 수 (Extractor와 동일해야 함)
* `letter_spacing`: 자간 조절
* `placement_method`: 글자 배치 알고리즘 (점수 우선 방식 또는 순차적 방식)
* `global_y_shift`: 내부 매트릭스 전체를 Y축으로 위아래로 이동 (px 단위)
* `y_tolerance` / `y_shift_penalty`: 글자가 라인에 맞춰 상하로 움직일 수 있는 허용 범위와 페널티
* `phase_congruency_weight` / `density_penalty_weight` / `missing_penalty_weight` / `frequency_weight`: 선화 궤적 위상 매칭을 위한 핵심 가중치들
* `semantic_roi_bypass`: ROI 마스크 기능 켜기/끄기
* `roi_*_weights`: ROI 마스크 내부에서만 독자적으로 적용되는 가중치들
* `bg_tone_mode`: 배경 톤 적용 모드 (Mode 1, 2, 3)
* `bg_tone_weight` / `bg_tone_contrast` / `bg_tone_brightness`: 빈 공간 치환 시 원본 톤을 얼마나 보정하여 반영할지 결정하는 파라미터
* *(Optional)* `opt_roi_mask`: Impact Pack의 BBox 모델 등을 통해 생성된 시맨틱 마스크 이미지

**출력 (Outputs)**

* `aa_image`: 폰트가 렌더링된 최종 아스키 아트 이미지
* `aa_text`: 복사하여 다른 곳에 붙여넣을 수 있는 실제 텍스트 문자열
* `grid_image`: 붉은색 스캔 라인, 인식된 글자 박스, 톤 대체 영역을 시각적으로 보여주는 디버깅용 이미지
