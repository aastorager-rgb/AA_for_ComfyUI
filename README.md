# ComfyUI Shift-JIS Art Generator Nodes (v1.1)

This project is a custom node set for ComfyUI that analyzes original images to automatically generate high-quality, variable-width Shift-JIS ASCII art, simulating the intricate work of human ASCII artists.

Moving beyond simple Western-style density (tone-based) mapping, it provides a hybrid pipeline that supports line-art extraction, skeletonization, topological morphology matching using Japanese Shift-JIS character templates, and Dynamic Programming (DP)-based background tone replacement.

## Pipeline Structure

The pipeline consists of three sequential nodes:

### 1. Shift_JIS Line Extractor

Extracts the structural framework (edges) from the image to be represented by characters.

* **Method**: Select from various algorithms including Segmentation (K-means), DoG (Soft Lines), Canny (Hard Edges), and Simple (Grayscale).
* **Line Thickness**: Adjusts the rendered line thickness to improve the stability of the subsequent thinning process.

### 2. Shift_JIS Thinning (Skeletonize)

Compresses the extracted line-art into a 1px-wide skeleton to create the exact trajectory where characters will be matched (utilizes the Guo-Hall Thinning algorithm).

* **Clean Strength & Anti Loop Fill**: Removes noise and fills unnecessarily closed loops to finalize a clean skeletal structure.

### 3. Shift_JIS Variable AA (Generator)

The main generator that simultaneously analyzes the thinned image and the original image's tone (brightness) to place the optimal Shift-JIS characters.

## Core Features (Generator Node)

### 1. Background Tone Application Modes (BG Tone Mode)

Utilizes the `char_tone.txt` dictionary to fill empty spaces lacking line-art with characters matching the original image's brightness.

* **Mode 1: Full Area Tone**: Matches characters to all areas exceeding a specified tone weight threshold, regardless of line-art presence. (When selected, Y-Tolerance is forcibly set to 0 to prevent top/bottom edge artifacts.)
* **Mode 2: Fill Empty Spaces (Dynamic Replacement)**: Completes the initial line-art based generation, then identifies continuous blocks of empty spaces or dots. It uses Dynamic Programming (DP) to calculate the actual local tone of the pixels in these blocks, seamlessly replacing them with the optimal combination of characters. Replaced areas are visually indicated by a light blue translucent box in the `grid_image` output.
* **Mode 3: Line-art Only**: Ignores background tones and places characters strictly along the extracted line-art trajectories.

*Tone Control Parameters:*

* `bg_tone_weight`: Determines how strongly the original image's tone is reflected. (Acts like Mode 3 if set to 0).
* `bg_tone_contrast` / `bg_tone_brightness`: Pre-processes the contrast and brightness of the original tone map before text replacement.

### 2. Global Y-Axis Shift

* `global_y_shift` (-16px to +16px)
* If line breaks or character placements are slightly misaligned, you can shift the entire input line-art, tone map, and ROI mask along the Y-axis to find the optimal hit point. New empty spaces created by this shift are treated as pure white/blank areas.

### 3. Semantic ROI & Eye Feature Preservation

Specially manages complex and easily distorted core areas like eyes and faces by using an optional mask (`opt_roi_mask`).

* Relaxes density/phase penalties within the ROI and heavily increases the selection weight (+15.0) for dedicated Kanji and full-width characters (8px or wider) to preserve delicate expressions.
* The `semantic_roi_bypass` toggle allows you to instantly enable or disable the ROI application.

### 4. Robust Custom CSV Parser & Auto-Caching

* **Safe Data Loading**: Features a custom parsing algorithm that flawlessly reads Japanese text files (CP932, UTF-8) even if they contain stray commas or unclosed quotes, preventing Pandas `ParserError` or `Out of bounds` exceptions.
* **Smart Cache Invalidation**: Automatically detects changes in the selected CSV font list or Tone text file, immediately purging and rebuilding the legacy tensor cache in memory to completely prevent dimension conflicts and character corruption.

### 5. Step-by-Step Progress Monitoring

Tracks the detailed 5-step generation process (Resource Load -> ROI Mask Check -> Image Text Matching -> Empty Space Tone Matching -> Final Rendering) in real-time via the terminal console and the ComfyUI progress bar.

## I/O Specifications

**Inputs**

* `original_image`: The original image (used for tone analysis and DP matching).
* `thinned_image`: The skeletal image passed from the Thinning node.
* `char_list_path` / `char_tone_path`: The character frequency list (.csv) and tone dictionary (.txt) files.
* `font_path`: The font file to use for rendering (.ttf).
* *(Optional)* `opt_roi_mask`: A mask image designating crucial semantic areas like the face or eyes.

**Outputs**

* `aa_image`: The final ASCII art image rendered with the selected font.
* `aa_text`: The raw, copy-pasteable text string data.
* `grid_image`: A debugging image visualizing red scan lines, character bounding boxes (green/blue), and tone replacement areas (light blue).

---

---

# ComfyUI Shift-JIS Art Generator Nodes (v1.1)

이 프로젝트는 ComfyUI 환경에서 원본 이미지를 분석하여 실제 사람이 수작업으로 만든 것과 같은 고품질 가변 폭(Variable-width) Shift-JIS 아스키 아트를 자동으로 생성하는 커스텀 노드 세트입니다.

단순한 밀도(Tone-based) 매핑을 넘어, 선화(Line-art) 추출, 골격화(Skeletonize), 일본어 문자 템플릿(Shift-JIS)의 위상수학적 형태학(Morphology) 매칭, 그리고 동적 계획법(DP)을 활용한 배경 톤 치환까지 지원하는 하이브리드 파이프라인을 제공합니다.

## 파이프라인 구조

파이프라인은 다음 3단계의 노드로 구성됩니다.

### 1. Shift_JIS Line Extractor (선화 추출)

이미지에서 문자로 표현될 뼈대(에지)를 추출합니다.

* **Method**: K-means 분할, DoG(Soft Lines), Canny(Hard Edges), Simple(Grayscale) 등 다양한 알고리즘을 선택할 수 있습니다.
* **Line Thickness**: 렌더링될 선의 두께를 조절하여 후속 세선화 과정의 안정성을 높입니다.

### 2. Shift_JIS Thinning (골격화/세선화)

추출된 선화 이미지에서 두께를 1px 단위의 골격(Skeleton)으로 압축하여 문자가 매칭될 정확한 궤적을 만듭니다. (Guo-Hall Thinning 알고리즘 사용)

* **Clean Strength & Anti Loop Fill**: 노이즈를 제거하고 불필요하게 닫힌 루프를 채워 깔끔한 뼈대를 완성합니다.

### 3. Shift_JIS Variable AA (Generator) (메인 생성기)

세선화된 이미지와 원본 이미지의 톤(밝기)을 동시에 분석하여 최적의 Shift-JIS 문자를 배치합니다.

## 핵심 주요 기능 (Generator Node)

### 1. 배경 톤 애플리케이션 모드 (BG Tone Mode)

`char_tone.txt` 파일의 명암 사전을 활용하여, 선화가 없는 공간도 원본 이미지의 밝기에 맞춰 문자로 채워 넣습니다.

* **Mode 1: Full Area Tone**: 선화 유무와 관계없이 지정된 톤 가중치 임계값을 넘는 모든 영역에 문자를 매칭합니다. (이 모드 선택 시 Y-Tolerance는 강제로 0으로 고정되어 상하단 가장자리 오류를 방지합니다.)
* **Mode 2: Fill Empty Spaces (동적 치환)**: 선화를 기반으로 1차 생성을 완료한 뒤, 연속된 공백이나 점으로 이루어진 '빈 공간 덩어리'를 찾아냅니다. 동적 계획법(DP)을 사용하여 해당 덩어리의 실제 픽셀 부분 톤(Local Tone)을 계산하고, 가장 오차가 적은 문자의 조합으로 빈 공간을 덮어씌웁니다. 치환된 영역은 `grid_image` 출력 시 옅은 파란색 반투명 박스로 표시됩니다.
* **Mode 3: Line-art Only**: 배경 톤을 무시하고 오직 추출된 선화 궤적에만 문자를 배치합니다.

*톤 제어 파라미터:*

* `bg_tone_weight`: 원본 이미지의 톤을 얼마나 강하게 반영할지 결정합니다. (0으로 설정 시 Mode 3과 동일하게 작동)
* `bg_tone_contrast` / `bg_tone_brightness`: 텍스트로 치환되기 전 원본 톤 맵의 명암비와 밝기를 전처리합니다.

### 2. 전역 Y축 이동 (Global Y Shift)

* `global_y_shift` (-16px ~ +16px)
* 줄바꿈이나 글자 매칭 위치가 미묘하게 어긋날 때, 입력된 선화, 톤 이미지, ROI 마스크 전체를 Y축으로 이동시켜 최적의 타격점(Hit point)을 찾을 수 있습니다. 이동으로 인해 생기는 빈 공간은 깔끔한 공백으로 처리됩니다.

### 3. 시맨틱 ROI 및 눈 보존

눈(Eyes)과 같이 형태가 복잡하고 뭉개지기 쉬운 핵심 영역을 마스킹(`opt_roi_mask`)하여 특별 관리합니다.

* ROI 영역 내에서는 밀도/위상 페널티를 완화하고, 전용 한자 및 전각 문자(폭 8px 이상)의 출현 가중치(+15.0)를 높여 정교한 표정을 살려냅니다.
* `semantic_roi_bypass` 옵션으로 ROI 적용 여부를 즉시 켜고 끌 수 있습니다.

### 4. 강력한 커스텀 CSV 파서 및 자동 캐싱

* **안전한 데이터 로드**: 일본어 텍스트 파일(CP932, UTF-8)의 특성상 내부에 콤마나 단일 따옴표가 섞여 있어도 에러(ParserError, Out of bounds) 없이 완벽하게 읽어내는 커스텀 파싱 알고리즘이 적용되었습니다.
* **스마트 캐시 초기화**: 사용할 CSV 폰트 리스트나 Tone 텍스트 파일이 변경되면, 메모리에 남아있던 이전 텐서 캐시를 즉각 폭파하고 새로 빌드하여 차원 충돌 및 글자 깨짐 현상을 차단합니다.

### 5. 진행 상황 모니터링

터미널 콘솔창 및 ComfyUI 프로그레스 바를 통해 총 5단계의 상세한 진행 상황(리소스 로드 -> ROI 마스크 확인 -> 이미지 글자 매칭 -> 빈 공간 톤 매칭 -> 최종 렌더링)을 실시간으로 추적할 수 있습니다.

## 입출력 (I/O) 안내

**입력 (Inputs)**

* `original_image`: 원본 이미지 (톤 분석 및 DP 매칭용)
* `thinned_image`: 세선화 노드에서 전달받은 뼈대 이미지
* `char_list_path` / `char_tone_path`: 사용할 문자 빈도 리스트(.csv) 및 톤 사전(.txt) 파일
* `font_path`: 렌더링에 사용할 폰트 (.ttf)
* *(Optional)* `opt_roi_mask`: 얼굴/눈 등 중요 부위를 지정하는 마스크 이미지

**출력 (Outputs)**

* `aa_image`: 폰트를 렌더링하여 최종 완성된 아스키 아트 이미지
* `aa_text`: 복사/붙여넣기 가능한 실제 텍스트 문자열 데이터
* `grid_image`: 붉은색 스캔 라인, 글자 인식용 Bounding Box(초록/파랑), 그리고 톤 대체 영역(옅은 파란색)을 시각화한 디버깅용 이미지
