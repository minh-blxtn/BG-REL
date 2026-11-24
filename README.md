# BG-REL V3.2 — HƯỚNG DẪN CÀI ĐẶT, SỬ DỤNG VÀ TÁI KIỂM NGHIÊN CỨU (R5)

> **Đọc mục 1 → 6 nếu bạn vừa tải dự án từ Google Drive.** Tài liệu này được viết để người không có máy của tác giả cũng có thể tạo môi trường Python mới, khôi phục đúng cấu trúc thư mục và **kiểm chứng các kết quả đã lưu**.
>
> **Trạng thái chính xác:** R5 Deep Merge đã được triển khai thành công trên máy gốc vào **07/11/2026** (`R5_SOURCE_ONLY_CUTOVER_PASS`, nâng cấp từ `R4_CLEAN`). Windows regression báo **61/61 unit tests PASS**, 19 lệnh CLI `--help` PASS; các checkpoint, dữ liệu frozen và số liệu trong bài báo đã được đối chiếu. **Chưa có bằng chứng** rằng một máy mới có thể tái chạy tự động toàn bộ P0–P9 hoặc tái hiện chính xác quá trình train lịch sử. Hướng dẫn cài mới bên dưới là **quy trình được suy ra và kiểm tra tĩnh từ source**, chưa có kết quả cài thử trên một máy Windows thứ hai.

**Tên đề tài:** *BG-REL: Lung-Aware Box-Guided Relative Evidence Learning for Chest Radiograph Lung-Opacity Classification*.

**Phạm vi:** Nghiên cứu phân loại nhị phân nhãn **RSNA Lung Opacity** trên ảnh X-quang ngực DICOM và nghiên cứu bằng chứng vùng tổn thương. Đây **không** phải phần mềm chẩn đoán viêm phổi, thiết bị y tế hay công cụ quyết định lâm sàng.

---

## Mục lục

1. [Bắt đầu nhanh: người mới cần làm gì?](#1-bắt-đầu-nhanh-người-mới-cần-làm-gì)
2. [Máy tính và phần mềm cần chuẩn bị](#2-máy-tính-và-phần-mềm-cần-chuẩn-bị)
3. [Tải dự án và khôi phục đúng cây thư mục](#3-tải-dự-án-và-khôi-phục-đúng-cây-thư-mục)
4. [Tạo Python 3.12 venv và cài dependency](#4-tạo-python-312-venv-và-cài-dependency)
5. [Tải/kiểm tra pretrained ImageNet và PSPNet](#5-tảikiểm-tra-pretrained-imagenet-và-pspnet)
6. [Kiểm tra installation và các kết quả đã lưu](#6-kiểm-tra-installation-và-các-kết-quả-đã-lưu)
7. [Các chế độ sử dụng và giới hạn thực tế](#7-các-chế-độ-sử-dụng-và-giới-hạn-thực-tế)
8. [Giải thích P0–P9: input, output và thứ tự](#8-giải-thích-p0p9-input-output-và-thứ-tự)
9. [Lệnh CLI của 29 module R5](#9-lệnh-cli-của-29-module-r5)
10. [Kiến trúc và phương pháp nghiên cứu](#10-kiến-trúc-và-phương-pháp-nghiên-cứu)
11. [Dataset, split và kết quả paper](#11-dataset-split-và-kết-quả-paper)
12. [Provenance, checkpoint SHA và tính toàn vẹn](#12-provenance-checkpoint-sha-và-tính-toàn-vẹn)
13. [Chia sẻ trên Google Drive/GitHub](#13-chia-sẻ-trên-google-drivegithub)
14. [Các lỗi phổ biến và cách xử lý](#14-các-lỗi-phổ-biến-và-cách-xử-lý)
15. [Giới hạn tái lập và checklist bàn giao](#15-giới-hạn-tái-lập-và-checklist-bàn-giao)

---

## 1. Bắt đầu nhanh: người mới cần làm gì?

### 1.1. Chọn đúng mục tiêu

| Mục tiêu của người tải | Cần tải | Có thể thực hiện ngay sau khi cài? |
|---|---|---|
| **A. Đọc code / chạy unit tests** | `code/`, `tests/`, `scripts/`, `reproducibility/`, `docs/`, README, Bash; môi trường Python | **Có thể thử** sau khi cài dependency; 61 tests đã PASS trên máy gốc, không bảo đảm mọi máy tương đương |
| **B. Khôi phục bản nghiên cứu và đối chiếu số liệu paper** | Toàn bộ SOURCE + `data/` + `models/` + `results/` với đúng đường dẫn | **Có**, sử dụng lệnh `verify`, `tests` và `check`; cần tài nguyên/hash khớp |
| **C. Chạy checkpoint inference trên ảnh** | SOURCE + đúng checkpoint + DICOM + môi trường model; script evaluation còn có nhiều kiểm tra provenance | **Không có CLI `predict một ảnh` độc lập đã được nghiệm thu**; cần dùng API model/đường evaluator có kiểm soát |
| **D. Train lại B0/B1/A3 và chạy P0–P9 từ đầu** | SOURCE + raw DICOM/JSON + PSPNet/masks/P3 + GPU + môi trường + nghiên cứu đã phê duyệt | **Chưa được xác nhận end-to-end trên máy sạch**; các bước ghi dữ liệu có guard chống ghi đè run lịch sử; xem §7–9 |

**Khuyến nghị cho người nhận Drive:** bắt đầu với **B** để xác minh bộ dữ liệu, checkpoint và bảng số liệu được chia sẻ. Không dùng lệnh train hay P9 `--run` làm phép thử cài đặt.

### 1.2. Quy trình nhanh (Windows, PowerShell + Git Bash)

1. Tải và giải nén SOURCE + DATA + MODELS + RESULTS vào **một** thư mục, ví dụ `D:\Research\bgrel` (cây đúng như §3).
2. Cài **Python 3.12 64-bit**, Git for Windows (để có Git Bash) và nếu có GPU NVIDIA thì cài driver tương thích.
3. Trong PowerShell tại thư mục dự án: `py -3.12 -m venv .venvs\bgrel`.
4. Cài PyTorch theo CPU hoặc CUDA bằng lệnh **đúng phiên bản** ở §4; cài `requirements.research.txt` đi kèm gói README này.
5. Kiểm tra `torch`, `torchvision`, `torchxrayvision`, CUDA (nếu dùng), các file và hash.
6. Mở **Git Bash** trong thư mục dự án, chạy `bash run_pipeline.sh status`, `verify`, `tests`; với dự án đầy đủ, có thể chạy `check`.

**Điểm cần phân biệt:** `scripts/INSTALL.ps1` **KHÔNG** phải installer cho máy mới. Đó là công cụ *nâng cấp source R3/R4 lên R5 có guard*, cần backup đã biết. **Không chạy `scripts/INSTALL.ps1` trên máy người nhận.**

---

## 2. Máy tính và phần mềm cần chuẩn bị

### 2.1. Hệ điều hành, Python và công cụ

**Đã có bằng chứng trên máy nghiên cứu:** Windows 11 64-bit; Python **3.12.0** trong `.venvs\bgrel`; PyTorch **2.9.1+cu128**; torchvision **0.24.1+cu128**; RTX 4050 Laptop GPU; Git Bash/PowerShell. CPU-only hoặc nền tảng khác là **phương án tham khảo, chưa được regression xác nhận giống môi trường gốc**.

Cài đặt trên Windows:

- **Python 3.12.x x64** từ [python.org](https://www.python.org/downloads/) (máy gốc: 3.12.0). Trong Windows installer có thể bật Python Launcher `py`.
- **Git for Windows** từ [git-scm.com](https://git-scm.com/downloads/win) để có Git Bash. Không bắt buộc dùng GitHub nếu chỉ giải nén Drive.
- **NVIDIA driver** nếu muốn chạy CUDA; kiểm tra bằng `nvidia-smi` (không cần cài toàn bộ CUDA Toolkit chỉ để dùng PyTorch wheel có CUDA runtime, miễn driver phù hợp).
- Kết nối Internet để tải pip packages và trọng số MobileNetV2 ImageNet nếu chưa có sẵn.
- Dung lượng ổ cứng đủ chứa thư mục DICOM nguồn, masks, checkpoint, môi trường; kích thước `data/lung_masks/` cần kiểm tra thực tế. Raw DICOM từng được ghi nhận khoảng **3,99 GB**, nhưng **đây không phải tổng dung lượng toàn dự án**.

**Không bắt buộc:** Docker, PostgreSQL, Jupyter hay một IDE cụ thể. Có thể dùng Cursor/VS Code chỉ để mở project và terminal.

### 2.2. Kiểm tra trước khi cài

Mở PowerShell:

```powershell
py -0p                  # Liệt kê Python hiện có
py -3.12 --version      # Kỳ vọng Python 3.12.x
nvidia-smi              # Chỉ cần nếu bạn có GPU NVIDIA
```

Nếu `py -3.12` không tồn tại, hãy cài Python 3.12 trước. **Không tự đổi sang Python 3.13/3.14** chỉ vì là bản mới nhất khi mục tiêu là tái kiểm đúng môi trường nghiên cứu.

---

## 3. Tải dự án và khôi phục đúng cây thư mục

### 3.1. Tải các thành phần từ Drive

Bộ chia sẻ lý tưởng gồm:

- **`01_SOURCE_R5`** — `code/`, `tests/`, `scripts/`, `reproducibility/`, `docs/`, `README.md`, `run_pipeline.sh`, `requirements.txt`, **`requirements.research.txt`**, `DRIVE_UPLOAD_GUIDE.md` nếu có.
- **`02_DATA`** — `data/processed/` (CSV frozen + split SHA), hai file JSON adjudicated, `data/raw/RSNA Pneumonia/raw/` chứa DICOM và `data/lung_masks/train/` chứa masks (nếu đã được phép chia sẻ).
- **`03_MODELS`** — `models/pretrained/pspnet_chestxray_best_model_4.pth` và ba thư mục `results/runs/` chứa checkpoint/config/training logs.
- **`04_RESULTS`** — `results/qa/`, `results/evaluation/`, `results/reports/`, `results/logs/` và các artifact đã công bố.

Đây là **cách chia gói đề xuất**; tên ZIP trên Drive có thể khác. Giải nén và **gộp theo cây thư mục con** vào **một** project root. Tránh tình trạng `bgrel/bgrel/code` hoặc `bgrel/01_SOURCE_R5/code`.

### 3.2. Cây thư mục phải trông như sau

```text
D:\Research\bgrel\
├── README.md                         # Hướng dẫn hiện tại
├── DRIVE_UPLOAD_GUIDE.md             # Checklist quyền chia sẻ (nếu có)
├── requirements.txt                  # Dependency P0 lịch sử, không đủ cho mọi bước
├── requirements.research.txt         # Dependency tham chiếu cho R5 trên Windows
├── run_pipeline.sh                   # Runbook R5 + kiểm tra an toàn
├── code\                             # 29 module Python R5
│   ├── audit_data.py                 # P0
│   ├── split_patients.py             # P1
│   ├── paths.py                      # Resolver đường dẫn immutable
│   ├── dataset.py, model.py          # DICOM preprocessing + MobileNetV2
│   ├── regions.py, losses.py         # Fractional ROI + REL/BCE
│   ├── mask_qa.py, mask_policy.py    # P2 QA + policy
│   ├── mask_pilots.py                # P2 pilot
│   ├── train_baseline.py             # B0 paper recipe
│   ├── train_t2a_noaug.py            # No augmentation T2-A, KHÔNG phải B0 paper
│   ├── train_bgrel.py                # A3 training
│   ├── train_ablations.py           # B1/A2 training
│   ├── evaluate_p8a.py               # P8a + bootstrap
│   └── ...                           # Các module R5 còn lại
├── data\
│   ├── raw\
│   │   ├── RSNA Pneumonia\
│   │   │   └── raw\
│   │   │       └── <StudyUID>\<SeriesUID>\<SOPUID>.dcm
│   │   ├── pneumonia-challenge-dataset-mappings_2018.json
│   │   └── pneumonia-challenge-annotations-adjudicated-kaggle_2018.json
│   ├── processed\
│   │   ├── master_manifest.csv
│   │   ├── boxes.csv
│   │   ├── split.csv
│   │   └── split.sha256
│   └── lung_masks\
│       └── train\                    # Derived PSPNet TRAIN masks (*.npz)
├── models\
│   └── pretrained\
│       └── pspnet_chestxray_best_model_4.pth
├── results\
│   ├── qa\
│   ├── runs\
│   │   ├── b0_dev_s42\               # best.pt + config/summary/metrics/log
│   │   ├── b1_boxpos_s42\
│   │   └── a3_l100_s42\
│   ├── evaluation\
│   │   ├── p8a_val\
│   │   ├── b1_val\
│   │   └── final_test\
│   ├── reports\
│   └── logs\
├── scripts\                           # Review/deploy/rollback của project gốc
├── tests\                             # R5 unit/regression tests
├── reproducibility\                   # Historical producers/hashes; KHÔNG XÓA
├── docs\                              # QA / mapping / research evidence
└── .venvs\
    └── bgrel\                         # TỰ TẠO TRÊN MÁY NGƯỜI NHẬN, KHÔNG UPLOAD
```

> **CỰC KỲ QUAN TRỌNG — đường dẫn DICOM:** giá trị trong `master_manifest.csv` được frozen dạng `RSNA Pneumonia/raw/...`. Code `code/paths.py` ánh xạ nó sang `data/raw/RSNA Pneumonia/raw/...` tại thời điểm đọc. **Không sửa CSV frozen để đổi đường dẫn.** Dữ liệu thực tế dùng cấu trúc 3 cấp StudyUID/SeriesUID/SOPUID đúng như cây trên.
>
> `D:\Research\bgrel` chỉ là **ví dụ cho máy mới**, **không** phải bắt buộc cài vào `C:\Users\Abc\Downloads\bgrel` như máy gốc.

### 3.3. Kiểm tra file sau khi giải nén (PowerShell)

```powershell
$root = 'D:\Research\bgrel'   # ĐỔI thành thư mục THẬT của bạn
Set-Location -LiteralPath $root

Test-Path '.\code\model.py'                                     # True
Test-Path '.\code\mask_qa.py'                                  # True
Test-Path '.\tests'                                            # True
Test-Path '.\data\processed\split.csv'                        # True nếu đã khôi phục DATA
Test-Path '.\data\raw\RSNA Pneumonia\raw'                     # True nếu có DICOM
Test-Path '.\models\pretrained\pspnet_chestxray_best_model_4.pth'  # True nếu có MODELS
Test-Path '.\results\runs\a3_l100_s42\best.pt'                # True nếu có MODELS/RUNS
```

Có thể dùng **7-Zip** hoặc Windows Extract All để giải nén. Với một dataset DICOM nặng, nên kiểm tra đủ số file và hash; không chỉ nhìn thấy thư mục là kết luận đã tải đủ. **Không đổi tên các JSON, `.dcm`, `.npz`, run IDs hay `best.pt`**.

### 3.4. Tại sao không copy `.venvs/` từ máy tác giả?

Venv thường chứa đường dẫn tuyệt đối, wheel gắn kiến trúc máy và PyTorch/CUDA build. Hãy **tạo lại venv trên máy người nhận** theo mục 4; không upload `.venvs/` lên Drive.

---

## 4. Tạo Python 3.12 venv và cài dependency

### 4.1. Tạo venv (PowerShell — chạy từ project root)

```powershell
$root = 'D:\Research\bgrel'        # ĐỔI đường dẫn theo máy bạn
Set-Location -LiteralPath $root

py -3.12 -m venv '.venvs\bgrel'
$py = Join-Path $root '.venvs\bgrel\Scripts\python.exe'
& $py --version
& $py -m pip --version
```

Không cần dùng `Activate.ps1`; gọi thẳng `& $py` tránh lỗi *execution policy*. Không tạo thêm một `.venv` ở nơi khác vì `run_pipeline.sh` **chỉ tìm `.venvs/bgrel`**.

**Nếu `.venvs\bgrel` đã có dữ liệu:** không xóa/gộp chồng venv cũ khi chưa biết đó có phải môi trường nghiên cứu đang hoạt động hay không. Những lệnh cài sau đây dành cho **thư mục tải về trên máy mới**.

### 4.2. Cài PyTorch trước (chọn ĐÚNG MỘT phương án)

**A — Có GPU NVIDIA, muốn sát môi trường gốc (CUDA 12.8):**

```powershell
& $py -m pip install torch==2.9.1 torchvision==0.24.1 `
    --index-url https://download.pytorch.org/whl/cu128
```

**B — Không có GPU NVIDIA / muốn kiểm thử CPU:**

```powershell
& $py -m pip install torch==2.9.1 torchvision==0.24.1 `
    --index-url https://download.pytorch.org/whl/cpu
```

Hai phiên bản và chỉ mục wheel trên được đối chiếu với [PyTorch official previous versions: v2.9.1](https://pytorch.org/get-started/previous-versions/). **Không cài cả A và B trên cùng venv.** Bản CPU không được xem là tương đương về tốc độ/số học BF16 cho việc tái train.

### 4.3. Cài thư viện còn lại

Gói tài liệu này có file **`requirements.research.txt`**, ghim các phiên bản top-level từ `meta/ENVIRONMENT_VERSIONS.json` trong evidence Windows. Đây là **bản tham chiếu**, **không phải** full hash-locked environment hay cài đặt fresh đã được nghiệm thu.

```powershell
# Chạy ở project root, với $py lấy từ mục 4.1
& $py -m pip install -r '.\requirements.research.txt'
& $py -m pip check
```

**Danh sách top-level ghi nhận trên máy gốc:**

| Gói | Phiên bản | Vai trò |
|---|---|---|
| `torch`, `torchvision` | 2.9.1+cu128, 0.24.1+cu128 | Model / GPU / tensor (cài riêng §4.2) |
| `numpy` | 1.26.4 | Array / DICOM |
| `pydicom` | 3.0.1 | DICOM reader |
| `Pillow` | 11.3.0 | Ảnh / JPEG decoding / QA overlays |
| `pandas` | 2.3.3 | Bảng QA/analysis, một số dependencies |
| `scipy` | 1.17.1 | Thống kê, image processing |
| `scikit-learn` | 1.7.2 | AUROC / AP / metrics |
| `scikit-image` | 0.26.0 | Image processing / TorchXRayVision dependencies |
| `torchxrayvision` | 1.5.7 | PSPNet lung segmentation |
| `tqdm` | 4.70.1 | Progress/provenance dependencies |

File gốc `requirements.txt` hiện **chỉ chứa** `numpy==1.26.4`, `pydicom==3.0.1`, `Pillow==11.3.0` cho P0. Do đó **không** dùng riêng `pip install -r requirements.txt` rồi kỳ vọng mọi bước hoạt động. Để giữ nguyên lịch sử, file gốc **không bị thay**; file mới có tên khác.

### 4.4. Kiểm tra Python, CUDA và import

Chạy trong PowerShell (không cần kích hoạt venv):

```powershell
& $py -c "import torch, torchvision, numpy, pydicom, PIL, pandas, scipy, sklearn, torchxrayvision; print('IMPORTS_OK'); print('torch=', torch.__version__); print('torchvision=', torchvision.__version__); print('CUDA_BUILT=', torch.version.cuda); print('CUDA_AVAILABLE=', torch.cuda.is_available()); print('GPU=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
& $py -m pip check
```

**Diễn giải kết quả:**

- `IMPORTS_OK` + `pip check` PASS: các import chính đã có, nhưng **chưa** chứng minh inference/segmentation chạy đúng trên ảnh DICOM thật.
- CUDA build `12.8` + `CUDA_AVAILABLE=True`: PyTorch nhìn thấy GPU tương thích; vẫn phải chạy smoke/test thực tế.
- `CUDA_AVAILABLE=False`: kiểm tra `nvidia-smi`, driver, và bạn đã chọn wheel `cu128` hay `cpu`; không giả định rằng GPU chỉ cần xuất hiện trong Device Manager là đã hoạt động.
- Nếu mục tiêu chỉ là kiểm tra **archived paper outputs**, có thể thử CPU; nếu định train như báo cáo, môi trường GPU BF16 là yêu cầu có ý nghĩa thực nghiệm.

### 4.5. Vì sao vẫn cần ghi lại `pip freeze`?

Sau khi cài thành công, lưu thông tin môi trường vào **máy của người nhận** (không ghi đè `requirements.txt` đã công bố):

```powershell
New-Item -ItemType Directory -Force '.\local_setup' | Out-Null
& $py -m pip freeze | Set-Content -Encoding UTF8 '.\local_setup\pip_freeze.txt'
& $py -m pip check  | Set-Content -Encoding UTF8 '.\local_setup\pip_check.txt'
```

Không upload file trong `local_setup/` nếu chứa path nội bộ/metadata nhạy cảm; đó là log của máy cá nhân. `requirements.research.txt` **không pin toàn bộ transitive dependencies** nên `pip freeze` là thông tin cần để báo lỗi có thể tái lập.

---

## 5. Tải/kiểm tra pretrained ImageNet và PSPNet

Có **hai nhóm trọng số khác nhau**:

1. **PSPNet `.pth` (≈273 MB trong evidence)** nằm trong project: `models/pretrained/pspnet_chestxray_best_model_4.pth`. Cần cho P2/P3/những bước regional evaluation liên quan. **Không** tự đổi tên hoặc thay bằng checkpoint PSPNet khác.
2. **MobileNetV2 ImageNet weights** do `torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1` sử dụng. Source `code/model.py` tìm trong **Torch Hub cache của từng người dùng**, không nhất thiết nằm trong `models/`. Checkpoint B0/B1/A3 đã lưu **không đồng nghĩa** với việc ImageNet cache đã có cho những preflight/train cần pretrained.

### 5.1. Tải ImageNet weights bằng torchvision (chỉ cần lần đầu)

Trong PowerShell, ở project root, với `$py` từ §4.1:

```powershell
& $py -c "from torchvision.models import mobilenet_v2, MobileNet_V2_Weights; mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1); print('IMAGENET_CACHE_READY')"
```

Lệnh này **tải/đọc pretrained weights**, không tải RSNA, không huấn luyện hoặc truy cập TEST. Khi offline, người nhận cần cache weights đúng từ nguồn hợp lệ trước; **không tải checkpoint ngẫu nhiên rồi sửa expected SHA**.

### 5.2. Check hash pretrained weights

Recorded SHA của MobileNetV2 ImageNet từ `results/runs/b0_dev_s42/config.json`:

```text
b03531047ffacf1e2488318dcd2aba1126cde36e3bfe1aa5cb07700aeeee9889
```

Kiểm tra bằng chính `code/model.py`, chạy từ project root:

```powershell
& $py -c "import sys,hashlib; sys.path.insert(0,'code'); from model import imagenet_weight_path; p=imagenet_weight_path(); print('PATH=',p); print('EXISTS=',p.is_file()); print('SHA256=',hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING')"
```

SHA phải khớp bản config nếu định sử dụng **đúng pretrained initialization** của research. Không phải mọi đường inference dùng pretrained weights vì model có thể load từ `best.pt`.

### 5.3. Kiểm tra trọng số PSPNet

```powershell
Get-FileHash -Algorithm SHA256 '.\models\pretrained\pspnet_chestxray_best_model_4.pth'
```

SHA mong đợi:

```text
019b167eac6b729fc1bb92bbbc185fc1730aaa65819f4e3fe718186cadc044fc
```

Nếu khác SHA: tải lại đúng file, kiểm tra quá trình truyền/giải nén và quyền phân phối; **không** thay hash trong source/manifest.

---

## 6. Kiểm tra installation và các kết quả đã lưu

### 6.1. Kiểm tra cây thư mục (Git Bash)

Mở **Git Bash** tại thư mục project và chạy:

```bash
bash run_pipeline.sh help
bash run_pipeline.sh status
bash run_pipeline.sh assets
```

- `help`: liệt kê các chức năng Bash thực tế.
- `status`: báo source theo dạng R5 style, đếm module Python, kiểm tra file. **Không xác minh checksum của source**.
- `assets`: in danh sách các file/thư mục người tải cần có; `MISSING` giúp xác định lỗi giải nén/thiếu gói.

**Lưu ý:** Các lệnh này dùng Python bên trong `.venvs/bgrel`; tạo venv đúng tên và cài dependency trước. Nếu chỉ mở PowerShell, có thể chạy Git Bash qua giao diện *Git Bash Here*, không cần tự viết chuyển đổi cú pháp Windows ↔ Bash.

### 6.2. Xác minh frozen hash và archived results

Chỉ chạy **khi đã tải đầy đủ SOURCE + DATA + MODELS + RESULTS**:

```bash
bash run_pipeline.sh verify
```

Hành động thực tế của `verify`:

- Tính SHA-256 của `master_manifest.csv`, `split.csv`, `boxes.csv`, **PSPNet pretrained** và 3 checkpoint `best.pt`.
- Kiểm `split.sha256`, 29.684 image IDs, phân chia TRAIN/VAL/TEST, 12.233 bệnh nhân không trùng split.
- Kiểm **sự hiện diện** của đường DICOM và 2 JSON; **không decode DICOM** trong chế độ này.
- Kiểm các metrics JSON đã lưu cho VAL/TEST và các evaluation/protocol manifests quan trọng.
- Xuất dòng `VERIFY_SUMMARY failures=0 test_pixels_opened=False training_started=False` nếu hoàn tất không có lỗi.

`verify` **không chứng minh** chính xác mọi byte của 30.000 DICOM/mask vì hiện tại nó không hash toàn bộ hai cây đó. Khi muốn kiểm chứng nguyên vẹn toàn bộ asset giữa hai máy, cần một **file manifest SHA-256 đầy đủ** tạo ở máy nguồn và đối chiếu sau khi tải về; hiện không có manifest toàn bộ DICOM/masks nào được xác nhận trong bản phát hành này.

### 6.3. Chạy R5 unit tests

```bash
bash run_pipeline.sh tests
```

Mục tiêu: unit/parity/regression tests trong `tests/`. Tại máy tác giả, **61/61 tests đã PASS** sau triển khai R5. Trên máy mới có thể khác vì phiên bản package, OS, đường dẫn hoặc asset chưa được phục hồi.

### 6.4. Kiểm tra một lần sau khi khôi phục đầy đủ

```bash
bash run_pipeline.sh check
```

`check` = `status` → `verify` → `tests`; sẽ dừng nếu có bước FAIL. **Không train lại, không mở TEST pixels**. Một số test có thể tạo file tạm trong temp directory; không nên hiểu đây là lệnh tạo lại thí nghiệm.

### 6.5. Verify riêng archived final TEST

Chạy trong PowerShell tại project root:

```powershell
& $py '.\code\evaluate_final_test.py' --verify-output
```

Đây là **kiểm tra đầu ra P9 đã có**, không phải tái inference từ ảnh TEST. **Không** thay `--verify-output` bằng `--run` trên project frozen.

### 6.6. Nguyên tắc PASS/FAIL

- **PASS sau `verify`, `tests`**: bộ artifacts đủ và source/tests phù hợp với **kiểm tra được thực thi**.
- **Không đồng nghĩa** train ra đúng checkpoint theo byte hoặc reproduction hoàn toàn từ máy trắng.
- **`MISSING` / `FAIL` sau khi chỉ tải SOURCE**: bình thường; hãy tải DATA/MODELS/RESULTS và khôi phục đúng đường dẫn trước khi kết luận code hỏng.
- **Hash mismatch**: không sửa expected hash hay chạy lại P0/P1; kiểm tra file copy sai, package thiếu, giải nén trùng cấp hoặc phiên bản khác.

---

## 7. Các chế độ sử dụng và giới hạn thực tế

### A — Chỉ đọc code, thử unit tests

Có thể cài môi trường và dùng `bash run_pipeline.sh tests`. Một số test dựa vào provenance/fixtures; một clone chỉ có code có thể không PASS tất cả. **Không cần tải 30.000 DICOM chỉ để đọc code**.

### B — Khôi phục nguyên bộ kết quả và kiểm chứng paper (**khuyến nghị**)

Tải đầy đủ SOURCE, DATA, MODELS, RESULTS đúng theo §3. Sau đó dùng `status`, `verify`, `tests`, `check` và `evaluate_final_test.py --verify-output`. Đây là luồng **không làm phát sinh train/TEST inference mới**.

### C — Inference với checkpoint đã huấn luyện

`code/model.py` định nghĩa MobileNetV2-OS16 và `results/runs/*/best.pt` là trọng số đã train. Tuy nhiên R5 **chưa cung cấp một CLI công khai, độc lập và đã nghiệm thu** cho việc `predict --dicom <file>` tùy ý. Các `evaluate_* --run` có thể ghi lại archived evaluation và/hoặc truy cập TEST; **không dùng để thử cài đặt**. Muốn xây API inference trên ảnh người dùng mới cần một bước triển khai và validation riêng.

### D — Fresh full P0–P9 training

**Chưa được đóng gói như workflow `git clone && train-all`.** Source R5 có nhiều CLI thật nhưng còn các guard dựa trên `run_id`, source SHA, QA protocol, source lịch sử và split seal. Để làm nghiên cứu mới, cần:

1. Sao chép SOURCE và raw input vào **một root độc lập**; không dùng lại `results/` frozen làm thư mục outputs của thí nghiệm mới.
2. Chốt trước protocol, split seed, nhãn, tham số, phiên bản packages, nguyên tắc phê duyệt QA, cách đặt run IDs mới.
3. Chạy từng phase theo §8 và `bash run_pipeline.sh plan`; xem `--help` để kiểm tra flag, **không copy mù lệnh có tác động ghi**.
4. Khi source guard chặn vì khác historical hash, **không sửa manifest/hash cũ để bypass**. Cần một phiên bản fresh-run runner được thiết kế và kiểm thử riêng cho protocol mới.
5. Giữ TEST sealed; mở TEST mới chỉ khi pre-registration/approvals và điều kiện cuối cùng đã hoàn thành. Không làm nhiều lần để lựa chọn model theo TEST.

### Vì sao có source lịch sử trong `reproducibility/`?

Code R5 hiện đại khác source tại thời điểm train paper. `reproducibility/` giữ code gốc, bản SHA đã xác minh và các snap R3/R4 để chứng minh **checkpoint đã được tạo từ đâu**. Không xóa để giảm số file; càng không thực thi source cũ tự động vào các output đã frozen. Việc khôi phục source không phải bằng chứng đã replay training trong environment cũ.

---

## 8. Giải thích P0–P9: input, output và thứ tự

> **Bảng này mô tả workflow và lệnh đã tồn tại**; không bảo đảm mọi bước có thể chạy liền nhau trong một fresh workspace mà không có QA/approval/migrations phù hợp.

| Bước | Source thực tế (R5) | Đầu vào cần có | Kết quả / gate |
|---|---|---|---|
| **P0 Audit** | `code/audit_data.py` | 30.000 DICOM RSNA + 2 JSON | `master_manifest.csv`, `boxes.csv`, audit logs; cohort 29.684 |
| **P1 Split** | `code/split_patients.py` | Frozen P0 tables | `split.csv` + `split.sha256` sau preview/review/freeze; patient-disjoint |
| **P2 Segmentation QA** | `generate_lung_masks.py`, `mask_pilots.py`, `mask_qa.py`, `mask_policy.py`, `build_train_lung_masks.py` | DICOM TRAIN, PSPNet `.pth`, P0/P1 QA | Pilot masks, overlays, QA200, policy, full TRAIN masks; QA/manual gate |
| **P3 Matching** | `sample_controls.py` | TRAIN lesion boxes, masks, policy | Frozen controls/sample trace; up to 2 region references/lesion |
| **P4–P5 B0** | `model.py`, `regions.py`, `losses.py`, `train_baseline.py` | ImageNet weights, TRAIN/VAL, frozen split | Image-only BCE baseline B0; `b0_dev_s42/best.pt` (paper run) |
| **P6 A3** | `p6_preflight.py`, `train_bgrel.py`, `rel_adapter.py`, `training_objectives.py` | B0 recipe, P3 controls, approval + TRAIN/VAL | BCE + REL checkpoint A3 (`a3_l100_s42`) |
| **P7 B1** | `train_ablations.py`, `training_objectives.py` | B0 recipe, positive TRAIN boxes | BCE + box-positive BCE checkpoint B1 (`b1_boxpos_s42`) |
| **P8a VAL** | `evaluate_p8a.py`, `evaluate_b1_val.py` | VAL + 3 checkpoints | AUROC, AP, 2.000 resample patient-bootstrap |
| **P8b Regions** | `evaluate_regions.py`, `evaluate_b1_regions.py`, `evaluate_lesion.py` | VAL, expert lesions, matched regions | REM, Top-1, blur RIE, mean-fill RIE và paired comparisons |
| **P9 TEST** | `evaluate_final_test.py` | 3 locked best checkpoints + freeze protocol | Frozen internal TEST once; sau đó **chỉ verify archive** |

### 8.1. Lệnh xem toàn bộ thứ tự (không thực thi)

```bash
bash run_pipeline.sh plan
```

R5 `plan` hiển thị đầy đủ lệnh với flag thật. Trong project đã hoàn thành, **chỉ đọc phần plan**; không thực thi các dòng `--freeze`, `--generate`, `--train` hoặc `--run` trên data đã frozen.

### 8.2. Preflight thủ công (có thể ghi log/QA)

```bash
bash run_pipeline.sh preflight P2
bash run_pipeline.sh preflight P3
bash run_pipeline.sh preflight P5
bash run_pipeline.sh preflight P6
bash run_pipeline.sh preflight P7
bash run_pipeline.sh preflight P8
bash run_pipeline.sh preflight P9
```

**Không chạy tất cả liên tục theo checklist nếu chỉ muốn kiểm tra cài đặt.** Một số preflight source hiện hành có thể ghi bổ sung log/QA; các bước P0/P1 bị chặn trong workspace frozen. P9 preflight của runbook thực chất gọi `--verify-output`, không decode TEST pixels.

---

## 9. Lệnh CLI của 29 module R5

Các file chỉ cung cấp logic import (ví dụ `model.py`, `dataset.py`, `losses.py`, `regions.py`, `paths.py`, `metrics.py`, `rel_adapter.py`, `training_objectives.py`, `refactor_helpers.py`) không cần chạy độc lập. Danh sách entrypoint có CLI trong bản R5 đang xét:

| Tác vụ | Câu lệnh xem trợ giúp (không ghi output nghiên cứu) |
|---|---|
| P0 | `python code/audit_data.py --help` |
| P1 | `python code/split_patients.py --help` |
| P2 smoke | `python code/generate_lung_masks.py --help` |
| P2 pilot | `python code/mask_pilots.py --help` |
| P2 QA | `python code/mask_qa.py --help` |
| P2 policy | `python code/mask_policy.py --help` |
| P2 full masks | `python code/build_train_lung_masks.py --help` |
| P3 | `python code/sample_controls.py --help` |
| B0 | `python code/train_baseline.py --help` |
| T2-A (không thuộc paper model B0) | `python code/train_t2a_noaug.py --help` |
| A3 prerequisite | `python code/p6_preflight.py --help` |
| A3 | `python code/train_bgrel.py --help` |
| B1 / A2 | `python code/train_ablations.py --help` |
| VAL P8a | `python code/evaluate_p8a.py --help` |
| VAL B1 | `python code/evaluate_b1_val.py --help` |
| VAL regions | `python code/evaluate_regions.py --help` |
| VAL B1 regions | `python code/evaluate_b1_regions.py --help` |
| VAL lesion diagnostics | `python code/evaluate_lesion.py --help` |
| Final TEST | `python code/evaluate_final_test.py --help` |

Trong PowerShell, thay tiền tố `python` ở trên bằng `& $py`, ví dụ:

```powershell
& $py '.\code\train_baseline.py' --help
& $py '.\code\evaluate_final_test.py' --help
```

Nếu CLI `--help` cũng bị lỗi vì thiếu import, quay lại §4 và xem traceback; **không sửa SHA hoặc data** để xử lý lỗi dependency.

---

## 10. Kiến trúc và phương pháp nghiên cứu

### 10.1. Pipeline tổng quan

```text
RSNA 30,000 DICOM + 2 JSON adjudicated
                  |
            P0 DATA AUDIT
                  |
           P1 PATIENT SPLIT
           TRAIN / VAL / TEST
                  |
      +-----------+---------------------------+
      |                                       |
  IMAGE PIPELINE                         P2 PSPNet MASKS
  MobileNetV2-OS16                       QA / POLICY
      |                                       |
  IMAGE BCE                                P3 MATCHING
      |                                  lesion -> K=0/1/2 controls
      +-------------+-------------------------+
                    |
             TRAINING OPTIONS
           B0 image BCE only
           B1 image + box BCE
           A3 image + listwise REL
                    |
              P8 VALIDATION
      AUROC/AP + REM/Top-1/RIE/bootstrap
                    |
            P9 FROZEN TEST
          AUROC/AP & paired CI
```

### 10.2. Những thành phần model quan trọng

- **Image preprocessing:** đọc DICOM, xác minh SOP/Patient IDs và ảnh 1024×1024; min-max chuẩn hóa từng ảnh, đảo MONOCHROME1 khi cần; resize bilinear 224×224; expand thành 3 channel; ImageNet normalization.
- **Backbone:** MobileNetV2 với output stride 16; feature tensor 1280×14×14 ở input 224×224; dùng cùng learned affine scoring head cho image và ROI.
- **Region pooling:** fractional area-weighted pooling từ feature grid, không phải crop pixel độc lập; receptive field có thể vượt ngoài bounding box.
- **PSPNet lung segmentation:** dùng pretrained weights để chọn vùng tham chiếu/lung occupancy; **không cần ở thời điểm image-only inference**.
- **Control sampling:** cùng ảnh, K≤2; lung occupancy ≥0.80, normalized vertical alignment tolerance ≤0.10, IoU giữa controls ≤0.20, 128 attempts per tier, grid step 8, tránh annotated lesions, không dùng classifier score để chọn control.
- **A3 REL:** với lesion L, controls C1..CK, dùng loss `logsumexp(sL, sC1, …) − sL`; trung bình loss theo lesion đủ điều kiện trong từng ảnh, sau đó theo ảnh; K=0 không tham gia REL nhưng vẫn tính image BCE.
- **B1:** thêm BCE target positive cho expert lesion boxes, không cần sampler controls; **khác A3**.
- **T2-A:** baseline *no augmentation* riêng, **không** được gắn nhãn B0 paper.

### 10.3. Hyperparameters ghi nhận

| Tham số | Giá trị trong paper |
|---|---|
| Image size | 224×224 |
| Optimizer | AdamW |
| Learning rate | 3e-5 |
| Weight decay | 1e-3 |
| Batch size | 4 |
| Epoch cap | 20 |
| Early stopping | patience 4, min ΔAUROC 0.001 |
| Schedule | CosineAnnealingLR / optimizer step |
| Training precision | BF16 autocast |
| Photometric augmentation | brightness, contrast độc lập trong [0.9, 1.1] |
| Seed | 42 |
| A3 auxiliary max lambda | 0.10 |
| A3/B1 lambda schedule | epoch 1: 0.02; epoch 2: 0.06; từ epoch 3: 0.10 |
| Best reported epoch B0/B1/A3 | 3 |
| Statistical bootstrap | 2,000 paired patient-cluster resamples, seed 42 |

Vui lòng tham khảo source và `results/runs/*/config.json` khi đối chiếu một run cụ thể. Thay đổi phiên bản library, BF16, driver, batch hay RNG có thể khiến run mới khác từng bit kể cả khi sử dụng cùng recipe.

---

## 11. Dataset, split và kết quả paper

### 11.1. Dataset

- Data gốc: **RSNA Pneumonia Detection Challenge**, nhưng nhãn dự đoán nghiên cứu là **Lung Opacity**, không phải diagnosis xác nhận.
- **30.000** DICOM được audit; cohort hợp lệ frozen là **29.684** ảnh từ **12.233** bệnh nhân.
- Mapping/annotation từ hai file JSON adjudicated năm 2018, **tên và nội dung phải giữ nguyên** để khớp source.
- Partition **patient-disjoint**, split tỷ lệ gần 80/10/10. TRAIN 23.748, VAL 2.969, TEST 2.967 ảnh.

| Split | Số ảnh | Số bệnh nhân | Positive images | Expert boxes |
|---|---:|---:|---:|---:|
| TRAIN | 23.748 | 9.786 | 5.685 | 9.161 |
| VAL | 2.969 | 1.223 | 711 | 1.122 |
| Internal TEST | 2.967 | 1.224 | 710 | 1.107 |
| **Tổng** | **29.684** | **12.233** | **7.106** | **11.390** |

### 11.2. Image-level discrimination

| Model | VAL AUROC | VAL AP | TEST AUROC | TEST AP |
|---|---:|---:|---:|---:|
| B0 (BCE) | 0.880948 | 0.703314 | 0.878775 | 0.717998 |
| B1 (BCE + box BCE) | 0.881258 | 0.706462 | 0.877135 | 0.709789 |
| A3 (BCE + REL) | 0.881102 | 0.700887 | 0.878496 | 0.712560 |

Đây là internal test, **không** phải external validation. AUROC primary; AP là *non-interpolated average precision*. Các khoảng CI của chênh lệch AUROC paired TEST đều chứa 0, nên **không khẳng định A3 cải thiện khả năng phân loại trên TEST**.

### 11.3. Regional evidence — CHỈ VAL

| Model | REM ↑ | Lesion-first Top-1 ↑ | Blur RIE ↑ | Mean-fill RIE ↑ |
|---|---:|---:|---:|---:|
| B0 | 1.652933 | 61.92% | 0.160801 | 0.199302 |
| B1 | 1.796636 | 67.44% | 0.168817 | 0.199618 |
| A3 | 3.074698 | 76.45% | 0.176137 | 0.207397 |

Chỉ **344 / 1.122** expert lesions trên VAL có ít nhất một control đạt tiêu chí; từ 308 ảnh và 168 bệnh nhân. Chênh lệch **patient-averaged A3 − B1 REM = +1.147613**, 95% paired patient-cluster bootstrap CI `[0.763703, 1.513700]`. Không được diễn giải controls chưa được annotate là mô lành đã được bác sĩ chứng thực, hay REM là bằng chứng nhân quả model nhìn đúng vùng bệnh.

---

## 12. Provenance, checkpoint SHA và tính toàn vẹn

Các SHA sau được kiểm chứng trên máy gốc trong review đã cung cấp:

| File | SHA-256 |
|---|---|
| `data/processed/master_manifest.csv` | `4c690ddf7918d0890b6715dba290be85a16959adc0dcaa70b5b4c5bd387d6c9b` |
| `data/processed/split.csv` | `ccabcf432b0d266264e4ea96ca018b05a67d944af65fca4ad159f5130da7f9a7` |
| `data/processed/boxes.csv` | `6372a1facc3e1b78fbcd97a5bd103d2fc33ed257c92f8076b1d36e9168078e45` |
| `models/pretrained/pspnet_chestxray_best_model_4.pth` | `019b167eac6b729fc1bb92bbbc185fc1730aaa65819f4e3fe718186cadc044fc` |
| `results/runs/b0_dev_s42/best.pt` | `f6bda8a74c5d72b948a9ee4ba35cf53c1c0176e7d15dbb457ecc78a4d79f2929` |
| `results/runs/b1_boxpos_s42/best.pt` | `f1b2ab19211d22dc4ebe53cb978d36e15fa436b58e16f389954f5d2da0e688f1` |
| `results/runs/a3_l100_s42/best.pt` | `7bd87ba3841f0b8c8287a859dc3c942027e6fb318b2d5122dbfe1321bca2743f` |

Trên Windows có thể kiểm riêng một file với `Get-FileHash -Algorithm SHA256 <path>`, hoặc dùng `bash run_pipeline.sh verify` để kiểm đồng loạt.

**Provenance R5:**

- 13/13 historical source SHA được tìm thấy trong `reproducibility/historical_verified/`; có source gốc B0 và T2-A riêng.
- `docs/` có nhiều báo cáo candidate/review *trước khi deploy*; chúng là lịch sử, **không** phủ định log triển khai thực tế `R5_SOURCE_ONLY_CUTOVER_PASS`.
- Không chỉnh `config.json`, `protocol_freeze.json`, `split.sha256`, SHA trong manifests để biến FAIL thành PASS.
- Source SHA của script R5 **không nhất thiết bằng** SHA của script đã tạo checkpoint paper. Đó là lý do phải giữ historical source, không sao chép lại `source_sha256` của mã mới lên run cũ.

---

## 13. Chia sẻ trên Google Drive/GitHub

### 13.1. Những gì người tải phải có

| Thư mục/file | Để đọc code | Để `verify` FULL | Để fresh train |
|---|---:|---:|---:|
| `code/`, `scripts/`, `tests/`, `reproducibility/`, `docs/` | Có | Có | Có |
| `README.md`, `run_pipeline.sh`, `requirements.research.txt`, `requirements.txt` | Có | Có | Có |
| `data/processed/` 4 file frozen | Không | **Có** | Tùy reconstruction protocol; thường cần |
| `data/raw/` hai JSON adjudicated | Không | **Có** | **Có** |
| `data/raw/RSNA Pneumonia/raw/` DICOM | Không | **Có ít nhất đúng thư mục** | **Có đầy đủ ảnh** |
| `data/lung_masks/train/` | Không | Không phải tất cả `verify` đều hash | A3/P3 cần hoặc phải tái tạo qua P2 |
| `models/pretrained/` PSPNet | Không | **Có** | **Có cho P2** |
| `results/runs/` ba published checkpoints | Không | **Có** | Có thể train mới (không bảo đảm tái tạo từng byte) |
| `results/qa/`, `results/evaluation/` | Không | **Có** | Là frozen reference; đừng ghi đè |

**Không upload:** `.venvs/`, `.git/` nếu chia sẻ ZIP thuần, `__pycache__/`, `.pytest_cache/`, `*.pyc`, env/token/secrets, cache/backup ngoài release, file tmp/partial. **Không xóa** các archived results và historical source chỉ vì nhìn trùng lặp.

### 13.2. Lưu trữ riêng tư và chia sẻ công khai là hai việc khác nhau

- **Drive riêng tư:** có thể backup đầy đủ những file bạn có quyền lưu và đồng bộ. Chú ý kiểm soát người được cấp quyền.
- **Drive công khai / GitHub công khai:** chỉ phát hành dữ liệu DICOM, hai JSON, pretrained weights, patient-level predictions và checkpoint khi có cơ sở pháp lý/quyền phân phối/đánh giá riêng tư phù hợp. Một cách an toàn hơn là chỉ chia sẻ code, bảng số liệu tổng hợp và hướng dẫn tải dataset từ nguồn chính thức theo quy định.
- **Nguồn chính thức RSNA:** [Kaggle RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) và [Rules](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules). Điều khoản cho phép mục đích nghiên cứu **không tự động đồng nghĩa** cho phép đưa mọi bản sao DICOM/raw metadata lên public Drive; người chia sẻ phải kiểm tra quyền của tài nguyên cụ thể.

### 13.3. Người nhận cần giải nén thế nào?

Bản SOURCE, DATA, MODELS, RESULTS có thể upload thành các ZIP riêng để dễ chuyển. Người nhận **giải nén tất cả vào cùng một root** và giữ đúng relative paths, thay vì để 4 ZIP trong 4 thư mục con rồi chạy Python từ nơi khác.

File `DRIVE_UPLOAD_GUIDE.md` (nếu được cung cấp) mô tả checklist quyền chia sẻ và phân nhóm artifacts; dùng cùng README này khi bàn giao.

---

## 14. Các lỗi phổ biến và cách xử lý

| Hiện tượng | Khả năng nguyên nhân | Cách xử lý an toàn |
|---|---|---|
| `py` / `py -3.12` không tồn tại | Chưa cài Python 3.12/Launcher | Cài bản x64, khởi động terminal mới; kiểm `py -0p` |
| `No .venvs/bgrel Python` | Venv đặt sai tên/vị trí | Tạo `py -3.12 -m venv .venvs\bgrel` tại root; đừng copy venv cũ |
| `pip check` FAIL | Dependency conflicting | Xem package và phiên bản chính xác, cài lại trong venv **mới**; không nâng cấp package tùy tiện trên máy nghiên cứu gốc |
| `ModuleNotFoundError: torchxrayvision` | Chỉ cài `requirements.txt` P0 | Cài thêm `requirements.research.txt` và kiểm wheel PyTorch tương thích |
| `CUDA_AVAILABLE=False` | Chọn CPU wheel, driver/NVIDIA CUDA không sẵn | Xem `nvidia-smi` và torch version, làm lại venv CUDA nếu cần |
| `MobileNetV2 weights not cached` | Torch Hub cache chưa có ImageNet V1 | Thực hiện §5.1 và kiểm SHA; không dùng weights khác |
| `FileNotFoundError` với `.dcm` | DATA chưa đủ hoặc giải nén sai cấp | Kiểm `data/raw/RSNA Pneumonia/raw/<Study>/<Series>/<SOP>.dcm` |
| `DICOM pixel data decoding failed` | JPEG Baseline decoder hoặc transfer syntax không được hỗ trợ | Kiểm Pillow/pydicom đúng bản, dùng một mẫu TRAIN/VAL để chẩn đoán; đừng chuyển đổi raw DICOM bất biến |
| `SHA256 mismatch` | File sai phiên bản/copy hỏng | Tải lại đúng asset, đối chiếu hash, **không** sửa expected SHA |
| `P0 BLOCKED: frozen split seal` | Research workspace đã frozen | **Đúng như thiết kế**. Không `--overwrite`; reconstruction phải có root mới |
| `SOURCE_LAYOUT=UNKNOWN_OR_INCOMPLETE` | Giải nén source R3/R4 hoặc thiếu file R5 | Khôi phục đúng SOURCE R5; kiểm `mask_qa.py`, `mask_pilots.py`, `train_t2a_noaug.py` |
| `scripts/INSTALL.ps1` từ chối source | Nó dành riêng cutover R3/R4 → R5, không phải fresh setup | **Không bypass SHA guard**. Cài venv theo README; không chạy migration installer |
| `verify` báo thiếu `results/evaluation/...` | Chưa tải đúng RESULTS / đặt sai đường dẫn | Kiểm danh mục Drive, giải nén `results/` vào đúng root; không chạy evaluator `--run` để “tạo file cho đủ” |
| Unit tests FAIL trên máy mới | Version/path/fixture khác máy tác giả | Giữ log và môi trường versions; phân biệt lỗi package, quyền filesystem, source vs missing artifacts |
| T2-A bị hiểu nhầm B0 | R3 trước đây dùng `train_baseline.py` cho T2-A | R5 có B0 trainer và `train_t2a_noaug.py` tách riêng; kiểm run ID/config thật |
| `pip`/file path có khoảng trắng | Chạy PowerShell thiếu quote | Dùng `Set-Location -LiteralPath`, quotes `'...'` và `& $py` như ví dụ |

**Thông tin cần có khi nhờ người khác debug:** (1) lỗi terminal đầy đủ, (2) Python/PyTorch/CUDA versions, (3) lệnh vừa chạy, (4) `pip check`, (5) `status`/`verify` log. **Không gửi file dữ liệu bệnh nhân, tokens hoặc ảnh y tế riêng tư chỉ để debug.**

---

## 15. Giới hạn tái lập và checklist bàn giao

### 15.1. Đã được kiểm tra trên máy tác giả (07/11/2026)

- R5 source-only deploy: `R5_SOURCE_ONLY_CUTOVER_PASS`, baseline `R4_CLEAN`.
- Windows R5 test report: **61 unit tests PASS**, 19/19 CLI help PASS.
- TRAIN/VAL DICOM/checkpoint smoke, 3 checkpoint SHA + FP32 parity trên 2 VAL images, frozen CSV/split.
- Archived Tables II–VI được đối chiếu từ các predictions lưu sẵn, bao gồm 2.000 patient-cluster bootstrap; không mở TEST pixels trong review.
- 13/13 historical source hashes recovered; deploy/rollback simulation PASS.

### 15.2. CHƯA được kiểm chứng

- Fresh clone + tất cả packages/weight caches được cài mới trên **máy Windows thứ hai**.
- Toàn bộ P0→P9 chạy liên tục trong workspace mới và các điều kiện approval được chứng minh trên máy mới.
- Byte-identical historical trainer replay hoặc reproduction hoàn toàn trên GPU khác/torch build khác.
- DICOM/mask file-by-file full manifest và public redistribution permission cho mọi asset.
- Một command CLI infer arbitrary user DICOM dùng cho bên ngoài dự án.

### 15.3. Checklist gửi cho người nhận

- [ ] SOURCE R5: `code/` đúng 29 `.py`, tests/scripts/docs/reproducibility, `run_pipeline.sh`.
- [ ] **README.md (bản này)** + `requirements.research.txt` đều nằm **ở root**.
- [ ] DATA/RAW: đã kiểm tra quyền phân phối và 30.000 DICOM + hai JSON được khôi phục đúng cấp.
- [ ] DATA/FROZEN: đủ `master_manifest.csv`, `split.csv`, `boxes.csv`, `split.sha256`, không sửa bytes.
- [ ] DATA/MASK: `data/lung_masks/train/` khi mục tiêu cần P2/P3/A3 training.
- [ ] MODELS: PSPNet `.pth` và 3 checkpoint `best.pt` kèm `config.json`.
- [ ] RESULTS: có `results/qa`, `results/evaluation`, metadata/regression logs cần thiết.
- [ ] Mỗi máy tạo venv mới `.venvs/bgrel`; cài PyTorch wheel phù hợp và requirements tham chiếu.
- [ ] `bash run_pipeline.sh status`, `verify`, `tests` đạt hoặc các lỗi `MISSING` được giải thích rõ.
- [ ] Không chạy các CLI có tác động ghi vào project đã frozen; không chỉnh SHA lịch sử.
- [ ] Với công bố học thuật: đính kèm PDF bài báo và trích dẫn dataset/pretrained models theo điều khoản nguồn.

### 15.4. Nguồn và tài liệu kỹ thuật

- **Source thực tế:** 29 module Python R5; CLI flags đối chiếu từ `argparse` trong các file.
- **Bằng chứng môi trường:** ZIP collector Windows (`meta/ENVIRONMENT_VERSIONS.json`) và `docs/ENVIRONMENT.md`.
- **Review R5:** `BGREL_R5_REVIEW_20261107_070814_d05fb3.zip` (log thực thi trên máy tác giả; không nhất thiết nằm trong ZIP project root).
- **PyTorch v2.9.1 / CUDA wheels:** <https://pytorch.org/get-started/previous-versions/>.
- **TorchXRayVision package:** <https://pypi.org/project/torchxrayvision/1.5.7/>.
- **RSNA competition/rules:** <https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules>.
- **Dữ liệu/source/weights bên thứ ba:** mỗi nguồn có điều khoản riêng; tài liệu này **không cấp phép tái phân phối** tài sản của bên thứ ba.

---

**Giấy phép:** Tài liệu bàn giao hiện không thấy giấy phép sử dụng code được lựa chọn trong materials đã gửi. Người nhận không nên tự suy ra quyền tái phân phối công khai code/dataset/checkpoint; cần chủ sở hữu ghi rõ LICENSE phù hợp trước khi phát hành rộng rãi.

**Nếu bạn chỉ muốn chạy lại các phép đối chiếu kết quả:** sau khi giải nén đầy đủ và cài venv, ba lệnh nên chạy là:

```bash
bash run_pipeline.sh status
bash run_pipeline.sh verify
bash run_pipeline.sh tests
```

**Nếu bạn muốn train lại từ đầu:** đọc kỹ §7–9, yêu cầu một fresh-run pipeline tách biệt với source/artifacts frozen trước khi tuyên bố đã tái lập toàn bộ paper.
