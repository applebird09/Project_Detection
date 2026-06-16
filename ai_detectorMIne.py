"""
================================================================================
  푸리에변환 기반 AI 생성 이미지 판별기 
================================================================================
[ 처음 사용하는 분을 위한 안내 ]

이 파일은 CPU에서도 동작하도록 만들어졌습니다 (GPU 없어도 OK).
다만 학습(train)은 CPU에서 매우 느리므로, 보통은:
  - 학습은 Kaggle 등 GPU 환경에서 미리 해두고 (best_model.pth 파일 확보)
  - 내 컴퓨터(VS Code)에서는 predict / ui 기능만 사용
하는 것을 권장합니다.

[ 필요한 라이브러리 설치 ]
  터미널에서 아래 명령을 실행하세요:
    pip install torch torchvision numpy scipy pillow gradio scikit-learn

[ 실행 방법 ]
  - 판별:  python ai_detectorMine.py predict 사진.jpg
  - UI:    python ai_detectorMine.py ui
  - 학습:  python ai_detectorMine.py train
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# [공통] 라이브러리 불러오기
# ──────────────────────────────────────────────────────────────────────────────
# sys: 명령행 인자(python 파일.py 뒤에 붙는 단어들)를 읽기 위해 사용
# os, io, random, pathlib: 파일/폴더 다루기, 임시 메모리, 무작위, 경로 처리
import sys
import os
import io
import random
from pathlib import Path

# numpy: 숫자 배열(행렬) 계산의 기본 도구
import numpy as np

# torch 계열: 딥러닝(신경망)을 만들고 학습시키는 핵심 라이브러리
import torch
import torch.nn as nn
import torch.nn.functional as F

# scipy: 과학 계산 도구. 여기서는 합성곱(convolve2d)에만 사용
from scipy.signal import convolve2d

# PIL(Pillow): 이미지 파일을 열고 변형하는 도구
from PIL import Image


# ==============================================================================
#  PART 1. 전처리 - 이미지를 푸리에 스펙트럼으로 변환
# ==============================================================================
# 이 코드는 푸리에 변환하는 장치이다. ai에게 그냥 이미지를 주는것보다는
# 주파수 기반으로 변환하는 것이 더 효과가 있기에 푸리에 변환을 시행한다.

def image_to_spectrum(img: np.ndarray, use_residual: bool = True) -> np.ndarray:
    """
    이미지(숫자 격자)를 받아서 log-magnitude 스펙트럼으로 변환합니다.

    매개변수:
        img: 이미지 데이터. (높이, 너비) 흑백 또는 (높이, 너비, 3) 컬러.
        use_residual: True면 고역통과 필터를 먼저 적용 (저주파 제거).

    반환값:
        (높이, 너비) 크기의 스펙트럼 (실수 배열).
    """

    # ── 1단계: 컬러 → 흑백 변환 ──────────────────────────────────────────────
    # 이미지를 푸리에 변환할때 중요한것은 컬러가 아니라 밝기의 주파수이기에
    # 전처리과정에서 이미지를 흑백변환한다,
    # img.ndim==3 이 부분은 이미지가 컬러인지 아닌지 판별한다
    # 만약 이미지가 흑백이면, 축 2개만 있으면 흑백값을 출력할 수 있기에
    # 2차원인데, 컬러이미지는 rgb라는 정보까지 추가로 필요하기에
    # , 축이 3개가 되어 3차원이다.
    # img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2] 이 부분은
    # 이미지를 흑백으로 변환하는 부분이다 각 rgb값에 선형적 연산을 하면 밝기가 나온다.

    if img.ndim == 3:
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    # ── 2단계: 숫자를 0~1 범위로 정리 ────────────────────────────────────────
    # 정수형끼리 계산하면 소수점이 사라질 수 있기에
    # 소수점 계산이 가능한 float 형식으로 바꾼다.
    # 픽셀값을 0~1로 바꾸면 계산이 용이 해서 if img.max() > 1.0:이 코드로
    # 0~255 범위인지 판별하고 255로 나누어 0~1 범위로 바꾼다 
    # 만약 원래 이미지자체가 0~255중 0~1 사이의 숫자로만 되어있는 것과 
    # 0~255에서 0~1로 변환된 이미지를 구별하는 것은, 
    # 0~255인데 0~1 사이에 숫자밖에 없으면 거의 검은 이미지라서
    # 학습에 무의미하기에 따로 과정을 추가하지 않았다.
     
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    # ── 3단계: 고역통과 필터 (라플라시안) ────────────────────────────────────
    # 라플라시안 커널은 ai가 학습하는데 필요한 고주파만 남기는 역할을 한다.
    # 라플라시안 커널은 2차 편미분의 결과인데, 이미지는 이산적이므로,
    # 테일러전개로 근사하고, 가로축 커널과 세로축 합을 구하고, -1을 곱하여 나타낸다.
    
    if use_residual:
        kernel = np.array([[ 0, -1,  0],
                           [-1,  4, -1],
                           [ 0, -1,  0]], dtype=np.float32)
        # convolve2d: 커널을 이미지 전체에 한 칸씩 이동하게 하여 곱,합을 반복한다..
        # mode='same' = 결과의 크기를 입력과 동일하게 유지한다..
        # boundary='symm' = 가장자리 바깥을 거울처럼 채워 왜곡 방지
        # 만약 가장자리 바깥을 채우지 않는다면 커널을 계산할 수 없게 될것이다..
    
        img = convolve2d(img, kernel, mode='same', boundary='symm')

    # ── 4단계: Hann 윈도우 적용 ──────────────────────────────────────────────
    # FFT는 이미지가 무한히 반복된다고 가정한다. 살짝 AB라는 이미지가 있으면
    # ...ABABABABAB...으로 이어진다고 생각하는 느낌이다.
    # 그런데 이미지의 오른쪽과 왼쪽의 밝기 차이가 크다면, 
    # 이미지사이의 가짜 경계가 만들어진다.
    # hann 윈도우는 이를 막기위해 이미지의 양끝을 0에 가깝게 만들어준다.
    # reshape는 크기를 유지한체 배열의 행,열 수를 바꾸는 코드를 의미한다
    # 이때  win_h = np.hanning(H).reshape(-1, 1에서 -1은 1은 행, 열의 크기를 맞추기위해
    # 자동으로 계산해서 채워넣으라는 것을 의미한다

    H, W = img.shape
    win_h = np.hanning(H).reshape(-1, 1)   # 세로 방향 창
    win_w = np.hanning(W).reshape(1, -1)   # 가로 방향 창
    img = img * (win_h * win_w)            # 가로·세로 창을 곱해 2D 창으로 적용

    # ── 5단계: 2차원 푸리에 변환 (핵심!) ─────────────────────────────────────
    # fft2: 이미지에 푸리에 변환을 실행한다. 푸리에변환이란
    # 함수를 입력하면, 주파수의 포함된 정도를 새로운 함수로 출력하는 것을 말한다.
    # 이미지는 이산적인 2차원 자료이므로, 2차원 이산 푸리에변환을 실행한다
    # fftshift: 결과에서 저주파를 중앙으로 옮긴다.
    # 이를 통해 고주파의 흔적을 쉽게 파악할 수 있다.

    F_complex = np.fft.fft2(img)
    F_complex = np.fft.fftshift(F_complex)

    # ── 6단계: 로그 변환 ─────────────────────────────────────────────────────
    # abs로 크기를 구한 뒤 log를 취한다.
    # 저주파와 고주파간의 차이가 매우커서 로그연산으로 차이를 줄인다.
    # log1p는 로그에 0이 들어가도 log(1+x) 의 구조라서 안전하다.

    spectrum = np.log1p(np.abs(F_complex))

    # ── 7단계: 정규화 (평균 0, 표준편차 1) ───────────────────────────────────
    # 정규화 과정은 모델이 학습하는 것이 편하게 값들을 정규화해준다.
    # 정규분포의 정규화 식과 같고, 평균을 0, 표준편차를 1이 되게한다..
    # +1e-8은 0으로 나누는 것을 막아준다.

    spectrum = (spectrum - spectrum.mean()) / (spectrum.std() + 1e-8)

    return spectrum.astype(np.float32)


# ==============================================================================
#  PART 2. 모델 정의 - CNN(합성곱 신경망)
# ==============================================================================
# 스펙트럼을 입력으로 받아 [실제, 가짜] 두 점수를 출력하는 신경망입니다.

class BasicBlock(nn.Module):
    """
    ResNet 스타일의 기본 블록.
    핵심 아이디어 = '잔차 연결(residual)': 입력을 출력에 그대로 더해줘서
    신경망이 깊어져도 학습이 잘 되게 한다 (정보가 사라지지 않도록).
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # super()은 원래의 class의 init을 먼저수행하도록 한다.
        gn1 = min(8, out_ch)
        gn2 = min(8, out_ch)
        # Conv2d: 합성곱 층. 이미지에서 특징(무늬)을 뽑아낸다.
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(gn1, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(gn2, out_ch)

        # shortcut을 이용하면 원본을 보존할 수 았다
        # 원본을 보존하지 않으면 학습이 진행될 수록 원본에서 점차 희미해져 가는데
        # 이를 막기위해 shortcut을 사용한다
        # 입력과 출력의 크기가 다르면 1x1 합성곱으로 크기를 맞춘다.
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
            )
        else:
            self.shortcut = nn.Identity()  # 크기가 같으면 그대로 통과
        self.act = nn.GELU()  # 합성곱연산은 선형적인데 이는 아무리 많이 학습을 해도 직선을 학습한것에 지나지 않는다.
                              # GELU는 비선형함수로, 선형적인 합성곱에 비선형 패턴을 추가해준다. 이것이 활성화함수이다.

    def forward(self, x):
        identity = self.shortcut(x)               # 입력을 따로 보관하여 원본을 유지한다
        out = self.act(self.norm1(self.conv1(x)))  # 합성곱→정규화→활성화
        out = self.norm2(self.conv2(out))          # 한 번 더 합성곱→정규화
        return self.act(out + identity)            # 입력을 더한 뒤 활성화 이를 통해 기울기 손실을 막는다


class CoordChannels(nn.Module):
    """
    스펙트럼에 '좌표 정보'를 추가 채널로 붙여준다.
    fftshift 후 중앙이 저주파(DC), 바깥이 고주파라는 '위치'를
    신경망이 알 수 있도록 거리(r)와 각도(theta) 정보를 함께 넣는다.
    """
    # cnn은 위치를 모른다는 단점이 있다. 중앙의 저주파와 가장자리의 고주파에
    # 같은 무늬가 나타나도 똑같이 반응하는 식이다.
    # 이 코드는 이러한 문제점을 해결하기 위해 스펙트럼 정보에 거리정보와 위치 정보를 추가한다.
    def forward(self, x):
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        # -1~1 범위의 좌표 격자를 만든다 (중앙이 0)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        r = torch.sqrt(xx ** 2 + yy ** 2)          # 중앙으로부터의 거리를 피타고라스 정리로 계산한다.
        theta = torch.atan2(yy, xx) / np.pi        # 각도를 계산한다
        # 거리·각도를 채널로 쌓아서 원래 입력 뒤에 붙인다
        coord = torch.stack([r, theta], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return torch.cat([x, coord], dim=1)


class SpectrumDetector(nn.Module):
    """
    전체 모델. 스펙트럼을 받아 [실제, 가짜] 점수를 출력한다.
    구조: 좌표채널 추가 → Stem → 4개의 레이어(점점 깊어짐) → 평균 → 분류기
    """
    # 이 코드는 앞에서 만든 두개의 class를 종합하여, 실제와 가짜 점수를 출력하는 부분이다.
    # 신경망은 커널의 값을 변화시켜가며, 숫자를 출력하고 이 값이 얼마나 원하는 값과 유사한지를 기준으로
    # 커널의 값을 다시 변화하는 학습과정을 반복하게된다.

    def __init__(self, in_channels=1, num_classes=2, use_coord=True, base_width=64):
        super().__init__()
        # 좌표 채널을 쓰면 입력 채널이 2개 늘어난다 (거리 + 각도)
        self.coord = CoordChannels() if use_coord else nn.Identity()
        actual_in = in_channels + (2 if use_coord else 0)
        w = base_width

        # Stem: 첫 입구. 작은 3x3 커널로 시작해 미세한 고주파 정보를 보존.
        self.stem = nn.Sequential(
            nn.Conv2d(actual_in, w, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, w), w),
            nn.GELU(),
        )
        # 4개의 단계. 뒤로 갈수록 채널(특징 수)은 늘고 해상도는 줄어든다. 해상도와 특징은 연산에 따른 증감(한쪽이 증가하면 한쪽은 감소)관계를 가진다.
        self.layer1 = self._make_layer(w,     w,     2, 1)
        self.layer2 = self._make_layer(w,     w * 2, 2, 2)
        self.layer3 = self._make_layer(w * 2, w * 4, 2, 2)
        self.layer4 = self._make_layer(w * 4, w * 8, 2, 2)
        # 마지막: 전체 평균을 내서 하나의 특징 벡터로 → 2개 클래스 점수로 변환
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.4),                 # ai의 학습과정에서 특정 가중치만 따르게되어 새로운 데이터를 판별 할 수 없게되는
                                             # 과적합이라는 현상이 발생할 수 있는데, 이 코드는 0.4개의 무작위 뉴런을 끔으로써 이를 방지한다.
            nn.Linear(w * 8, w * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(w * 2, num_classes),   # 최종 출력: [실제 점수, 가짜 점수]
        )
        self._init_weights()

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        """BasicBlock 여러 개를 이어 붙여 하나의 레이어를 만든다."""
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """학습 시작 전 가중치를 적절한 값으로 초기화 (학습이 잘 되도록)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """입력 x가 신경망을 통과하는 순서."""
        x = self.coord(x)    # 좌표 채널 추가
        x = self.stem(x)     # 입구
        x = self.layer1(x)   # 레이어 1
        x = self.layer2(x)   # 레이어 2
        x = self.layer3(x)   # 레이어 3
        x = self.layer4(x)   # 레이어 4
        x = self.gap(x).flatten(1)  # 전체 평균 → 1차원 벡터
        return self.head(x)  # 최종 [실제, 가짜] 점수


# ==============================================================================
#  PART 3. 데이터셋 - 폴더에서 이미지를 읽어 스펙트럼으로 변환
# ==============================================================================
# 학습(train)할 때만 사용됩니다.

from torch.utils.data import Dataset, DataLoader


class SpectrumDataset(Dataset):
    """
    이미지 폴더를 읽어, 각 이미지를 스펙트럼으로 바꿔 신경망에 공급한다.
    폴더 구조:  루트/real/*.jpg (실제),  루트/fake/*.jpg (AI 생성)
    (REAL/FAKE 대문자도 자동 인식)
    """
    # 이 코드는 학습의 과정을 담았다. 데이터셋이 주어졌다고 하면, 그 데이터셋은 실제 이미지와 ai생성 이미지로 구별될 것이다.
    # 코드를 실행하면, 그 이미지를 실제면 0, 거짓이면 1 이런 방식으로 라벨링하고, 스펙트럼 변환하여 신경망에 전달한다.

    def __init__(self, root_dir, image_size=256, augment=True, use_residual=True):
        self.root = Path(root_dir)
        self.image_size = image_size
        self.augment = augment            # 학습 때만 데이터 증강 ON
        self.use_residual = use_residual

        # real=0, fake=1 라벨로 파일 목록을 모은다 (대소문자 모두 지원)
        self.samples = []
        for label, names in enumerate([['real', 'REAL', 'Real'],
                                       ['fake', 'FAKE', 'Fake']]):
            for name in names:
                cls_dir = self.root / name
                if cls_dir.exists():
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
                        self.samples.extend((p, label) for p in cls_dir.glob(ext))
                        self.samples.extend((p, label) for p in cls_dir.glob(ext.upper()))
                    break
        if not self.samples:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {root_dir}")

        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print(f"[{root_dir}] 총 {len(self.samples)}장 (real: {n_real}, fake: {n_fake})")

        # 데이터의 길이를 반환한다
    def __len__(self):
        return len(self.samples)

    def _augment_image(self, img):
        """데이터 증강: 같은 이미지를 조금씩 변형해 학습 데이터를 다양하게 한다."""
        # 데이터 증강이란 같은 이미지라도, 다른 처리를 하여 데이터를 늘리고, 모델이 jpeg 압축을 학습하는등의 오류를 방지하는 방식을 말한다.

        # (1) 50% 확률로 좌우 반전
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # (2) 무작위 크롭 (이미지 일부를 잘라냄)
        w, h = img.size
        crop_scale = random.uniform(0.7, 1.0)
        cw, ch = int(w * crop_scale), int(h * crop_scale)
        if cw < w or ch < h:
            x = random.randint(0, w - cw)
            y = random.randint(0, h - ch)
            img = img.crop((x, y, x + cw, y + ch))
        # (3) 무작위 JPEG 압축
        # 무작위 jpeg 압축을 진행하는 이유는 모델이 실제 사진과 ai사진의 구별이 아닌 jpeg 압축 자체를 학습할 수 있기 때문이다.
        if random.random() < 0.7:
            buffer = io.BytesIO()
            quality = random.randint(30, 95)
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
            img.load()
        return img

    def __getitem__(self, idx):
        """idx번째 이미지를 읽어 (스펙트럼, 라벨) 형태로 반환한다."""
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')   # 어떤 형식이든 RGB로 통일
        except Exception:
            # 깨진 파일이면 다음 이미지로 건너뛴다
            return self.__getitem__((idx + 1) % len(self))

        if self.augment:
            img = self._augment_image(img)

        # 고정 크기로 리사이즈 후 스펙트럼으로 변환
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32)
        spectrum = image_to_spectrum(img_np, use_residual=self.use_residual)
        # (높이, 너비) → (1, 높이, 너비) 로 채널 차원 추가
        spec_tensor = torch.from_numpy(spectrum).unsqueeze(0)
        return spec_tensor, label


# ==============================================================================
#  PART 4. 학습 관련 함수들
# ==============================================================================
#  이 부분은 실제 학습을 위한 여러 함수들이 제공되어있다.

# 이 코드는 AUC라는 모델의 성능 측도를 계산하게 해주는 부분이다. 
def compute_auc(labels, scores):
    """
    AUC(곡선 아래 면적) 계산. 모델이 진짜/가짜를 얼마나 잘 구별하는지 0~1 점수.
    1에 가까울수록 좋고, 0.5면 동전 던지기 수준.
    """
    order = np.argsort(-scores)              # 점수 높은 순으로 정렬: 원래 argsort는 오름차순으로 정렬을 해서 가장 높은 점수는 맨 끝으로 간다.
                                             # 우리는 이와 반대인 내림차순 정렬을 원하기에 - 부호를 붙여서 가장 큰 숫자가 가장 작아지도록한다.
    labels_sorted = labels[order]
    n_pos = labels_sorted.sum()              # 가짜(양성) 개수
    n_neg = len(labels_sorted) - n_pos       # 진짜(음성) 개수
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tpr = np.cumsum(labels_sorted) / n_pos        # 참 양성 비율
    fpr = np.cumsum(1 - labels_sorted) / n_neg    # 거짓 양성 비율
    # 면적 계산. NumPy 버전에 따라 함수 이름이 다르므로 둘 다 대비.
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(tpr, fpr))
    else:
        return float(np.trapz(tpr, fpr))


def train_one_epoch(model, loader, optimizer, criterion, device):
    """모델을 1 에폭(전체 데이터 한 바퀴) 학습시킨다."""
    # 모델을 실제로 학습시키는 부분으로, 학습의 목표는 손실함수의 최소화이다. 
    # 손실함수의 최소화를 위해서 편미분값을 계산하고, 손실을 줄이는 방향으로 가중치를 추가한다.

    model.train()  # 학습 모드 (Dropout 등 활성화)
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (specs, labels) in enumerate(loader):
        specs = specs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()         # 이전 기울기 초기화
        logits = model(specs)         # 예측
        loss = criterion(logits, labels)  # 손실 계산
        loss.backward()               # 손실을 줄이는 방향 계산 
        optimizer.step()              # 그 방향으로 가중치 업데이트

        bs = specs.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += bs
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx:4d}/{len(loader)}: "
                  f"loss={loss.item():.4f}, acc={correct/total*100:.2f}%")
    return total_loss / total, correct / total


@torch.no_grad()  # 이 코드는 지금까지의 학습을 평가하여 auc 점수를 매기는 코드이다.
def validate(model, loader, criterion, device):
    """학습에 쓰지 않은 데이터로 모델 성능을 평가한다."""
    model.eval()  # 평가 모드로 전환하여 뉴런을 끄는 dropout등의 기능을 비활성화한다.
    total_loss, correct, total = 0.0, 0, 0
    all_scores, all_labels = [], []
    for specs, labels in loader:
        specs = specs.to(device)
        labels = labels.to(device)
        logits = model(specs)
        loss = criterion(logits, labels)
        bs = specs.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += bs
        scores = F.softmax(logits, dim=1)[:, 1]  # '가짜일 확률'을 모은다
        all_scores.append(scores.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    auc = compute_auc(all_labels, all_scores)
    return total_loss / total, correct / total, auc


def run_training():
    """[기능 1] 학습 전체 흐름. 'python ai_detectorMine.py train' 으로 실행."""
    # ----- 설정 (필요에 맞게 수정하세요) -----
    # 이 부분은 딕셔너리로 여러 설정값들을 모아놓은 부분이다.
    config = {
        'data_root': './data',     # 데이터 폴더 (안에 train/, val/ 가 있어야 함)
        'batch_size': 16,           # 한 번에 처리할 이미지 수 (CPU면 작게)
        'num_workers': 0,           # CPU/Windows에서는 0이 안전
        'epochs': 5,                # 학습 반복 횟수
        'lr': 1e-4,                 # 학습률
        'weight_decay': 1e-4,       # 과적합을 방지하기위한 변수이다.
        'image_size': 256,
        'save_path': './best_model.pth',
    }
    # 만약 GPU가 있다면 cuda를 선택한다.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cpu':
        print("⚠️  CPU로 학습합니다. 매우 느릴 수 있습니다 (GPU 환경 권장).")

    # 데이터셋 준비 (학습 = 증강 ON, 검증= 증강 OFF)
    # 검증할때는 데이터셋을 변형할 필요가 없이 원본이 필요하기 때문에 증강을 끈다.
    # 또한 검증할때는 데이터셋을 섞는 suffle도 끈다.

    train_ds = SpectrumDataset(os.path.join(config['data_root'], 'train'),
                               image_size=config['image_size'], augment=True)
    val_ds = SpectrumDataset(os.path.join(config['data_root'], 'val'),
                             image_size=config['image_size'], augment=False)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True, num_workers=config['num_workers'], drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'],
                            shuffle=False, num_workers=config['num_workers'])

    # 모델, 옵티마이저(가중치 업데이트 방법), 스케줄러(학습률 조절), 손실함수
    # 학습에 필요한 여러 장치들을 정의한다.

    model = SpectrumDetector(in_channels=1, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])   # 학습률을 줄여준다. 나중으로 갈수록 기울기의 세세한 조정이 가능하다.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 학습 루프: 모델의 AUC가 최고 기록일때만 저장해서 모델이 성장하게한다.
    best_auc = 0.0
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*50}\nEpoch {epoch}/{config['epochs']}\n{'='*50}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"\nTrain: loss={train_loss:.4f}, acc={train_acc*100:.2f}%")
        print(f"Val:   loss={val_loss:.4f}, acc={val_acc*100:.2f}%, AUC={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_auc': val_auc, 'val_acc': val_acc, 'config': config},
                       config['save_path'])
            print(f"  ✅ 최고 모델 저장 (AUC={val_auc:.4f})")
    print(f"\n학습 완료! 최고 AUC: {best_auc:.4f}  저장: {config['save_path']}")


# ==============================================================================
#  PART 5. 판별(추론) - 학습된 모델로 새 이미지 판별
# ==============================================================================

# 모델을 한 번만 불러와서 재사용하기 위한 전역 변수
_loaded_model = None
_loaded_device = None


def load_trained_model(model_path='./best_model.pth'):
    """저장된 모델 파일을 불러온다 (한 번만 불러오고 재사용)."""
    #  if _loaded_model is not None:return _loaded_model, _loaded_device 부분은 
    #  모델 파일이 이미 저장되어 있다면 새로 불러오지 않고 저장되어 있는 것을 재사용한다는 것을 의미한다.

    global _loaded_model, _loaded_device
    if _loaded_model is not None:
        return _loaded_model, _loaded_device
    
    # 모델 파일이 있는지 확인하고 없다면 raise 코드를 이용하여 오류를 발생시킨다.
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {model_path}\n"
            f"→ Kaggle에서 학습한 best_model.pth를 이 파일과 같은 폴더에 두세요.\n"
            f"→ 또는 'python ai_detectorMine.py train' 으로 먼저 학습하세요."
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = SpectrumDetector(in_channels=1, num_classes=2).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()    # 모델을 평가모드로 전환한다.
    print(f"모델 로드 완료 (AUC={ckpt.get('val_auc', 0):.4f})")
    _loaded_model, _loaded_device = model, device
    return model, device


@torch.no_grad()
def predict_image(pil_image, model_path='./best_model.pth'):
    """
    PIL 이미지 한 장을 판별해서 결과를 딕셔너리로 반환.
    반환: {'real_prob': 실제확률, 'fake_prob': 가짜확률, 'prediction': 판정}
    """
    # 이 부분은 실제로 확률을 계산하여 출력하는 부분이다.
    model, device = load_trained_model(model_path)

    img = pil_image.convert('RGB').resize((256, 256), Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32)
    spectrum = image_to_spectrum(img_np, use_residual=True)     # 이미지를 전처리하는 과정이다.
    x = torch.from_numpy(spectrum).unsqueeze(0).unsqueeze(0).to(device)     # 모델은 4차원 정보를 원하므로 2차원 스펙트럼의 차원을 2개 추가한다.

    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]   #softmax 함수로 logit이라는 점수를 0~1 사이의 합이 1이되는 실제,가짜 확률로 반환하여 저장한다.
    return {
        'real_prob': float(probs[0]),
        'fake_prob': float(probs[1]),
        'prediction': 'AI 생성 (FAKE)' if probs[1] > 0.5 else '실제 사진 (REAL)',
    }


def run_predict(target_path):
    """[기능 2] 파일/폴더 판별. 'python ai_detectorMine.py predict <경로>' 로 실행."""
    # 이 부분은 입력이 파일인지 폴더인지 구분하고, 각각의 경우에 따른 확률 출력과정을 담고있다.
    # 만약 파일이라면, 단일로 폴더라면, 확률을 일일이 계산하여 출력해준다.
    target = Path(target_path)
    exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

    # 판별할 파일 목록 만들기 (단일 파일 or 폴더 전체)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(p for p in target.iterdir() if p.suffix.lower() in exts)
    else:
        print(f"경로를 찾을 수 없습니다: {target}")
        return
    # 파일에서 각각의 이미지에 대해 확률을 출력하는 부분이다.
    for f in files:
        try:
            img = Image.open(f)
            result = predict_image(img)
            print(f"\n파일: {f.name}")
            print(f"  실제 확률: {result['real_prob']*100:6.2f}%")
            print(f"  가짜 확률: {result['fake_prob']*100:6.2f}%")
            print(f"  >>> 판정: {result['prediction']}")
        except Exception as e:
            print(f"  {f.name} 처리 실패: {e}")


# ==============================================================================
#  PART 6. 웹 UI - Gradio로 이미지 업로드 화면 만들기
# ==============================================================================
# 이 코드는 ai 이미지 판별기가 작동하도록 webui를 구축하는 코드이고, gradio를 활용했다. 
def run_ui():
    """[기능 3] 웹 UI 실행. 'python ai_detectorMine.py ui' 로 실행."""
    # gradio를 불러오거나 설치되어있는지 확인하는 부분이다.
    try:
        import gradio as gr     # 함수안에 import가 있는 이유는 train과정에서는 gradio가 필요하지 않아서 열어야 할때만 불러오기 위해서이다.
    except ImportError:
        print("Gradio가 설치되어 있지 않습니다. 터미널에서 실행하세요:")
        print("    pip install gradio")
        return

    # 시작 시 모델을 미리 불러온다 모델이 없다면 에러를 발생시킨다.
    try:
        load_trained_model()
    except FileNotFoundError as e:
        print(e)
        return

    def predict_for_ui(image):
        """Gradio가 넘겨준 이미지를 판별해 확률 딕셔너리로 반환."""
        # gradio의 ui를 이용하여 확률을 출력하는 부분이다.
        if image is None:
            return {"이미지를 올려주세요": 1.0}
        result = predict_image(image)
        return {
            "실제 사진 ": result['real_prob'],
            "AI 생성 ": result['fake_prob'],
        }

    # UI 구성: 이미지 입력 → 확률 막대 출력
    # gr.Interface를 이용하였고 gradio는 이에 따라 ui를 생성해준다.
    demo = gr.Interface(
        fn=predict_for_ui,
        inputs=gr.Image(type="pil", label="이미지를 업로드해주세요"),
        outputs=gr.Label(num_top_classes=2, label="판별 결과"),
        title="푸리에변환기반 AI 생성 이미지 판별기",
        description="이미지를 올리면 푸리에 스펙트럼을 분석해 AI 생성 여부를 판별할 수 있어요!",
    )
    # share=False면 내 컴퓨터에서만 접속할 수 있다. (http://127.0.0.1:7860)
    demo.launch(share=True)     #만약 상태가 TRUE이면, 다른 사람들도 이용가능하다.


# ==============================================================================
#  PART 7. 진입점 - 어떤 기능을 실행할지 결정
# ==============================================================================
# 'python ai_detectorMine.py 뒤에 오는 단어'에 따라 기능을 고른다.
#  main함수는 프로그램의 진입점 역할을 한다. 터미널에서 받은 명령을 바탕으로 역할을 수행하는 코드이다.

def main():
    # sys.argv = ['ai_detectorMine.py', '명령어', '추가인자...']
    if len(sys.argv) < 2:   # sys.argv는 사용자가 입력한 터미널의 명령어를 의미하는데, 명령어의 길이가 2미만이라는 것은 사용자가 아무런 명령어도 입력하지 않았음을 의미하고, 이에 따라 사용법을 출력해준다.
        # 아무 명령도 없으면 사용법 안내
        print(__doc__)  # 파이썬에서는 코드 맨위 큰 따옴표 3개의 문자열을 독스트링이라고하는데, 이를 출력하는 부분이다.
        print("\n사용법:")
        print("  python ai_detectorMine.py train              # 학습")
        print("  python ai_detectorMine.py predict <경로>     # 이미지/폴더 판별")
        print("  python ai_detectorMine.py ui                 # 웹 UI 실행")
        return

    command = sys.argv[1].lower()
    # 다음코드는 커맨드에 따라 다른 역할을 수행하는 부분이다.

    if command == 'train':
        run_training()
    elif command == 'predict':
        if len(sys.argv) < 3:
            print("판별할 이미지나 폴더 경로를 입력하세요.")
            print("예: python ai_detectorMine.py predict 사진.jpg")
            return
        run_predict(sys.argv[2])
    elif command == 'ui':
        run_ui()
    else:
        print(f"알 수 없는 명령: {command}")
        print("사용 가능한 명령: train, predict, ui")


# 이 파일을 직접 실행했을 때만 main()을 호출한다.
# 만약 이 코드가 없다면, 다른 코드에서 함수만 import해갔을때 전체 코드가 실행될 수 있어서, 이 위험을 예방한다.
if __name__ == "__main__":
    main()
