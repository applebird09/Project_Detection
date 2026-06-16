"""
================================================================================
  화면 영역 캡처 AI 이미지 판별기 - 다크 + 원형 메뉴 (개선판)
================================================================================

사용법:
  1. ★중요★ 아래 'from ai_detector' 를 본인 파일명으로 바꾸세요!
     예: 파일이 ai_detectorMine.py 이면  ->  from ai_detectorMine import ...
  2. best_model.pth 가 같은 폴더에 있어야 실제 판별됩니다.
     (없거나 import 실패 시 데모 모드 = 항상 77%)
  3. pip install pillow mss matplotlib
  4. python dark_detector.py

기능:
  - 플로팅 + 버튼: 더블클릭 → 화면 영역 캡처 판별
  - 마우스 올리면 원형 메뉴 3개: 📁업로드  🕘히스토리  🎨로고
  - 로고 커스텀: 이미지 업로드 → 버튼 로고 변경 (다음에 켜도 유지 / 되돌리기 가능)
  - 히스토리: 판별 기록 보관(스크롤 가능), 클릭하면 그때 결과 다시 표시
  - 단계별 로딩창 → 결과창
================================================================================
"""
import sys, os
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import tkinter as tk
from tkinter import filedialog
import io
import numpy as np
from PIL import Image, ImageTk, ImageDraw

try:
    import mss
    _USE_MSS = True
except ImportError:
    _USE_MSS = False
    from PIL import ImageGrab

# ★★★ 여기를 본인 파일명으로! (ai_detector → ai_detectorMine 등) ★★★
try:
    from ai_detectorMIne import image_to_spectrum, predict_image
    MODEL_AVAILABLE = True
except Exception as e:
    print(f"[안내] 모델 모듈 로드 실패(데모 모드, 항상 77%): {e}")
    MODEL_AVAILABLE = False

BG = "#14141c"; CARD = "#1c1c28"; BORDER = "#2a2a3a"
ACCENT = "#7D94FF"; TEXT = "#ffffff"; MUTED = "#9aa3b5"; TRANSP = "#010101"

# 커스텀 로고 저장 경로 (이 스크립트와 같은 폴더)
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_logo.png")

HISTORY = []   # [{captured_pil, spec_pil, real, fake, thumb_pil}]


def make_plus_icon(size=56):
    """기본 + 아이콘 (테두리 없음)."""
    scale = 3; big = size * scale
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    m = int(big * 0.16)
    draw.rounded_rectangle([m, m, big-m, big-m], radius=int(big*0.18),
                           fill=(28, 28, 40, 255))   # 외곽선 제거
    c = big / 2; arm = big * 0.20; t = int(big * 0.022)
    draw.line([c, c-arm, c, c+arm], fill=(125, 148, 255, 255), width=t*2)
    draw.line([c-arm, c, c+arm, c], fill=(125, 148, 255, 255), width=t*2)
    return img.resize((size, size), Image.LANCZOS)


def make_logo_from_image(pil_image, size=56):
    """업로드 이미지를 둥근 로고로 (테두리 없음)."""
    img = pil_image.convert("RGBA").resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size-1, size-1],
                                           radius=int(size*0.18), fill=255)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def grab_region(x1, y1, x2, y2):
    left, top = int(min(x1, x2)), int(min(y1, y2))
    right, bottom = int(max(x1, x2)), int(max(y1, y2))
    if _USE_MSS:
        with mss.mss() as sct:
            region = {"left": left, "top": top,
                      "width": max(1, right-left), "height": max(1, bottom-top)}
            shot = sct.grab(region)
            return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    else:
        return ImageGrab.grab(bbox=(left, top, right, bottom))


def make_spectrum_pil(pil_image):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr = np.array(pil_image.convert("RGB").resize((256, 256)), dtype=np.float32)
    if MODEL_AVAILABLE:
        spec = image_to_spectrum(arr, use_residual=True)
    else:
        gray = arr.mean(axis=2)
        f = np.fft.fftshift(np.fft.fft2(gray))
        spec = np.log1p(np.abs(f)); spec = (spec-spec.mean())/(spec.std()+1e-8)
    fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=100)
    ax.imshow(spec, cmap="bone"); ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig); buf.seek(0)
    return Image.open(buf)


def predict_probs(pil_image):
    fake, real = 0.77, 0.23
    if MODEL_AVAILABLE:
        try:
            r = predict_image(pil_image)
            fake, real = r["fake_prob"], r["real_prob"]
        except FileNotFoundError:
            pass
    return real, fake


# ── 영역 선택 ────────────────────────────────────────────────
class RegionSelector:
    def __init__(self, root, on_done):
        self.root = root; self.on_done = on_done
        self.start_x = self.start_y = 0; self.rect = None
        self.win = tk.Toplevel(root)
        self.win.attributes("-topmost", True); self.win.overrideredirect(True)
        self.sw = self.win.winfo_screenwidth(); self.sh = self.win.winfo_screenheight()
        self.win.geometry(f"{self.sw}x{self.sh}+0+0")
        self.win.attributes("-alpha", 0.55); self.win.config(cursor="cross")
        self.canvas = tk.Canvas(self.win, highlightthickness=0, bg=BG,
                                width=self.sw, height=self.sh)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(self.sw//2, 34, text="드래그하여 영역 선택   (ESC: 취소)",
                                fill=ACCENT, font=("맑은 고딕", 15))
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.win.bind("<Escape>", lambda e: self.win.destroy())
        self.win.focus_force()
    def _press(self, e):
        self.start_x, self.start_y = e.x, e.y
        self.rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline=ACCENT, width=2)
    def _drag(self, e):
        if self.rect: self.canvas.coords(self.rect, self.start_x, self.start_y, e.x, e.y)
    def _release(self, e):
        x1, y1, x2, y2 = self.start_x, self.start_y, e.x, e.y
        if abs(x2-x1) < 5 or abs(y2-y1) < 5: return
        self.win.destroy(); self.on_done(x1, y1, x2, y2)


# ── 단계별 로딩창 ────────────────────────────────────────────
class LoadingWindow:
    STEPS = [("이미지를 불러왔습니다", 25), ("푸리에 스펙트럼으로 변환 중", 50),
             ("AI 모델로 분석 중", 75), ("결과를 정리하는 중", 100)]
    def __init__(self, root, img, on_complete):
        self.root = root; self.img = img; self.on_complete = on_complete
        self.step_idx = 0; self.spec_img = None; self.real = self.fake = None
        self.win = tk.Toplevel(root); self.win.title("분석 중")
        self.win.configure(bg=CARD); self.win.geometry("340x150")
        self.win.attributes("-topmost", True); self.win.resizable(False, False)
        tk.Label(self.win, text="분석 중...", bg=CARD, fg=TEXT,
                 font=("맑은 고딕", 13, "bold")).pack(pady=(22, 6))
        self.status = tk.Label(self.win, text="", bg=CARD, fg=MUTED, font=("맑은 고딕", 10))
        self.status.pack()
        self.bar_w = 280
        self.canvas = tk.Canvas(self.win, width=self.bar_w, height=6, bg=BORDER, highlightthickness=0)
        self.canvas.pack(pady=(14, 4))
        self.bar = self.canvas.create_rectangle(0, 0, 0, 6, fill=ACCENT, width=0)
        self.pct = tk.Label(self.win, text="0%", bg=CARD, fg=MUTED, font=("맑은 고딕", 9))
        self.pct.pack()
        self.win.after(300, self._next)
    def _next(self):
        if self.step_idx >= len(self.STEPS):
            self.win.destroy()
            self.on_complete(self.img, self.spec_img, self.real, self.fake); return
        text, target = self.STEPS[self.step_idx]
        self.status.config(text=text)
        if self.step_idx == 1: self.spec_img = make_spectrum_pil(self.img)
        elif self.step_idx == 2: self.real, self.fake = predict_probs(self.img)
        self._fill(target, self._after)
    def _after(self):
        self.step_idx += 1; self.win.after(250, self._next)
    def _fill(self, target, cb):
        cur = self.canvas.coords(self.bar)[2]
        tw = self.bar_w * target / 100; step = (tw - cur) / 8
        def grow(c):
            if c >= 8:
                self.canvas.coords(self.bar, 0, 0, tw, 6)
                self.pct.config(text=f"{target}%"); cb(); return
            nw = cur + step*(c+1)
            self.canvas.coords(self.bar, 0, 0, nw, 6)
            self.pct.config(text=f"{int(nw/self.bar_w*100)}%")
            self.win.after(30, lambda: grow(c+1))
        grow(0)


# ── 결과 표시 (히스토리 기록과 분리) ─────────────────────────
def display_result(root, captured_img, spec_img, real, fake):
    """결과창을 띄우기만 함 (히스토리에 추가하지 않음)."""
    win = tk.Toplevel(root); win.title("판별 결과")
    win.configure(bg=CARD); win.geometry("380x430"); win.attributes("-topmost", True)
    fake_pct, real_pct = round(fake*100), round(real*100)
    is_fake = fake > 0.5
    big = fake_pct if is_fake else real_pct
    sub = "AI 생성으로 추정" if is_fake else "실제 사진으로 추정"
    tk.Label(win, text="분석 결과", bg=CARD, fg=MUTED, font=("맑은 고딕", 11)).pack(pady=(22, 2))
    num = tk.Frame(win, bg=CARD); num.pack()
    tk.Label(num, text=f"{big}", bg=CARD, fg=TEXT, font=("맑은 고딕", 40, "bold")).pack(side="left")
    tk.Label(num, text="%", bg=CARD, fg=MUTED, font=("맑은 고딕", 18)).pack(side="left", anchor="s", pady=(0, 10))
    tk.Label(win, text=sub, bg=CARD, fg=MUTED, font=("맑은 고딕", 11)).pack()
    bar_w = 300
    cv = tk.Canvas(win, width=bar_w, height=6, bg=BORDER, highlightthickness=0)
    cv.pack(pady=(14, 6))
    cv.create_rectangle(0, 0, bar_w*(big/100), 6, fill=ACCENT, width=0)
    tk.Label(win, text=f"실제 {real_pct}%   ·   AI 생성 {fake_pct}%",
             bg=CARD, fg=MUTED, font=("맑은 고딕", 9)).pack(pady=(0, 12))
    tk.Label(win, text="푸리에 스펙트럼", bg=CARD, fg=MUTED, font=("맑은 고딕", 9)).pack()
    spec_tk = ImageTk.PhotoImage(spec_img.resize((150, 150)))
    sl = tk.Label(win, image=spec_tk, bg=CARD); sl.image = spec_tk; sl.pack(pady=(4, 0))


def record_and_show(root, captured_img, spec_img, real, fake):
    """판별 직후: 히스토리에 기록 + 결과창 표시."""
    thumb = captured_img.convert("RGB").copy(); thumb.thumbnail((48, 48))
    HISTORY.append({"captured": captured_img, "spec": spec_img,
                    "real": real, "fake": fake, "thumb_pil": thumb})
    display_result(root, captured_img, spec_img, real, fake)


# ── 히스토리 창 (스크롤 + 클릭하면 결과 다시 표시) ───────────
def show_history(root):
    win = tk.Toplevel(root); win.title("판별 기록")
    win.configure(bg=CARD); win.geometry("320x420"); win.attributes("-topmost", True)
    tk.Label(win, text="🕘 판별 기록", bg=CARD, fg=TEXT,
             font=("맑은 고딕", 13, "bold")).pack(pady=(16, 10))
    if not HISTORY:
        tk.Label(win, text="아직 판별 기록이 없습니다.", bg=CARD, fg=MUTED,
                 font=("맑은 고딕", 10)).pack(pady=20)
        return

    # 스크롤 가능한 영역 (Canvas + Frame + Scrollbar)
    container = tk.Frame(win, bg=CARD); container.pack(fill="both", expand=True, padx=10, pady=(0,10))
    canvas = tk.Canvas(container, bg=CARD, highlightthickness=0)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas, bg=CARD)
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    # 마우스 휠 스크롤
    canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta/120), "units"))

    refs = []  # PhotoImage 참조 유지
    for h in reversed(HISTORY):   # 최근 것이 위로
        row = tk.Frame(inner, bg=BORDER, cursor="hand2"); row.pack(fill="x", pady=3)
        tk_img = ImageTk.PhotoImage(h["thumb_pil"]); refs.append(tk_img)
        lbl = tk.Label(row, image=tk_img, bg=BORDER); lbl.pack(side="left", padx=6, pady=6)
        fake_pct = round(h["fake"]*100)
        verdict = "AI 생성" if h["fake"] > 0.5 else "실제 사진"
        info = tk.Frame(row, bg=BORDER); info.pack(side="left", padx=6)
        tk.Label(info, text=verdict, bg=BORDER, fg=TEXT, font=("맑은 고딕", 10, "bold")).pack(anchor="w")
        tk.Label(info, text=f"AI {fake_pct}%", bg=BORDER, fg=MUTED, font=("맑은 고딕", 9)).pack(anchor="w")
        # 클릭하면 그때 결과 다시 표시 (모든 자식 위젯에 바인딩)
        def make_handler(entry):
            return lambda e: display_result(root, entry["captured"], entry["spec"],
                                            entry["real"], entry["fake"])
        handler = make_handler(h)
        for w in (row, lbl, info, *info.winfo_children()):
            w.bind("<Button-1>", handler)
    win._img_refs = refs   # GC 방지


class TrashZone:
    def __init__(self, root):
        sw = root.winfo_screenwidth(); sh = root.winfo_screenheight()
        w, h = 180, 60
        self.x1 = sw//2 - w//2; self.y1 = sh - h - 30
        self.x2 = self.x1 + w; self.y2 = self.y1 + h
        self.win = tk.Toplevel(root); self.win.overrideredirect(True)
        self.win.attributes("-topmost", True); self.win.attributes("-alpha", 0.92)
        self.win.geometry(f"{w}x{h}+{self.x1}+{self.y1}"); self.win.configure(bg="#2a1a1a")
        c = tk.Canvas(self.win, width=w, height=h, bg="#2a1a1a",
                      highlightthickness=1, highlightbackground="#c05a5a"); c.pack()
        c.create_text(w//2, h//2, text="✕  여기에 놓으면 종료",
                      fill="#e08a8a", font=("맑은 고딕", 11))
    def contains(self, gx, gy):
        return self.x1 <= gx <= self.x2 and self.y1 <= gy <= self.y2
    def destroy(self): self.win.destroy()


# ── 원형 메뉴 (테두리 은은하게) ──────────────────────────────
class RadialMenu:
    def __init__(self, root, cx, cy, actions, on_enter, on_leave):
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True); self.win.attributes("-topmost", True)
        try:
            self.win.attributes("-transparentcolor", TRANSP); bg = TRANSP
        except tk.TclError:
            bg = BG
        W = H = 200
        self.win.geometry(f"{W}x{H}+{cx-W//2}+{cy-H//2}")
        self.canvas = tk.Canvas(self.win, width=W, height=H, bg=bg, highlightthickness=0)
        self.canvas.pack()
        positions = [(W//2, 36), (44, H-44), (W-44, H-44)]
        labels = [("📁", "업로드"), ("🕘", "기록"), ("🎨", "로고")]
        r = 26
        for (px, py), (icon, name), action in zip(positions, labels, actions):
            # 테두리 없이 채움 + 은은한 구분
            self.canvas.create_oval(px-r, py-r, px+r, py+r, fill=CARD, outline=BORDER, width=1)
            tid = self.canvas.create_text(px, py-4, text=icon, font=("맑은 고딕", 16))
            ntid = self.canvas.create_text(px, py+13, text=name, fill=MUTED, font=("맑은 고딕", 7))
            hit = self.canvas.create_oval(px-r, py-r, px+r, py+r, fill="", outline="")
            for item in (tid, ntid, hit):
                self.canvas.tag_bind(item, "<Button-1>", lambda e, a=action: a())
        self.win.bind("<Enter>", lambda e: on_enter())
        self.win.bind("<Leave>", lambda e: on_leave())
    def destroy(self): self.win.destroy()


# ── 플로팅 버튼 ──────────────────────────────────────────────
class FloatingButton:
    def __init__(self, root):
        self.root = root; self.trash = None; self.menu = None
        self._drag_moved = False; self._close_timer = None
        self.size = 56
        sw = root.winfo_screenwidth(); sh = root.winfo_screenheight()
        self.x, self.y = sw - 120, sh - 160
        self.win = tk.Toplevel(root); self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.geometry(f"{self.size}x{self.size}+{self.x}+{self.y}")
        try:
            self.win.attributes("-transparentcolor", TRANSP); bg = TRANSP
        except tk.TclError:
            bg = BG
        self.canvas = tk.Canvas(self.win, width=self.size, height=self.size,
                                bg=bg, highlightthickness=0)
        self.canvas.pack()
        # 시작 시 저장된 커스텀 로고가 있으면 불러오기
        self.icon = ImageTk.PhotoImage(self._load_icon())
        self.img_id = self.canvas.create_image(self.size//2, self.size//2, image=self.icon)
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Double-Button-1>", self._double)
        self.win.bind("<Enter>", lambda e: self._hover_enter())
        self.win.bind("<Leave>", lambda e: self._hover_leave())

    def _load_icon(self):
        """저장된 커스텀 로고가 있으면 그걸로, 없으면 기본 +."""
        if os.path.exists(LOGO_PATH):
            try:
                return make_logo_from_image(Image.open(LOGO_PATH), self.size)
            except Exception:
                pass
        return make_plus_icon(self.size)

    # 호버
    def _hover_enter(self):
        if self._close_timer:
            self.root.after_cancel(self._close_timer); self._close_timer = None
        if self.menu is None:
            cx = self.win.winfo_x() + self.size//2
            cy = self.win.winfo_y() + self.size//2
            self.menu = RadialMenu(self.root, cx, cy,
                actions=[self._do_upload, self._do_history, self._do_logo],
                on_enter=self._hover_enter, on_leave=self._hover_leave)
    def _hover_leave(self):
        if self._close_timer: self.root.after_cancel(self._close_timer)
        self._close_timer = self.root.after(350, self._close_menu)
    def _close_menu(self):
        if self.menu: self.menu.destroy(); self.menu = None
        self._close_timer = None

    # 업로드 판별
    def _do_upload(self):
        self._close_menu()
        path = filedialog.askopenfilename(title="이미지 선택",
            filetypes=[("이미지", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if not path: return
        img = Image.open(path)
        LoadingWindow(self.root, img,
                      on_complete=lambda c, s, r, f: record_and_show(self.root, c, s, r, f))

    # 히스토리
    def _do_history(self):
        self._close_menu(); show_history(self.root)

    # 로고 커스텀 (+ 영구 저장, 되돌리기)
    def _do_logo(self):
        self._close_menu()
        win = tk.Toplevel(self.root); win.title("로고 설정")
        win.configure(bg=CARD); win.geometry("260x150"); win.attributes("-topmost", True)
        tk.Label(win, text="🎨 로고 설정", bg=CARD, fg=TEXT,
                 font=("맑은 고딕", 12, "bold")).pack(pady=(18, 12))
        def choose():
            path = filedialog.askopenfilename(title="로고 이미지 선택",
                filetypes=[("이미지", "*.png *.jpg *.jpeg *.bmp *.webp")])
            if not path: return
            img = Image.open(path)
            # 화면 표시 + 파일로 저장(다음에 켜도 유지)
            logo = make_logo_from_image(img, self.size)
            self.icon = ImageTk.PhotoImage(logo)
            self.canvas.itemconfig(self.img_id, image=self.icon)
            try:
                logo.save(LOGO_PATH)
            except Exception as ex:
                print("로고 저장 실패:", ex)
            win.destroy()
        def revert():
            self.icon = ImageTk.PhotoImage(make_plus_icon(self.size))
            self.canvas.itemconfig(self.img_id, image=self.icon)
            if os.path.exists(LOGO_PATH):
                try: os.remove(LOGO_PATH)
                except Exception: pass
            win.destroy()
        tk.Button(win, text="이미지 선택해서 로고 바꾸기", command=choose,
                  bg=ACCENT, fg="#101018", relief="flat", font=("맑은 고딕", 10, "bold"),
                  padx=8, pady=6).pack(pady=4, padx=20, fill="x")
        tk.Button(win, text="기본 로고로 되돌리기", command=revert,
                  bg=CARD, fg=MUTED, relief="solid", bd=1, font=("맑은 고딕", 9),
                  padx=8, pady=4).pack(pady=4, padx=20, fill="x")

    # 드래그/더블클릭
    def _press(self, e):
        self._drag_moved = False; self._off_x, self._off_y = e.x, e.y
    def _drag(self, e):
        self._drag_moved = True; self._close_menu()
        nx = self.win.winfo_x() + e.x - self._off_x
        ny = self.win.winfo_y() + e.y - self._off_y
        self.win.geometry(f"+{nx}+{ny}")
        if self.trash is None: self.trash = TrashZone(self.root)
        self.win.lift()
    def _release(self, e):
        if self._drag_moved and self.trash is not None:
            gx = self.win.winfo_x() + self.size//2
            gy = self.win.winfo_y() + self.size//2
            if self.trash.contains(gx, gy):
                self.root.quit(); return
        if self.trash is not None:
            self.trash.destroy(); self.trash = None
    def _double(self, e):
        self._close_menu(); self.win.withdraw()
        self.root.after(200, lambda: RegionSelector(self.root, self._on_region))
    def _on_region(self, x1, y1, x2, y2):
        img = grab_region(x1, y1, x2, y2)
        self.win.deiconify()
        LoadingWindow(self.root, img,
                      on_complete=lambda c, s, r, f: record_and_show(self.root, c, s, r, f))


def main():
    root = tk.Tk(); root.withdraw()
    FloatingButton(root)
    root.mainloop(); root.destroy()


if __name__ == "__main__":
    main()