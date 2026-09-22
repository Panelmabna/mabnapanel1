import threading
import queue
import time
import hashlib
import numpy as np
import mss
from PIL import Image
from deep_translator import GoogleTranslator
import tkinter as tk
import logging
import sys

# تنظیم لاگ برای دیدن خطاها در کنسول
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', stream=sys.stdout)
logger = logging.getLogger("LiveTranslator")

try:
    import winocr
except ImportError:
    winocr = None
    print("CRITICAL: winocr is not installed. Please run: pip install winocr")

class LiveSubtitleTranslator:
    def __init__(self, bbox, target_lang="fa"):
        self.bbox = bbox
        self.target_lang = target_lang
        self.running = True
        
        # صف‌های ۱ تایی: فقط آخرین فریم/متن مهم است (حالت لایو)
        self.frame_queue = queue.Queue(maxsize=1)
        self.text_queue = queue.Queue(maxsize=1)
        
        self.last_frame_hash = ""
        self.translation_cache = {}
        self.cache_lock = threading.Lock()
        
        try:
            self.translator = GoogleTranslator(source="auto", target=self.target_lang)
        except Exception as e:
            logger.error(f"Failed to init translator: {e}")
            self.translator = None
            
        self.setup_ui()
        self.start_pipeline()
        
        self.root.mainloop()

    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("Live Translator")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.85)
        self.root.configure(bg="#111111")
        
        width = max(400, self.bbox["width"])
        x = self.bbox["left"]
        y = self.bbox["top"] + self.bbox["height"] + 5 
        
        self.root.geometry(f"{width}x80+{x}+{y}")
        
        self.label = tk.Label(
            self.root,
            text="در انتظار زیرنویس...",
            font=("Segoe UI", 14, "bold"),
            fg="#00FF00",
            bg="#111111",
            wraplength=width-40,
            justify="center",
        )
        self.label.pack(expand=True, fill="both", padx=10, pady=10)
        
        # دکمه ضربدر (Close Button)
        close_button = tk.Button(
            self.root,
            text="X",
            command=self.stop,
            bg="#333333",
            fg="white",
            activebackground="#555555",
            activeforeground="white",
            bd=0,
            font=("Arial", 10, "bold"),
        )
        close_button.place(relx=1.0, x=-4, y=4, anchor="ne")
        
        # قابلیت جابجایی با موس
        self.root.bind("<ButtonPress-1>", self.start_move)
        self.root.bind("<B1-Motion>", self.on_move)
        self.root.bind("<Escape>", lambda e: self.stop())
        
    def start_move(self, event):
        self.mx = event.x
        self.my = event.y
        
    def on_move(self, event):
        x = self.root.winfo_x() + (event.x - self.mx)
        y = self.root.winfo_y() + (event.y - self.my)
        self.root.geometry(f"+{x}+{y}")
        
    def stop(self):
        self.running = False
        try:
            self.root.destroy()
        except:
            pass
        
    def start_pipeline(self):
        threading.Thread(target=self.capture_loop, daemon=True).start()
        threading.Thread(target=self.ocr_loop, daemon=True).start()
        threading.Thread(target=self.translation_loop, daemon=True).start()

    def capture_loop(self):
        with mss.MSS() as sct:
            while self.running:
                try:
                    frame = sct.grab(self.bbox)
                    img_np = np.array(frame)
                    
                    # هش سریع برای جلوگیری از پردازش فریم تکراری
                    try:
                        img = Image.fromarray(img_np).convert('L').resize((32, 32))
                        h = hashlib.md5(img.tobytes()).hexdigest()
                    except:
                        h = ""
                        
                    if h and h == self.last_frame_hash:
                        time.sleep(0.03) # حدود ۳۰ فریم بر ثانیه
                        continue
                        
                    self.last_frame_hash = h
                    
                    if self.frame_queue.full():
                        try: self.frame_queue.get_nowait()
                        except queue.Empty: pass
                    self.frame_queue.put_nowait(img_np)
                    
                    time.sleep(0.03)
                except Exception as e:
                    logger.debug(f"Capture error: {e}")
                    time.sleep(0.1)

    def ocr_loop(self):
        if winocr is None:
            logger.error("winocr is missing")
            return
            
        while self.running:
            try:
                img_np = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            except Exception:
                continue
                
            try:
                # استفاده از حالت sync برای پایداری بیشتر
                result = winocr.recognize_cv2_sync(img_np)
                text = self.extract_text(result)
                
                if text and len(text) > 2:
                    if self.text_queue.full():
                        try: self.text_queue.get_nowait()
                        except queue.Empty: pass
                    self.text_queue.put_nowait(text)
                else:
                    # دیباگ: اگر متنی پیدا نشد، در کنسول چاپ کن تا بفهمیم چرا
                    if result:
                        print(f"[DEBUG OCR] No valid text found. Raw keys: {result.keys() if isinstance(result, dict) else type(result)}")
            except Exception as e:
                print(f"[ERROR OCR] {e}")

    def extract_text(self, result):
        lines = []
        if isinstance(result, dict) and "lines" in result:
            raw_lines = result["lines"]
            
            # تابع کمکی برای استخراج مختصات Y جهت مرتب‌سازی خطوط
            def get_y(line):
                bb = line.get("bounding_box", None)
                if isinstance(bb, list) and len(bb) >= 4:
                    return bb[1] # y
                elif isinstance(bb, dict):
                    return bb.get("top", bb.get("y", 0))
                return 0
                
            sorted_lines = sorted(raw_lines, key=get_y)
            
            for line in sorted_lines:
                if isinstance(line, dict):
                    t = line.get("text", "").strip()
                    if t and not self.is_ui_element(t):
                        lines.append(t)
                elif hasattr(line, "text"):
                    t = getattr(line, "text", "").strip()
                    if t and not self.is_ui_element(t):
                        lines.append(t)
        elif isinstance(result, list):
            for line in result:
                 t = line.get("text", "").strip() if isinstance(line, dict) else str(line)
                 if t and not self.is_ui_element(t):
                     lines.append(t)
                     
        return " ".join(lines).strip()

    def is_ui_element(self, text):
        # فیلتر بسیار ساده‌تر تا زیرنویس‌های واقعی اشتباهاً حذف نشوند
        t = text.lower().strip()
        if len(t) < 2: return True
        
        # فقط کلمات دقیقاً منطبق با دکمه‌های معروف
        ui_words = {"play", "pause", "subscribe", "subscribed", "like", "likes", "next", "search", "menu", "settings", "youtube", "netflix", "skip", "share", "save"}
        if t in ui_words: return True
        
        # اگر متن اصلاً حروف الفبا نداشته باشد (مثلاً فقط عدد یا علامت)
        if not any(c.isalpha() for c in t): return True
        
        return False

    def translation_loop(self):
        while self.running:
            try:
                text = self.text_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            except Exception:
                continue
                
            translated = self.translate(text)
            if translated:
                try:
                    self.root.after(0, self.update_label, translated)
                except tk.TclError:
                    pass 

    def translate(self, text):
        with self.cache_lock:
            if text in self.translation_cache:
                return self.translation_cache[text]
                
        if not self.translator:
            return text
            
        try:
            res = self.translator.translate(text)
            if res and res.strip():
                with self.cache_lock:
                    self.translation_cache[text] = res
                return res
        except Exception as e:
            logger.debug(f"Translation error: {e}")
        return ""

    def update_label(self, text):
        if self.running:
            self.label.config(text=text)
            self.label.update_idletasks()
            h = max(60, self.label.winfo_reqheight() + 20)
            w = self.root.winfo_width()
            x = self.root.winfo_x()
            y = self.root.winfo_y()
            try:
                self.root.geometry(f"{w}x{h}+{x}+{y}")
            except:
                pass

def select_region():
    overlay = tk.Tk()
    overlay.title("ناحیه زیرنویس را انتخاب کنید (بکشید و رها کنید)")
    overlay.attributes("-alpha", 0.3, "-fullscreen", True, "-topmost", True)
    overlay.configure(bg="black")
    canvas = tk.Canvas(overlay, cursor="cross", bg="black", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    
    selection = {"bbox": None, "start_x": 0, "start_y": 0, "rect": None}
    
    def on_press(event):
        selection["start_x"] = event.x
        selection["start_y"] = event.y
        selection["rect"] = canvas.create_rectangle(event.x, event.y, event.x+1, event.y+1, outline="red", width=2)
        
    def on_drag(event):
        if selection["rect"]:
            canvas.coords(selection["rect"], selection["start_x"], selection["start_y"], event.x, event.y)
            
    def on_release(event):
        left = min(selection["start_x"], event.x)
        top = min(selection["start_y"], event.y)
        width = max(50, abs(event.x - selection["start_x"]))
        height = max(20, abs(event.y - selection["start_y"]))
        selection["bbox"] = {"left": left, "top": top, "width": width, "height": height}
        overlay.destroy()
        
    def on_escape(event):
        overlay.destroy()
        
    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    overlay.bind("<Escape>", on_escape)
    
    overlay.mainloop()
    return selection["bbox"]

if __name__ == "__main__":
    region = select_region()
    if region:
        LiveSubtitleTranslator(region)