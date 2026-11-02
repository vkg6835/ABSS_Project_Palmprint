#!/usr/bin/env python3
"""
demo.py — the live demonstration for Phase II.

    python demo.py                 synthetic hands (generated on first run)
    python demo.py --data PATH     any dataset folder, see palmprint/engine.py
    python demo.py --check         rehearse every scenario automatically and
                                   save screenshots and a report to _screens/
    python demo.py --fullscreen    start in full screen

Double-clicking run_demo.bat does the first of these.

Keys — a presentation clicker's Page Down and Page Up work too:
    Space  Right  Page Down   next step        Left  Page Up   previous step
    A   show every step       R   back to step 1
    G   genuine attempt       I   impostor     C   closest impostor
    U   unreadable image      O   open an image
    E   enrol a new person
    T   verify / identify     M   RLOC / Han et al. 2003
    H   hide or show the controls    F11   full screen    Esc   leave it

Every decision is made by palmprint/engine.py and the picture is drawn by
palmprint/stage.py; this file is only the window around them. That is why it
is short, and why everything it displays was tested without a screen.
"""
import argparse
import datetime
import os
import queue
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import tkinter as tk
    import tkinter.font as tkfont
    from tkinter import filedialog, messagebox, simpledialog, ttk
except ImportError:                      # reported properly in main()
    tk = None

import numpy as np                       # noqa: E402
import PIL                               # noqa: E402

import make_demo_data                    # noqa: E402
from palmprint import engine as E        # noqa: E402
from palmprint import stage as S         # noqa: E402

DEFAULT_DATA = os.path.join(HERE, "demo_data", "synthetic")
LOG_PATH = os.path.join(HERE, "demo_error.log")
SCREENS = os.path.join(HERE, "_screens")
IMG_TYPES = [("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.pgm"),
             ("All files", "*.*")]

INK, SLATE, MUTED, FAINT = "#1F2933", "#4A5560", "#6B7280", "#9AA1A9"
RULE, LIGHT, PALE, WHITE = "#D5DAE0", "#EDEFF2", "#F6F7F8", "#FFFFFF"


def _dpi_aware():
    """
    Ask Windows for crisp, unscaled rendering. Without this, a laptop at
    125% or 150% display scaling stretches the whole window as a bitmap and
    every line and letter comes out blurred — on a projector, for an
    audience. Must be called before the first window is created.
    """
    if sys.platform != "win32":
        return "not Windows"
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        return "system-aware (shcore)"
    except Exception:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
            return "system-aware (user32)"
        except Exception:
            return "unavailable"


def _photo(img):
    """PIL image -> Tk image. A function of its own so tests can stub it."""
    from PIL import ImageTk
    return ImageTk.PhotoImage(img)


class App(object):
    def __init__(self, root, args, dpi="unknown"):
        self.root = root
        self.args = args
        self.dpi = dpi
        self.exit_code = 0
        self.eng = None
        self.q = queue.Queue()
        self.trace = None
        self.truth = None
        self.note = None
        self.results = {}
        self.step = 0
        self.controls_on = True
        self.fullscreen = bool(args.fullscreen)
        self.render_ms = 0.0
        self._photo_ref = None
        self._render_job = None
        self._size = (0, 0)
        self._font_px = None
        self._last_dir = None
        self._last_stage = None
        self.check_errors = []

        self.task = tk.StringVar(value="verify")
        self.matcher = tk.StringVar(value="rloc")
        self.claim = tk.StringVar(value="")
        self.status_left = tk.StringVar(value="")
        self.status_right = tk.StringVar(value="")
        self.probe_text = tk.StringVar(value="no probe loaded")
        self.load_msg = tk.StringVar(value="starting")

        root.title("Palmprint verification  -  AI461 Phase II")
        root.configure(bg=WHITE)
        root.minsize(900, 600)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.report_callback_exception = self._on_error
        self._init_fonts()
        self._init_window()

    # ============================================================ set-up
    def _init_fonts(self):
        fams = set(tkfont.families(self.root))
        for fam in ("Segoe UI", "Calibri", "Arial", "DejaVu Sans",
                    "Liberation Sans"):
            if fam in fams:
                break
        else:
            fam = tkfont.nametofont("TkDefaultFont").actual("family")
        self.family = fam
        # Negative sizes are pixels, so the interface scales with the
        # window rather than with the display's DPI setting.
        self.f_ui = tkfont.Font(root=self.root, family=fam, size=-14)
        self.f_bold = tkfont.Font(root=self.root, family=fam, size=-14,
                                  weight="bold")
        self.f_small = tkfont.Font(root=self.root, family=fam, size=-12)
        self.f_title = tkfont.Font(root=self.root, family=fam, size=-21,
                                   weight="bold")
        st = ttk.Style(self.root)
        st.configure(".", font=self.f_ui)
        st.configure("TFrame", background=WHITE)
        st.configure("TLabel", background=WHITE, foreground=INK)
        st.configure("TRadiobutton", background=WHITE, foreground=INK)
        st.configure("Head.TLabel", font=self.f_bold, foreground=SLATE)
        st.configure("Muted.TLabel", font=self.f_small, foreground=MUTED)
        st.configure("TButton", padding=(8, 3))
        self.style = st

    def _scale_fonts(self, window_h):
        """Legible from the back of a room on a projector, without being
        oversized on a laptop."""
        px = int(max(12, min(18, round(window_h / 54.0))))
        if px == self._font_px:
            return
        self._font_px = px
        self.f_ui.configure(size=-px)
        self.f_bold.configure(size=-px)
        self.f_small.configure(size=-max(11, int(round(px * 0.86))))
        self.f_title.configure(size=-int(round(px * 1.45)))
        if hasattr(self, "probe_label"):
            wrap = max(170, px * 15)
            self.probe_label.configure(wraplength=wrap)
            self.keys_label.configure(wraplength=wrap)

    def _init_window(self):
        a = self.args
        if a.check:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            self.root.geometry("%dx%d+10+10" % (min(1366, sw - 40),
                                                min(768, sh - 110)))
        elif a.window:
            self.root.geometry(a.window)
        else:
            try:
                self.root.state("zoomed")                    # Windows
            except tk.TclError:
                try:
                    self.root.attributes("-zoomed", True)    # X11
                except tk.TclError:
                    pass
        if self.fullscreen:
            self.root.attributes("-fullscreen", True)

    # ======================================================= loading screen
    def start(self):
        self.loading = ttk.Frame(self.root, padding=40)
        self.loading.place(relx=0.5, rely=0.45, anchor="center")
        ttk.Label(self.loading, text="Palmprint verification",
                  font=self.f_title).pack(anchor="w")
        ttk.Label(self.loading,
                  text="AI461 Advanced Biometric Systems and Security  ·  "
                       "Phase II", style="Muted.TLabel"
                  ).pack(anchor="w", pady=(2, 18))
        ttk.Label(self.loading, textvariable=self.load_msg).pack(anchor="w")
        self.load_bar = ttk.Progressbar(self.loading, length=420,
                                        maximum=100.0, mode="determinate")
        self.load_bar.pack(anchor="w", pady=(8, 0))
        threading.Thread(target=self._worker, daemon=True).start()
        self.root.after(100, self._poll)

    def _worker(self):
        """Runs off the UI thread, so the window stays alive while a first
        start-up preprocesses every image. Talks to the UI only through
        the queue, because Tk is not thread-safe."""
        try:
            data = os.path.abspath(self.args.data)
            base = 0.0
            if not os.path.isdir(data):
                if data != os.path.abspath(DEFAULT_DATA):
                    raise E.EngineError("dataset folder not found: %s" % data)
                base = 0.45

                def gen(done, total):
                    self.q.put(("msg", base * done / float(total),
                                "first run: generating synthetic hand %d of %d"
                                % (done, total)))
                make_demo_data.make(data, progress=gen)
            eng = E.Engine(data)

            def prog(frac, msg):
                self.q.put(("msg", base + (1.0 - base) * frac, msg))
            eng.prepare(progress=prog, rebuild=self.args.rebuild,
                        max_palms=self.args.palms)
            self.q.put(("ready", eng))
        except Exception as e:
            self.q.put(("error", "%s: %s" % (e.__class__.__name__, e),
                        traceback.format_exc()))

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "msg":
                    self.load_bar["value"] = 100.0 * item[1]
                    self.load_msg.set(item[2])
                elif item[0] == "ready":
                    self.eng = item[1]
                    self.loading.destroy()
                    self._build_main()
                    if self.args.check:
                        self.root.after(300, self.run_check)
                    return
                elif item[0] == "error":
                    self._log(item[2])
                    if self.args.check:
                        print("FAILED TO START: %s" % item[1])
                        print(item[2])
                    else:
                        messagebox.showerror(
                            "Could not start the demo",
                            "%s\n\nDetails were written to demo_error.log."
                            % item[1], parent=self.root)
                    self.exit_code = 1
                    self.quit()
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ========================================================= main window
    def _build_main(self):
        r = self.root
        r.grid_columnconfigure(0, weight=1)
        r.grid_rowconfigure(2, weight=1)
        self._scale_fonts(r.winfo_height())

        head = tk.Frame(r, bg=WHITE)
        head.grid(row=0, column=0, sticky="ew", padx=18, pady=(10, 6))
        head.grid_columnconfigure(1, weight=1)
        tk.Label(head, text="Palmprint verification", font=self.f_title,
                 bg=WHITE, fg=INK).grid(row=0, column=0, sticky="w")
        tk.Label(head, text="RLOC: Jia, Huang & Zhang (2008)   ·   baseline: "
                            "Han et al. (2003)   ·   AI461 Phase II, "
                            "SVNIT Surat",
                 font=self.f_small, bg=WHITE, fg=MUTED
                 ).grid(row=1, column=0, sticky="w")
        tk.Label(head, text=self._badge(), font=self.f_small, bg=LIGHT,
                 fg=SLATE, padx=10, pady=4
                 ).grid(row=0, column=2, rowspan=2, sticky="e")
        tk.Frame(r, bg=RULE, height=1).grid(row=1, column=0, sticky="ew")

        body = tk.Frame(r, bg=WHITE)
        body.grid(row=2, column=0, sticky="nsew")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        self._build_controls(body)

        self.stage_box = tk.Frame(body, bg=WHITE, bd=0, highlightthickness=0)
        self.stage_box.grid(row=0, column=1, sticky="nsew", padx=(4, 6),
                            pady=4)
        self.stage = tk.Label(self.stage_box, bg=WHITE, bd=0,
                              highlightthickness=0)
        self.stage.place(x=0, y=0, relwidth=1, relheight=1)
        self.stage_box.bind("<Configure>", self._on_resize)

        tk.Frame(r, bg=RULE, height=1).grid(row=3, column=0, sticky="ew")
        bar = tk.Frame(r, bg=PALE)
        bar.grid(row=4, column=0, sticky="ew")
        tk.Label(bar, textvariable=self.status_left, font=self.f_small,
                 bg=PALE, fg=SLATE, anchor="w"
                 ).pack(side="left", padx=14, pady=4)
        tk.Label(bar, textvariable=self.status_right, font=self.f_small,
                 bg=PALE, fg=MUTED, anchor="e"
                 ).pack(side="right", padx=14, pady=4)

        self._refresh_identities()
        self._bind_keys()
        self.refresh()
        self._refocus()

    def _badge(self):
        m = self.eng.manifest
        if m.get("synthetic"):
            return "SYNTHETIC DATA  ·  not real palms"
        return "Dataset: %s" % self.eng.name

    def _button(self, parent, text, command, row, column=0, span=2):
        def run():
            command()
            self._refocus()
        # width=0: size to the text. The Windows theme otherwise gives every
        # button a minimum of about 11 characters, which made three short
        # buttons on one row push the whole column to 360 px wide and shrink
        # every panel — seen in the second --check run's screenshots.
        b = ttk.Button(parent, text=text, command=run, takefocus=0, width=0)
        b.grid(row=row, column=column, columnspan=span, sticky="ew", pady=1,
               padx=((0, 3) if (span == 1 and column == 0) else
                     (3, 0) if span == 1 else 0))
        return b

    def _build_controls(self, body):
        """
        The control column, packed tight enough to fit a 1280 x 720 window
        with room to spare: paired choices share a row, and so do the four
        examples and the three step buttons. (The first version stacked one
        control per row and was cut off at the bottom of every window up to
        768 pixels tall — found by the --check screenshots on the real
        laptop, which is what they are for.)
        """
        c = ttk.Frame(body, padding=(16, 8, 10, 8))
        c.grid(row=0, column=0, sticky="ns")
        c.grid_columnconfigure(0, weight=1, uniform="half")
        c.grid_columnconfigure(1, weight=1, uniform="half")
        self.controls = c
        row = [0]

        def heading(text):
            ttk.Label(c, text=text, style="Head.TLabel").grid(
                row=row[0], column=0, columnspan=2, sticky="w",
                pady=(8 if row[0] else 0, 2))
            row[0] += 1

        def radio_pair(var, left, right):
            for col, (text, value) in enumerate((left, right)):
                ttk.Radiobutton(c, text=text, value=value, variable=var,
                                command=self._changed, takefocus=0).grid(
                    row=row[0], column=col, sticky="w")
            row[0] += 1

        heading("Task")
        radio_pair(self.task, ("Verify  1:1", "verify"),
                   ("Identify  1:N", "identify"))
        heading("Matcher")
        radio_pair(self.matcher, ("RLOC  2008", "rloc"), ("Han  2003", "han"))

        heading("Claimed identity")
        self.combo = ttk.Combobox(c, textvariable=self.claim,
                                  state="readonly", width=18,
                                  font=self.f_ui, takefocus=0)
        self.combo.grid(row=row[0], column=0, columnspan=2, sticky="ew")
        self.combo.bind("<<ComboboxSelected>>", lambda e: self._changed())
        row[0] += 1

        heading("Probe")
        self._button(c, "Open an image…  (O)", self.open_probe, row[0])
        row[0] += 1
        self.probe_label = ttk.Label(c, textvariable=self.probe_text,
                                     style="Muted.TLabel", wraplength=210,
                                     justify="left")
        self.probe_label.grid(row=row[0], column=0, columnspan=2,
                              sticky="w", pady=(2, 0))
        row[0] += 1

        heading("Quick examples")
        have = self.eng.examples()
        for k, (key, text) in enumerate((("genuine", "Genuine  (G)"),
                                         ("impostor", "Impostor  (I)"),
                                         ("hardest", "Closest  (C)"),
                                         ("unreadable", "Unreadable  (U)"))):
            b = self._button(c, text, lambda key=key: self.example(key),
                             row[0], column=k % 2, span=1)
            if key not in have:
                b.state(["disabled"])
            if k % 2:
                row[0] += 1

        heading("Steps")
        nav = ttk.Frame(c)
        nav.grid(row=row[0], column=0, columnspan=2, sticky="ew")
        row[0] += 1
        for k, (text, fn) in enumerate((("‹  Back", self.prev_step),
                                        ("Next  ›", self.next_step),
                                        ("All  (A)", self.show_all))):
            ttk.Button(nav, text=text, takefocus=0, width=0,
                       command=lambda fn=fn: (fn(), self._refocus())
                       ).pack(side="left", fill="x", expand=True,
                              padx=(0 if k == 0 else 4, 0))

        heading("Enrolment")
        self._button(c, "Enrol a new person…  (E)", self.enrol_dialog, row[0])
        row[0] += 1

        c.grid_rowconfigure(row[0], weight=1)
        row[0] += 1
        self.keys_label = ttk.Label(
            c, text="Space next step  ·  H hide these controls  ·  "
                    "F11 full screen", style="Muted.TLabel", wraplength=210,
            justify="left")
        self.keys_label.grid(row=row[0], column=0, columnspan=2, sticky="sw")

    def _bind_keys(self):
        pairs = [
            (("space", "Right", "Next", "Down", "Return"), self.next_step),
            (("Left", "Prior", "Up", "BackSpace"), self.prev_step),
            (("a", "A"), self.show_all),
            (("r", "R"), self.restart),
            (("g", "G"), lambda: self.example("genuine")),
            (("i", "I"), lambda: self.example("impostor")),
            (("c", "C"), lambda: self.example("hardest")),
            (("u", "U"), lambda: self.example("unreadable")),
            (("o", "O"), self.open_probe),
            (("e", "E"), self.enrol_dialog),
            (("t", "T"), self.toggle_task),
            (("m", "M"), self.toggle_matcher),
            (("h", "H"), self.toggle_controls),
            (("F11",), self.toggle_fullscreen),
            (("Escape",), self.leave_fullscreen),
        ]
        for keys, fn in pairs:
            for k in keys:
                self.root.bind("<KeyPress-%s>" % k,
                               lambda e, fn=fn: self._key(e, fn))

    def _key(self, event, fn):
        if self.eng is None or self._typing(event):
            return None
        fn()
        return "break"

    @staticmethod
    def _typing(event):
        """True when a key belongs to a widget that uses it itself."""
        w = event.widget
        try:
            cls = w.winfo_class()
        except Exception:
            return True
        return cls in ("TCombobox", "Entry", "TEntry", "Text", "Listbox",
                       "Spinbox", "TSpinbox")

    def _refocus(self):
        """Hand the keyboard back to the stage after any click, so the
        clicker's next key steps the demo instead of pressing a button."""
        try:
            self.stage.focus_set()
        except Exception:
            pass

    # ============================================================== state
    def _refresh_identities(self):
        ids = list(self.eng.identities)
        self.combo.configure(values=ids)
        if self.claim.get() not in ids and ids:
            self.claim.set(ids[0])

    def _max_step(self):
        if self.trace is None:
            return 0
        if self.note is not None:
            return 2
        return S.max_step(self.trace)

    def _source(self):
        return self.trace.source if self.trace is not None else None

    def _result(self):
        if self.trace is None or self.note is not None:
            return None
        task, m = self.task.get(), self.matcher.get()
        claim = self.claim.get() if task == "verify" else None
        key = (task, m, claim)
        if key not in self.results:
            if task == "verify":
                if claim not in self.eng.identities:
                    return None
                self.results[key] = self.eng.verify(self.trace, claim, m)
            else:
                self.results[key] = self.eng.identify(self.trace, m)
        return self.results[key]

    def refresh(self):
        top = self._max_step()
        lo = 0 if self.trace is None else 1
        self.step = max(lo, min(self.step, top))
        self.combo.configure(state="readonly" if self.task.get() == "verify"
                             else "disabled")
        self._status()
        self._render_now()

    def _render_now(self):
        w = self.stage_box.winfo_width()
        h = self.stage_box.winfo_height()
        if w < 60 or h < 60:
            return
        spec = S.build(self.eng, self.trace, self.task.get(),
                       self.matcher.get(), self.claim.get(), self.step,
                       self._result(), self.truth, self.note)
        t0 = time.perf_counter()
        img = S.compose(spec, w, h)
        self.render_ms = 1000.0 * (time.perf_counter() - t0)
        self._last_stage = img
        self._photo_ref = _photo(img)
        self.stage.configure(image=self._photo_ref)

    def _on_resize(self, event):
        size = (event.width, event.height)
        if size == self._size:
            return
        self._size = size
        self._scale_fonts(self.root.winfo_height())
        if self._render_job is not None:
            self.root.after_cancel(self._render_job)
        self._render_job = self.root.after(120, self._resized)

    def _resized(self):
        self._render_job = None
        self._render_now()

    def _status(self):
        s = self.eng.summary()
        parts = [s["name"], "%d enrolled" % len(self.eng.identities)]
        r, h = s["rloc"], s["han"]
        cross = s.get("protocol") == "cross-session"
        if r["genuine"]:
            parts.append("%s genuine and %s impostor%s comparisons"
                         % (format(r["genuine"], ","),
                            format(r["impostor"], ","),
                            " cross-session" if cross else ""))
        if r["eer"] is not None and h["eer"] is not None:
            parts.append("EER%s: RLOC %.2f %%, Han %.2f %%"
                         % (" (cross-session)" if cross else "",
                            100 * r["eer"], 100 * h["eer"]))
        if s["declined"]:
            parts.append("%d images declined" % s["declined"])
        if self.eng.live_enrolled:
            parts.append("enrolled live: %s"
                         % ", ".join(self.eng.live_enrolled))
        self.status_left.set("   ·   ".join(parts))

        tr = self.trace
        if tr is not None and tr.ms:
            t = ["ROI %.0f ms" % tr.ms.get("roi", 0.0)]
            if "encode" in tr.ms:
                t.append("code %.0f ms" % tr.ms["encode"])
            res = self._result()
            if res is not None and res.get("ms"):
                t.append("match %.1f ms" % res["ms"])
            self.status_right.set("   ·   ".join(t))
        else:
            self.status_right.set("templates %s" % s["cache"])

    # ============================================================ actions
    def load(self, source, claim=None):
        tr = self.eng.process(source)
        self.trace = tr
        self.truth = S.truth_of(self.eng, tr.source)
        self.note = None
        if claim and claim in self.eng.identities:
            self.claim.set(claim)
        self.results = {}
        self.step = 1
        self.probe_text.set(tr.name + ("   (true identity %s)" % self.truth
                                       if self.truth else ""))
        if tr.source:
            self._last_dir = os.path.dirname(tr.source)
        self.refresh()

    def example(self, which):
        ex = self.eng.examples().get(which)
        if not ex:
            self.status_right.set("this dataset has no '%s' example" % which)
            return
        self.load(ex["path"], ex.get("claim"))

    def _visitors_dir(self):
        v = self.eng.manifest.get("visitors")
        p = os.path.join(self.eng.root, v) if v else None
        return p if p and os.path.isdir(p) else self.eng.root

    def open_probe(self):
        path = filedialog.askopenfilename(
            parent=self.root, title="Choose a probe image",
            initialdir=self._last_dir or self.eng.root, filetypes=IMG_TYPES)
        self._refocus()
        if path:
            self.load(path)

    def enrol_dialog(self):
        name = simpledialog.askstring("Enrol a new person",
                                      "Name for the new enrolment:",
                                      parent=self.root)
        self._refocus()
        if not name or not name.strip():
            return
        path = filedialog.askopenfilename(
            parent=self.root, title="Image to enrol %s from" % name.strip(),
            initialdir=self._visitors_dir(), filetypes=IMG_TYPES)
        self._refocus()
        if path:
            self.enrol(name.strip(), path)

    def enrol(self, name, path):
        existed = name in self.eng.identities
        tr = self.eng.enrol(name, path)
        self.trace = tr
        self.truth = None
        self.results = {}
        self.step = 2
        if not tr.ok:
            self.note = ["Could not enrol %s" % name,
                         "The image was declined: %s." % tr.reason,
                         "Nothing has been stored. Try an image with the "
                         "whole hand in view and the fingers spread."]
            self.probe_text.set("enrolment of %s declined" % name)
            self.refresh()
            return False
        self._refresh_identities()
        self.claim.set(name)
        self.task.set("verify")
        n = len(self.eng.cfg.rotations)
        self.note = [
            "Enrolled %s" % name,
            "The ROI on the left was encoded and stored as %d rotated "
            "templates%s." % (n, ", added to the existing ones"
                                 if existed else ""),
            "Next: open a second image of the same hand (O) to verify it, "
            "or switch to identify (T) and try someone who is not enrolled.",
        ]
        self.probe_text.set("enrolled %s from %s" % (name, tr.name))
        self.refresh()
        return True

    def next_step(self):
        if self.trace is not None and self.step < self._max_step():
            self.step += 1
            self.refresh()

    def prev_step(self):
        if self.trace is not None and self.step > 1:
            self.step -= 1
            self.refresh()

    def show_all(self):
        if self.trace is not None:
            self.step = self._max_step()
            self.refresh()

    def restart(self):
        if self.trace is not None:
            self.step = 1
            self.refresh()

    def _changed(self):
        self.refresh()
        self._refocus()

    def toggle_task(self):
        self.task.set("identify" if self.task.get() == "verify"
                      else "verify")
        self._changed()

    def toggle_matcher(self):
        self.matcher.set("han" if self.matcher.get() == "rloc" else "rloc")
        self._changed()

    def toggle_controls(self):
        if self.controls_on:
            self.controls.grid_remove()
        else:
            self.controls.grid()
        self.controls_on = not self.controls_on

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def leave_fullscreen(self):
        if self.fullscreen:
            self.fullscreen = False
            self.root.attributes("-fullscreen", False)

    def quit(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    # ============================================================= errors
    def _on_error(self, exc, val, tb):
        """
        Any error inside a button or key handler lands here instead of
        killing the window: logged in full, summarised on screen, and the
        demo carries on. On stage, a message box beats a vanished window.
        """
        text = "".join(traceback.format_exception(exc, val, tb))
        self._log(text)
        if self.args.check:
            self.check_errors.append(text)
            return
        try:
            messagebox.showerror(
                "Something went wrong",
                "%s: %s\n\nThe demo is still running. Details are in "
                "demo_error.log." % (exc.__name__, val), parent=self.root)
        except Exception:
            pass

    def _log(self, text):
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write("\n=== %s ===\n%s\n"
                        % (datetime.datetime.now().isoformat(
                            timespec="seconds"), text))
        except OSError:
            pass

    # ========================================================= rehearsal
    def run_check(self):
        """
        Drive every scenario of the live demo, as a presenter would, and
        record what happened: a screenshot of the real window for each, and
        a pass/fail line for each expectation. Run it on the laptop that
        will be used in class, before the day.
        """
        os.makedirs(SCREENS, exist_ok=True)
        # a re-run must not leave stale screenshots mixed in with fresh ones
        for f in os.listdir(SCREENS):
            if f.endswith(".png") or f == "check_report.txt":
                try:
                    os.remove(os.path.join(SCREENS, f))
                except OSError:
                    pass
        self.root.attributes("-topmost", True)
        self.check_log = []
        self.check_n = 0
        ex = self.eng.examples()
        vis = self._visitors_dir()
        va1 = os.path.join(vis, "visitor_A_1.png")
        va2 = os.path.join(vis, "visitor_A_2.png")
        vb1 = os.path.join(vis, "visitor_B_1.png")
        have_visitors = all(os.path.isfile(p) for p in (va1, va2, vb1))

        def setv(task=None, matcher=None):
            if task:
                self.task.set(task)
            if matcher:
                self.matcher.set(matcher)

        def show(which, task="verify", matcher="rloc"):
            setv(task, matcher)
            self.example(which)
            self.show_all()

        plan = []

        def add(name, action, expect=None):
            plan.append((name, action, expect))

        add("welcome", lambda: None, lambda: self._ok(self.trace is None,
                                                      "nothing loaded"))
        if "genuine" in ex:
            add("genuine_step1",
                lambda: (setv("verify", "rloc"), self.example("genuine")),
                lambda: self._ok(self.step == 1, "step %d" % self.step))
            for k in (2, 3, 4, 5):
                add("genuine_step%d" % k, self.next_step,
                    lambda k=k: self._ok(self.step == k,
                                         "step %d" % self.step))
            add("genuine_decision", lambda: None,
                lambda: self._all(self._decision_is("accept"),
                                  self._controls_fit()))
            add("back_to_step4", self.prev_step,
                lambda: self._ok(self.step == 4, "step %d" % self.step))
        if "impostor" in ex:
            add("impostor", lambda: show("impostor"),
                lambda: self._decision_is("reject"))
        if "hardest" in ex:
            add("closest_impostor", lambda: show("hardest"),
                self._has_decision)
        if "unreadable" in ex:
            add("unreadable", lambda: show("unreadable"),
                lambda: self._ok(self.trace is not None and not self.trace.ok
                                 and self.step == 2, "declined at step %d"
                                 % self.step))
        if "genuine" in ex:
            add("identify_genuine", lambda: show("genuine", "identify"),
                lambda: self._ok((self._result() or {}).get("identity")
                                 == self.truth, "identified as %s, truth %s"
                                 % ((self._result() or {}).get("identity"),
                                    self.truth)))
            add("han_genuine", lambda: show("genuine", "verify", "han"),
                lambda: self._decision_is("accept"))
        if "impostor" in ex:
            add("han_identify_impostor_probe",
                lambda: show("impostor", "identify", "han"),
                self._has_decision)
        if have_visitors:
            add("enrol_visitor_A",
                lambda: (setv("verify", "rloc"),
                         self.enrol("Visitor A", va1)),
                lambda: self._ok("Visitor A" in self.eng.identities,
                                 "enrolled: %s" % self.eng.live_enrolled))
            add("verify_visitor_A",
                lambda: (setv("verify", "rloc"), self.claim.set("Visitor A"),
                         self.load(va2), self.show_all()),
                lambda: self._decision_is("accept"))
            add("identify_visitor_B_not_enrolled",
                lambda: (setv("identify", "rloc"), self.load(vb1),
                         self.show_all()),
                lambda: self._decision_is("reject"))
        if "genuine" in ex:
            add("controls_hidden",
                lambda: (self.toggle_controls(),
                         show("genuine", "verify", "rloc")),
                lambda: self._decision_is("accept"))
            add("controls_back", self.toggle_controls, None)

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.check_skipped = []

        def at_size(fn):
            def go():
                fn()
                show("genuine", "verify", "rloc")
            return go

        def fits_and_accepts():
            return self._all(self._decision_is("accept"),
                             self._controls_fit())

        for w, h in ((1024, 768), (1280, 720), (1366, 768), (1600, 900)):
            if w <= sw - 20 and h <= sh - 90:
                add("size_%dx%d" % (w, h),
                    at_size(lambda w=w, h=h: self.root.geometry(
                        "%dx%d+10+10" % (w, h))),
                    fits_and_accepts)
            else:
                self.check_skipped.append(
                    ("skip", "size_%dx%d" % (w, h),
                     "does not fit on this %dx%d screen" % (sw, sh)))
        # the two ways the demo is really run on this laptop
        add("maximized", at_size(lambda: self._set_window("zoomed")),
            fits_and_accepts)
        add("fullscreen", at_size(lambda: self._set_window("full")),
            fits_and_accepts)
        add("window_restored",
            lambda: (self._set_window("normal"),
                     self.root.geometry("1366x768+10+10")), None)
        self._check_step(plan, 0)

    def _set_window(self, mode):
        """normal, zoomed (maximised) or full (full screen)."""
        if mode == "full":
            self.fullscreen = True
            self.root.attributes("-fullscreen", True)
            return
        if self.fullscreen:
            self.fullscreen = False
            self.root.attributes("-fullscreen", False)
        try:
            self.root.state("zoomed" if mode == "zoomed" else "normal")
        except tk.TclError:
            pass

    @staticmethod
    def _all(*results):
        return (all(ok for ok, _ in results),
                "; ".join(msg for _, msg in results if msg))

    def _controls_fit(self):
        """
        Is the control column cut off, or hogging the screen? Tk knows the
        first: a widget given less height than it asked for is being clipped.
        The second is a share of the window width — every pixel the controls
        take is a pixel the audience's panels lose.
        """
        if not self.controls_on:
            return True, ""
        self.root.update_idletasks()
        need = self.controls.winfo_reqheight()
        got = self.controls.winfo_height()
        cw = self.controls.winfo_width()
        ww = max(1, self.root.winfo_width())
        ok = need <= got and cw <= 0.28 * ww
        return ok, ("controls need %d px, have %d px, %d px wide (%.0f%% of "
                    "window); window %dx%d"
                    % (need, got, cw, 100.0 * cw / ww, ww,
                       self.root.winfo_height()))

    @staticmethod
    def _ok(cond, msg=""):
        return bool(cond), msg

    def _decision_is(self, want):
        res = self._result()
        got = res.get("decision") if res else None
        right = S.correctness(self.eng, self.task.get(), res, self.truth,
                              self._source()) \
            if res else None
        return got == want, "decision %s%s" % (
            got, ("; " + right) if right else "")

    def _has_decision(self):
        res = self._result()
        got = res.get("decision") if res else None
        right = S.correctness(self.eng, self.task.get(), res, self.truth,
                              self._source()) \
            if res else None
        return got in ("accept", "reject"), "decision %s%s" % (
            got, ("; " + right) if right else "")

    def _check_step(self, plan, i):
        if i >= len(plan):
            self._check_finish()
            return
        name, action, expect = plan[i]
        n_err = len(self.check_errors)
        try:
            action()
        except Exception:
            self.check_errors.append(traceback.format_exc())
        self.root.update_idletasks()
        self.root.after(500, lambda: self._check_capture(plan, i, n_err))

    def _check_capture(self, plan, i, n_err):
        name, action, expect = plan[i]
        self.root.update()
        self._render_now()
        self.root.update()
        self.check_n += 1
        fname = "%02d_%s.png" % (self.check_n, name)
        grabbed, why = self._grab(os.path.join(SCREENS, fname))
        if not grabbed and self._last_stage is not None:
            self._last_stage.save(os.path.join(SCREENS, "stage_" + fname))
        ok, msg = (expect() if expect else (True, ""))
        if len(self.check_errors) > n_err:
            ok = False
            msg += "; error: %s" % self.check_errors[-1].strip(
            ).splitlines()[-1]
        if not grabbed:
            msg += "; screenshot failed (%s)" % why
        msg += "; drawn in %.0f ms" % self.render_ms
        self.check_log.append(("ok" if ok else "FAIL", name, msg))
        self.root.after(80, lambda: self._check_step(plan, i + 1))

    def _grab(self, path):
        try:
            from PIL import ImageGrab
            self.root.update()
            x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
            w, h = self.root.winfo_width(), self.root.winfo_height()
            ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(path)
            return True, ""
        except Exception as e:
            return False, "%s: %s" % (e.__class__.__name__, e)

    def _check_finish(self):
        self.root.attributes("-topmost", False)
        s = self.eng.summary()
        try:
            from palmprint import render as R
            pil_font = " ".join(R.font(16).getname())
        except Exception as e:
            pil_font = "unavailable (%s)" % e
        lines = [
            "AI461 Phase II - demo check   %s"
            % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python %s, numpy %s, Pillow %s, Tk %s, tk scaling %.2f"
            % (sys.version.split()[0], np.__version__, PIL.__version__,
               self.root.tk.call("info", "patchlevel"),
               float(self.root.tk.call("tk", "scaling"))),
            "screen %d x %d, DPI awareness: %s"
            % (self.root.winfo_screenwidth(), self.root.winfo_screenheight(),
               self.dpi),
            "window font: %s;  drawing font: %s" % (self.family, pil_font),
            "dataset: %s, %d enrolled (+%d live during the check), %d probes, "
            "%d declined; templates %s in %.1f s"
            % (s["name"], s["enrolled"] - len(self.eng.live_enrolled),
               len(self.eng.live_enrolled), s["probes"], s["declined"],
               s["cache"], s["seconds"]),
            "-" * 72,
        ]
        n_ok = n_fail = 0
        for status, name, msg in self.check_log + self.check_skipped:
            lines.append("[%-4s] %-34s %s" % (status, name, msg))
            n_ok += status == "ok"
            n_fail += status == "FAIL"
        lines.append("-" * 72)
        lines.append("%d passed, %d failed. Screenshots: %s"
                     % (n_ok, n_fail, SCREENS))
        if self.check_errors:
            lines.append("")
            lines.append("Errors:")
            for t in self.check_errors:
                lines.append(t)
        report = "\n".join(lines)
        with open(os.path.join(SCREENS, "check_report.txt"), "w",
                  encoding="utf-8") as f:
            f.write(report + "\n")
        try:
            print(report)
        except UnicodeEncodeError:
            print(report.encode("ascii", "replace").decode("ascii"))
        self.exit_code = 0 if n_fail == 0 and not self.check_errors else 1
        self.root.after(300, self.quit)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Live palmprint verification demo (AI461 Phase II).")
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="dataset folder (default: the synthetic demo set)")
    ap.add_argument("--check", action="store_true",
                    help="rehearse every scenario and save screenshots")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--window", default=None,
                    help="window size, e.g. 1366x768")
    ap.add_argument("--rebuild", action="store_true",
                    help="ignore cached templates and recompute them")
    ap.add_argument("--palms", type=int, default=None, metavar="N",
                    help="use only the first N people. The demo opens in "
                         "seconds with a few dozen; the full Tongji release "
                         "is tens of minutes and belongs in the experiments.")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if tk is None:
        print("This Python has no tkinter, which the demo window needs.\n"
              "On Windows, re-run the Python installer, choose Modify, and "
              "tick 'tcl/tk and IDLE'.")
        return 2
    dpi = _dpi_aware()
    root = tk.Tk()
    app = App(root, args, dpi)
    app.start()
    root.mainloop()
    return app.exit_code


if __name__ == "__main__":
    sys.exit(main())
