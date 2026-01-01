import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict

# ---------------- NLP SETUP ----------------
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception:
        pass

download_nltk_data()

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    stop_words = set()

# ---------------- DATASETS FOR NAIVE BAYES ----------------
datasets = {
    "Animals Information": {
        "data": [
            ["Dog","Medium","Black","Yes"],
            ["Dog","Big","White","No"],
            ["Rat","Small","White","Yes"],
            ["Cow","Big","White","Yes"],
            ["Cow","Small","Brown","No"],
            ["Cow","Big","Black","Yes"],
            ["Rat","Big","Brown","No"],
            ["Dog","Small","Brown","Yes"],
            ["Dog","Medium","Brown","Yes"],
            ["Cow","Medium","White","No"],
            ["Dog","Small","Black","Yes"],
            ["Rat","Medium","Black","No"],
            ["Rat","Small","Brown","No"],
            ["Cow","Big","White","Yes"]
        ],
        "cols": ["Animal","Size","Color","Pettable"]
    },
    "Weather and Road Condition": {
        "data": [
            ["Rain","Bad","High","No","Yes"],
            ["Snow","Average","Normal","Yes","No"],
            ["Clear","Bad","Light","No","No"],
            ["Clear","Good","Light","Yes","Yes"],
            ["Snow","Good","Normal","No","No"],
            ["Rain","Average","Light","No","No"],
            ["Rain","Good","Normal","No","No"],
            ["Snow","Bad","High","No","Yes"],
            ["Clear","Good","High","Yes","No"],
            ["Clear","Bad","High","Yes","Yes"]
        ],
        "cols": ["Weather","Road","Traffic","Engine Problem","Accident"]
    },
    "Loan Approval": {
        "data": [
            ["Young","High","Good","Employed","Yes"],
            ["Middle","Low","Poor","Unemployed","No"],
            ["Senior","High","Excellent","Retired","Yes"],
            ["Young","Low","Good","Employed","No"],
            ["Middle","High","Poor","Employed","Yes"],
            ["Senior","Low","Excellent","Retired","No"],
            ["Young","High","Poor","Employed","No"],
            ["Middle","High","Excellent","Employed","Yes"],
            ["Senior","Low","Good","Retired","No"],
            ["Young","Low","Excellent","Employed","Yes"]
        ],
        "cols": ["Age","Income","Credit","Employment","Approved"]
    },
    "Email Spam Detection": {
        "data": [
            ["Yes","Yes","Yes","No","Yes"],
            ["No","No","Yes","Yes","No"],
            ["Yes","Yes","No","No","Yes"],
            ["No","Yes","Yes","Yes","No"],
            ["Yes","No","No","No","Yes"],
            ["No","No","Yes","No","No"],
            ["Yes","Yes","Yes","Yes","No"],
            ["No","Yes","No","No","Yes"],
            ["Yes","No","Yes","Yes","No"],
            ["No","No","No","No","Yes"]
        ],
        "cols": ["Offer","Link","Greeting","Sender Known","Spam"]
    }
}

# ---------------- NAIVE BAYES (Laplace smoothing + proba) ----------------
def naive_bayes_predict_proba(df, feature_count, inputs):
    targets = [row[-1] for row in df]
    classes = sorted(set(targets))
    total = len(df)

    priors = {c: (targets.count(c) + 1) / (total + len(classes)) for c in classes}

    feature_values = [set() for _ in range(feature_count)]
    for row in df:
        for i in range(feature_count):
            feature_values[i].add(row[i])

    per_class_rows = {c: [r for r in df if r[-1] == c] for c in classes}
    likelihoods = defaultdict(float)

    for c in classes:
        rows_c = per_class_rows[c]
        denom = len(rows_c)
        for i in range(feature_count):
            values = feature_values[i]
            V = len(values) or 1
            for v in values:
                count = sum(1 for r in rows_c if r[i] == v)
                likelihoods[(c, i, v)] = (count + 1) / (denom + V)

    scores = {}
    for c in classes:
        score = priors[c]
        for i, v in enumerate(inputs):
            if v == "":
                continue
            V = len(feature_values[i]) or 1
            p = likelihoods.get((c, i, v), 1 / (len(per_class_rows[c]) + V))
            score *= p
        scores[c] = score

    total_score = sum(scores.values()) or 1.0
    return {c: scores[c] / total_score for c in classes}

def naive_bayes_predict(df, features, target_col, inputs):
    probs = naive_bayes_predict_proba(df, len(features), inputs)
    return max(probs, key=probs.get), probs

# ---------------- THEME (Dark + Red Sunset) ----------------
def apply_dark_theme(root):
    root.configure(bg="#0b0e14")
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    # Palette
    BG = "#0b0e14"
    SIDEBAR = "#10131d"
    CARD = "#121621"
    CARD_HI = "#171b28"
    BORDER = "#262b36"
    TEXT = "#e6e7ef"
    MUTED = "#9aa4b2"
    ACCENT = "#ff4d4f"      # base red
    ACCENT_SOFT = "#ff7a7c" # lighter red
    ACCENT_DARK = "#c53a3c" # dark red

    # Base styles
    style.configure(".", background=BG, foreground=TEXT, font=("Segoe UI", 10))
    style.configure("TFrame", background=BG)
    style.configure("Sidebar.TFrame", background=SIDEBAR)
    style.configure("Card.TFrame", background=CARD, relief="flat", borderwidth=0)
    style.map("Card.TFrame", background=[("active", CARD_HI)])

    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Muted.TLabel", background=BG, foreground=MUTED)
    style.configure("Section.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 11))

    style.configure("TButton", background=CARD, foreground=TEXT, padding=9, borderwidth=0)
    style.map("TButton", background=[("active", CARD_HI)], relief=[("pressed", "sunken")])

    style.configure("Accent.TButton", background=ACCENT, foreground=BG, padding=10, borderwidth=0)
    style.map("Accent.TButton", background=[("active", ACCENT_SOFT)])

    style.configure("Sidebar.TButton", background=SIDEBAR, foreground=TEXT, padding=10, borderwidth=0)
    style.map("Sidebar.TButton", background=[("active", "#0e111a")])

    style.configure("TCombobox", fieldbackground=CARD, background=CARD, foreground=TEXT, arrowcolor=TEXT, padding=6)
    style.map("TCombobox", fieldbackground=[("readonly", CARD)], foreground=[("readonly", TEXT)])

    style.configure("Horizontal.TProgressbar", troughcolor=CARD, background=ACCENT, bordercolor=CARD)

    style.configure("Treeview",
                    background=CARD, fieldbackground=CARD, foreground=TEXT,
                    rowheight=24, borderwidth=0)
    style.configure("Treeview.Heading", background=CARD_HI, foreground=TEXT)
    style.map("Treeview", background=[("selected", ACCENT_DARK)], foreground=[("selected", TEXT)])

    style.configure("Status.TLabel", background="#0a0d15", foreground=MUTED, font=("Segoe UI", 9))

    return {
        "BG": BG, "SIDEBAR": SIDEBAR, "CARD": CARD, "CARD_HI": CARD_HI,
        "BORDER": BORDER, "TEXT": TEXT, "MUTED": MUTED,
        "ACCENT": ACCENT, "ACCENT_SOFT": ACCENT_SOFT, "ACCENT_DARK": ACCENT_DARK
    }

# ---------- Tooltips ----------
class Tooltip:
    def __init__(self, widget, text, bg="#11151f", fg="#c9d1d9"):
        self.widget = widget
        self.text = text
        self.bg = bg
        self.fg = fg
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _=None):
        if self.tip: return
        x, y, _, _ = self.widget.bbox("insert") or (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 20
        y += self.widget.winfo_rooty() + 20
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(tw, text=self.text, justify="left",
                       background=self.bg, foreground=self.fg,
                       relief="solid", borderwidth=1, font=("Segoe UI", 9), padx=6, pady=4)
        lbl.pack()

    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None

# ---------------- MAIN APP ----------------
class CombinedApp:
    def __init__(self, root):
        self.root = root
        self.c = apply_dark_theme(root)

        self.root.title("🧠 NLP + Naive Bayes • Dark (Red Sunset)")
        self.root.geometry("1220x780")
        self.root.minsize(1120, 720)

        self.mode = tk.StringVar(value="NLP")  # NLP / NB

        self._build_layout()
        self._build_sidebar()
        self._build_toolbar()
        self._build_cards()
        self._show_mode(self.mode.get())
        self._set_status("Ready.")

        # Shortcuts
        self.root.bind("<Control-Return>", lambda e: self._run_action())
        self.root.bind("<F1>", lambda e: self._show_mode("NLP"))
        self.root.bind("<F2>", lambda e: self._show_mode("NB"))

    # ----- Layout shell -----
    def _build_layout(self):
        self.shell = ttk.Frame(self.root)
        self.shell.pack(fill="both", expand=True)

        # Sidebar + Main
        self.sidebar = ttk.Frame(self.shell, style="Sidebar.TFrame", width=200)
        self.sidebar.pack(side="left", fill="y")
        self.main = ttk.Frame(self.shell)
        self.main.pack(side="left", fill="both", expand=True)

        # Header
        self.header = tk.Canvas(self.main, height=74, highlightthickness=0, bd=0, bg=self.c["BG"])
        self.header.pack(fill="x")
        self._draw_header()

        # Toolbar
        self.toolbar = ttk.Frame(self.main, style="Card.TFrame")
        self.toolbar.pack(fill="x", padx=14, pady=(0, 10))

        # Body
        self.body = ttk.Frame(self.main)
        self.body.pack(fill="both", expand=True, padx=14, pady=(0, 12))

        # Status bar
        self.status = ttk.Label(self.root, text="", anchor="w", style="Status.TLabel")
        self.status.pack(fill="x")

    def _draw_header(self):
        self.header.delete("all")
        w = max(self.root.winfo_width(), 1200)
        # gradient
        for i in range(0, w, 4):
            t = i / max(w, 1)
            color = self._mix(self.c["ACCENT_SOFT"], self.c["ACCENT"], t)
            self.header.create_rectangle(i, 0, i + 4, 74, outline="", fill=color)
        # solid highlight (no alpha)
        self.header.create_rectangle(0, 0, w, 34, outline="", fill="#e6e6e6")
        self.header.create_text(20, 10, anchor="nw", text="🧠  NLP & Naive Bayes",
                                font=("Segoe UI Semibold", 18), fill="#0a0a0a")
        self.root.bind("<Configure>", lambda e: self._draw_header())

    def _mix(self, c1, c2, t):
        def h2i(h): return int(h, 16)
        r1,g1,b1 = h2i(c1[1:3]), h2i(c1[3:5]), h2i(c1[5:7])
        r2,g2,b2 = h2i(c2[1:3]), h2i(c2[3:5]), h2i(c2[5:7])
        r = int(r1*(1-t) + r2*t); g = int(g1*(1-t) + g2*t); b = int(b1*(1-t) + b2*t)
        return f"#{r:02x}{g:02x}{b:02x}"

    # ----- Sidebar -----
    def _build_sidebar(self):
        pad = 10
        ttk.Label(self.sidebar, text="Navigation", style="Section.TLabel").pack(anchor="w", padx=pad, pady=(12, 6))

        self.btn_nlp = ttk.Button(self.sidebar, text="🔍  NLP Tools", style="Sidebar.TButton",
                                  command=lambda: self._show_mode("NLP"))
        self.btn_nb = ttk.Button(self.sidebar, text="🧮  Naive Bayes", style="Sidebar.TButton",
                                 command=lambda: self._show_mode("NB"))
        self.btn_nlp.pack(fill="x", padx=pad, pady=4)
        self.btn_nb.pack(fill="x", padx=pad, pady=4)
        Tooltip(self.btn_nlp, "Switch to NLP tools (F1)")
        Tooltip(self.btn_nb, "Switch to Naive Bayes (F2)")

        ttk.Separator(self.sidebar, orient="horizontal").pack(fill="x", padx=pad, pady=12)

        ttk.Label(self.sidebar, text="Quick Tips", style="Section.TLabel").pack(anchor="w", padx=pad, pady=(0, 6))
        tips = tk.Label(self.sidebar,
                        text="• F1 / F2 to switch\n• Ctrl+Enter to Run\n• Click table headers to sort\n• Filter dataset in the preview",
                        bg=self.c["SIDEBAR"], fg=self.c["MUTED"], justify="left", wraplength=170)
        tips.pack(anchor="w", padx=pad)

    # ----- Toolbar -----
    def _build_toolbar(self):
        pad = 8
        self.toolbar.columnconfigure(0, weight=1)

        self.mode_label = ttk.Label(self.toolbar, text="Mode: NLP", style="Section.TLabel")
        self.mode_label.grid(row=0, column=0, sticky="w", padx=12, pady=8)

        self.run_btn = ttk.Button(self.toolbar, text="▶ Run (Ctrl+Enter)", style="Accent.TButton", command=self._run_action)
        self.clear_btn = ttk.Button(self.toolbar, text="🧹 Clear", command=self._clear_action)
        self.run_btn.grid(row=0, column=1, padx=pad, pady=8)
        self.clear_btn.grid(row=0, column=2, padx=pad, pady=8)
        Tooltip(self.run_btn, "Run the current operation")
        Tooltip(self.clear_btn, "Clear inputs/outputs")

    # ----- Cards (screens) -----
    def _build_cards(self):
        self.card_nlp = ttk.Frame(self.body, style="Card.TFrame")
        self._build_nlp_card(self.card_nlp)

        self.card_nb = ttk.Frame(self.body, style="Card.TFrame")
        self._build_nb_card(self.card_nb)

    # NLP UI
    def _build_nlp_card(self, parent):
        pad = 14
        ttk.Label(parent, text="Natural Language Processing", style="Section.TLabel").pack(anchor="w", padx=pad, pady=(pad, 8))

        ctr = ttk.Frame(parent)
        ctr.pack(fill="x", padx=pad)

        ttk.Label(ctr, text="Operation:").pack(side="left")
        self.operation_var = tk.StringVar()
        operations = ["Vocabulary", "Stemming", "Lemmatization", "Stop Words",
                      "Tokenization", "POS Tagging", "Bag of Words (BoW)", "TF-IDF"]
        self.operation_dropdown = ttk.Combobox(ctr, textvariable=self.operation_var,
                                               values=operations, state="readonly", width=30)
        self.operation_dropdown.current(0)
        self.operation_dropdown.pack(side="left", padx=(8, 12))

        ttk.Label(parent, text="Enter text", style="Muted.TLabel").pack(anchor="w", padx=pad, pady=(pad, 6))
        self.text_input = scrolledtext.ScrolledText(parent, height=8, wrap=tk.WORD,
                                                    bg=self.c["CARD"], fg=self.c["TEXT"],
                                                    insertbackground=self.c["TEXT"], bd=0,
                                                    highlightthickness=1, highlightbackground=self.c["BORDER"],
                                                    font=("Consolas", 11))
        self.text_input.pack(fill="x", padx=pad)

        ttk.Label(parent, text="Output", style="Muted.TLabel").pack(anchor="w", padx=pad, pady=(pad, 6))
        self.text_output = scrolledtext.ScrolledText(parent, height=14, wrap=tk.WORD,
                                                     bg=self.c["CARD_HI"], fg=self.c["TEXT"],
                                                     insertbackground=self.c["TEXT"], bd=0,
                                                     highlightthickness=1, highlightbackground=self.c["BORDER"],
                                                     font=("Consolas", 11))
        self.text_output.pack(fill="both", expand=True, padx=pad, pady=(0, pad))

    # Naive Bayes UI (with Preview + Filter + Sort + Zebra)
    def _build_nb_card(self, parent):
        pad = 14
        ttk.Label(parent, text="Naive Bayes Prediction", style="Section.TLabel").pack(anchor="w", padx=pad, pady=(pad, 8))

        top = ttk.Frame(parent)
        top.pack(fill="x", padx=pad)

        ttk.Label(top, text="Dataset:").pack(side="left")
        self.dataset_var = tk.StringVar()
        self.dataset_menu = ttk.Combobox(top, textvariable=self.dataset_var,
                                         values=list(datasets.keys()), state="readonly", width=30)
        self.dataset_menu.pack(side="left", padx=(8, 12))
        self.dataset_menu.bind("<<ComboboxSelected>>", self._load_features)

        ttk.Button(top, text="Predict", style="Accent.TButton", command=self.nb_predict).pack(side="left")

        # Preview + filter
        prev_wrap = ttk.LabelFrame(parent, text="Dataset Preview", style="Card.TFrame")
        prev_wrap.pack(fill="both", expand=False, padx=pad, pady=(pad, 8))

        filter_row = ttk.Frame(prev_wrap)
        filter_row.pack(fill="x", padx=10, pady=(8, 2))
        ttk.Label(filter_row, text="Filter rows:").pack(side="left")
        self.filter_var = tk.StringVar()
        self.filter_entry = ttk.Entry(filter_row, textvariable=self.filter_var, width=30)
        self.filter_entry.pack(side="left", padx=(8, 12))
        ttk.Button(filter_row, text="Apply", command=self._apply_preview_filter).pack(side="left", padx=(0, 6))
        ttk.Button(filter_row, text="Reset", command=lambda: self._reset_preview_filter(clear=True)).pack(side="left")
        Tooltip(self.filter_entry, "Type any text to filter preview rows")

        prev_container = ttk.Frame(prev_wrap)
        prev_container.pack(fill="both", expand=True, padx=10, pady=10)

        self.prev_tree = ttk.Treeview(prev_container, show="headings", height=9)
        self.prev_tree.pack(side="left", fill="both", expand=True)

        vsb = ttk.Scrollbar(prev_container, orient="vertical", command=self.prev_tree.yview)
        hsb = ttk.Scrollbar(prev_container, orient="horizontal", command=self.prev_tree.xview)
        self.prev_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        # Feature selectors
        self.features_frame = ttk.Frame(parent)
        self.features_frame.pack(fill="x", padx=pad, pady=(pad, 8))

        # Results
        self.result_card = ttk.Frame(parent, style="Card.TFrame")
        self.result_card.pack(fill="both", expand=True, padx=pad, pady=(6, pad))

        self.result_title = ttk.Label(self.result_card, text="Result", style="Section.TLabel")
        self.result_title.pack(anchor="w", padx=pad, pady=(pad, 6))

        self.result_label = ttk.Label(self.result_card, text="—", font=("Segoe UI", 12, "bold"))
        self.result_label.pack(anchor="w", padx=pad)

        self.progress = ttk.Progressbar(self.result_card, orient="horizontal", mode="determinate", length=260)
        self.progress.pack(anchor="w", padx=pad, pady=(6, 12))

        area = ttk.Frame(self.result_card)
        area.pack(fill="both", expand=True, padx=pad, pady=(0, pad))

        self.prob_tree = ttk.Treeview(area, columns=("class", "prob"), show="headings", height=6)
        self.prob_tree.heading("class", text="Class")
        self.prob_tree.heading("prob", text="Probability")
        self.prob_tree.column("class", width=160, anchor="w")
        self.prob_tree.column("prob", width=140, anchor="center")
        self.prob_tree.pack(side="left", fill="y", padx=(0, 8), pady=8)

        self.bars_canvas = tk.Canvas(area, height=200, width=420, bg=self.c["CARD"], highlightthickness=0)
        self.bars_canvas.pack(side="left", fill="both", expand=True, pady=8)

        # state for preview filtering
        self._preview_all_rows = []
        self._preview_cols = []

    # ----- Mode switching / toolbar state -----
    def _show_mode(self, mode):
        self.mode.set(mode)
        for w in self.body.winfo_children():
            w.pack_forget()
        if mode == "NLP":
            self.card_nlp.pack(fill="both", expand=True, padx=10, pady=10)
            self.mode_label.config(text="Mode: NLP")
        else:
            self.card_nb.pack(fill="both", expand=True, padx=10, pady=10)
            self.mode_label.config(text="Mode: Naive Bayes")
        self._set_status(f"Switched to {mode} mode.")

    # ----- Toolbar actions -----
    def _run_action(self):
        if self.mode.get() == "NLP":
            self.process_text()
        else:
            self.nb_predict()

    def _clear_action(self):
        if self.mode.get() == "NLP":
            self.text_input.delete("1.0", tk.END)
            self.text_output.delete("1.0", tk.END)
        else:
            # clear selections & result
            for var in getattr(self, "feature_vars", []):
                var.set("")
            self.result_label.config(text="—")
            self.progress["value"] = 0
            for i in self.prob_tree.get_children():
                self.prob_tree.delete(i)
            self.bars_canvas.delete("all")
        self._set_status("Cleared.")

    # ----- Feature controls + preview populate -----
    def _load_features(self, *_):
        for w in self.features_frame.winfo_children():
            w.destroy()

        dname = self.dataset_var.get()
        if not dname:
            return

        data_info = datasets[dname]
        data, cols = data_info["data"], data_info["cols"]
        features = cols[:-1]
        self.feature_vars = []

        # Populate dataset preview table (with zebra / sorting)
        self._populate_preview(cols, data)

        grid = ttk.Frame(self.features_frame)
        grid.pack(fill="x", padx=6, pady=6)
        for i, f in enumerate(features):
            r, c = divmod(i, 2)
            cell = ttk.Frame(grid)
            cell.grid(row=r, column=c, sticky="ew", padx=(0 if c == 0 else 12), pady=6)
            cell.columnconfigure(1, weight=1)

            ttk.Label(cell, text=f"{f}:").grid(row=0, column=0, sticky="w")
            var = tk.StringVar()
            values = sorted({row[i] for row in data})
            cb = ttk.Combobox(cell, textvariable=var, values=values, state="readonly", width=18)
            cb.grid(row=0, column=1, sticky="ew")
            self.feature_vars.append(var)

        self._set_status(f"Loaded dataset: {dname}")

    # Build preview with zebra stripes, sorting, filtering
    def _populate_preview(self, cols, data, max_rows=500):
        self._preview_cols = ["Index"] + cols
        self._preview_all_rows = [[idx] + row for idx, row in enumerate(data[:max_rows])]

        self.prev_tree.configure(columns=self._preview_cols)

        # clear existing
        for item in self.prev_tree.get_children():
            self.prev_tree.delete(item)
        for c in self._preview_cols:
            self.prev_tree.heading(c, text=c, command=lambda col=c: self._sort_preview(col, False))
            self.prev_tree.column(c, width=140, anchor="w")
        self.prev_tree.column("Index", width=60, anchor="center")

        # zebra style via tags
        self.prev_tree.tag_configure("odd", background=self.c["CARD"])
        self.prev_tree.tag_configure("even", background=self.c["CARD_HI"])

        for i, row in enumerate(self._preview_all_rows):
            tag = "even" if i % 2 == 0 else "odd"
            self.prev_tree.insert("", "end", values=row, tags=(tag,))

        self._reset_preview_filter(clear=False)

    def _sort_preview(self, col, reverse):
        # get column index
        col_idx = self._preview_cols.index(col)
        rows = [(self.prev_tree.set(k, col), k) for k in self.prev_tree.get_children("")]
        try:
            rows.sort(key=lambda t: float(t[0]) if t[0].replace('.', '', 1).isdigit() else t[0], reverse=reverse)
        except Exception:
            rows.sort(key=lambda t: t[0], reverse=reverse)
        for index, (val, k) in enumerate(rows):
            self.prev_tree.move(k, "", index)
            # re-apply zebra after sort
            self.prev_tree.item(k, tags=("even" if index % 2 == 0 else "odd",))
        # toggle next sort direction
        self.prev_tree.heading(col, text=col, command=lambda c=col: self._sort_preview(c, not reverse))

    def _apply_preview_filter(self):
        needle = self.filter_var.get().strip().lower()
        if not needle:
            self._reset_preview_filter(clear=False)
            self._set_status("Filter cleared.")
            return

        for item in self.prev_tree.get_children():
            self.prev_tree.delete(item)

        filtered = []
        for row in self._preview_all_rows:
            txt = " ".join(str(x).lower() for x in row)
            if needle in txt:
                filtered.append(row)

        for i, row in enumerate(filtered):
            tag = "even" if i % 2 == 0 else "odd"
            self.prev_tree.insert("", "end", values=row, tags=(tag,))
        self._set_status(f"Filter applied: {len(filtered)} row(s)")

    def _reset_preview_filter(self, clear=True):
        if clear:
            self.filter_var.set("")
        for item in self.prev_tree.get_children():
            self.prev_tree.delete(item)
        for i, row in enumerate(self._preview_all_rows):
            tag = "even" if i % 2 == 0 else "odd"
            self.prev_tree.insert("", "end", values=row, tags=(tag,))

    # ----- NLP processing -----
    def process_text(self):
        corpus = self.text_input.get("1.0", tk.END).strip()
        if not corpus:
            messagebox.showwarning("Warning", "Please enter some text first!")
            return
        self.text_output.delete("1.0", tk.END)
        op = self.operation_var.get()
        tokens = word_tokenize(corpus.lower())

        if op == "Vocabulary":
            vocab = sorted(set(tokens))
            self.text_output.insert(tk.END, ", ".join(vocab))
        elif op == "Stemming":
            for w in tokens:
                self.text_output.insert(tk.END, f"{w} → {stemmer.stem(w)}\n")
        elif op == "Lemmatization":
            for w in tokens:
                self.text_output.insert(tk.END, f"{w} → {lemmatizer.lemmatize(w)}\n")
        elif op == "Stop Words":
            filtered = [w for w in tokens if w not in stop_words]
            self.text_output.insert(tk.END, ", ".join(filtered))
        elif op == "Tokenization":
            sents = sent_tokenize(corpus)
            self.text_output.insert(tk.END, f"Word Tokens:\n{tokens}\n\nSentence Tokens:\n")
            for i, s in enumerate(sents, 1):
                self.text_output.insert(tk.END, f"{i}. {s}\n")
        elif op == "POS Tagging":
            for w, tag in pos_tag(word_tokenize(corpus)):
                self.text_output.insert(tk.END, f"{w} → {tag}\n")
        elif op == "Bag of Words (BoW)":
            vec = CountVectorizer()
            bow = vec.fit_transform([corpus])
            for w, c in zip(vec.get_feature_names_out(), bow.toarray()[0]):
                self.text_output.insert(tk.END, f"{w}: {c}\n")
        elif op == "TF-IDF":
            vec = TfidfVectorizer()
            tfidf = vec.fit_transform([corpus])
            for w, score in zip(vec.get_feature_names_out(), tfidf.toarray()[0]):
                self.text_output.insert(tk.END, f"{w}: {score:.4f}\n")
        self._set_status(f"Ran NLP op: {op}")

    # ----- Prediction + bars -----
    def nb_predict(self):
        dname = self.dataset_var.get()
        if not dname:
            messagebox.showinfo("Tip", "Please select a dataset first.")
            return

        data_info = datasets[dname]
        data, cols = data_info["data"], data_info["cols"]
        inputs = [v.get() for v in getattr(self, "feature_vars", [])]

        pred, probs = naive_bayes_predict(data, cols[:-1], cols[-1], inputs)
        conf = probs[pred] * 100.0

        self.result_label.config(text=f"✅ Predicted {cols[-1]}: {pred}  ({conf:.1f}%)")
        self.progress["value"] = conf

        for i in self.prob_tree.get_children():
            self.prob_tree.delete(i)
        for c, p in sorted(probs.items(), key=lambda x: -x[1]):
            self.prob_tree.insert("", "end", values=(c, f"{p*100:.1f}%"))

        self._draw_prob_bars(probs, highlight=pred)
        self._set_status(f"Predicted {pred} with {conf:.1f}% confidence.")

    def _draw_prob_bars(self, probs, highlight=None):
        self.bars_canvas.delete("all")
        padding = 18
        w = int(self.bars_canvas["width"])
        max_w = w - 2 * padding
        y = padding
        bar_h = 26
        gap = 14

        ordered = sorted(probs.items(), key=lambda x: -x[1])
        for cls, p in ordered:
            self.bars_canvas.create_text(padding, y - 16, anchor="nw",
                                         text=cls, fill=self.c["TEXT"], font=("Segoe UI", 10, "bold"))
            self._round_rect(self.bars_canvas, padding, y, padding + max_w, y + bar_h,
                             radius=10, fill=self.c["CARD_HI"], outline="")
            fill_w = int(max_w * p)
            fill_color = self.c["ACCENT"] if cls == highlight else self.c["ACCENT_DARK"]
            self._round_rect(self.bars_canvas, padding, y, padding + fill_w, y + bar_h,
                             radius=10, fill=fill_color, outline="")
            self.bars_canvas.create_text(padding + max_w + 8, y + bar_h / 2,
                                         text=f"{p*100:.1f}%", fill=self.c["TEXT"],
                                         font=("Segoe UI", 10), anchor="w")
            y += bar_h + gap

    @staticmethod
    def _round_rect(canvas, x1, y1, x2, y2, radius=8, **kwargs):
        points = [
            x1+radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1
        ]
        return canvas.create_polygon(points, smooth=True, **kwargs)

    def _set_status(self, text):
        self.status.config(text="  " + text)

# ---------------- RUN ----------------
if __name__ == "__main__":
    # Tooltip needs tk imported as Tooltip uses tk.Label; ensure here:
    import tkinter as tk  # (already imported above, harmless)
    root = tk.Tk()
    app = CombinedApp(root)
    root.mainloop()
