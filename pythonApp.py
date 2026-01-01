import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download required NLTK data
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

download_nltk_data()

# Initialize tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Color scheme
COLORS = {
    "primary": "#2C3E50",
    "secondary": "#34495E",
    "accent": "#3498DB",
    "success": "#27AE60",
    "warning": "#E67E22",
    "danger": "#E74C3C",
    "light": "#ECF0F1",
    "dark": "#2C3E50",
    "text": "#2C3E50",
    "bg": "#F8F9FA"
}

# Main Application Class
class NLPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NLP Text Processing Toolkit")
        self.root.geometry("900x700")
        self.root.configure(bg=COLORS["bg"])
        
        # Configure styles
        self.setup_styles()
        
        # Header Frame
        header_frame = tk.Frame(root, bg=COLORS["primary"], height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, 
                              text="🔍 NLP Text Processing Toolkit", 
                              font=("Segoe UI", 18, "bold"),
                              fg="white",
                              bg=COLORS["primary"])
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(header_frame,
                                 text="Advanced Natural Language Processing Made Simple",
                                 font=("Segoe UI", 10),
                                 fg="#BDC3C7",
                                 bg=COLORS["primary"])
        subtitle_label.pack(expand=True)
        
        # Main Container
        main_container = tk.Frame(root, bg=COLORS["bg"])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Input Section
        input_card = tk.Frame(main_container, bg="white", relief="flat", bd=1)
        input_card.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(input_card, text="📥 Input Text", 
                font=("Segoe UI", 12, "bold"),
                fg=COLORS["primary"],
                bg="white").pack(anchor=tk.W, padx=15, pady=(10, 5))
        
        tk.Label(input_card, text="Enter your text corpus below:", 
                font=("Segoe UI", 9),
                fg=COLORS["text"],
                bg="white").pack(anchor=tk.W, padx=15)
        
        self.text_input = scrolledtext.ScrolledText(input_card, 
                                                   height=6, 
                                                   wrap=tk.WORD,
                                                   font=("Consolas", 10),
                                                   bg="#F8F9FA",
                                                   relief="solid",
                                                   bd=1,
                                                   padx=10,
                                                   pady=10)
        self.text_input.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Control Panel
        control_card = tk.Frame(main_container, bg="white", relief="flat", bd=1)
        control_card.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(control_card, text="⚙️ Processing Options", 
                font=("Segoe UI", 12, "bold"),
                fg=COLORS["primary"],
                bg="white").pack(anchor=tk.W, padx=15, pady=(10, 5))
        
        control_inner = tk.Frame(control_card, bg="white")
        control_inner.pack(fill=tk.X, padx=15, pady=10)
        
        tk.Label(control_inner, text="Select Operation:", 
                font=("Segoe UI", 10),
                fg=COLORS["text"],
                bg="white").pack(side=tk.LEFT, padx=(0, 10))
        
        self.operation_var = tk.StringVar()
        operations = ["Vocabulary", "Stemming", "Lemmatization", "Stop Words", 
                     "Tokenization", "POS Tagging", "Bag of Words (BoW)", "TF-IDF"]
        
        self.operation_dropdown = ttk.Combobox(control_inner, 
                                              textvariable=self.operation_var,
                                              values=operations,
                                              state="readonly",
                                              width=25,
                                              height=15,
                                              font=("Segoe UI", 10))
        self.operation_dropdown.current(0)
        self.operation_dropdown.pack(side=tk.LEFT, padx=(0, 20))
        
        # Process Button with modern style
        self.process_button = tk.Button(control_inner, 
                                       text="🚀 Process Text", 
                                       command=self.process_text,
                                       bg=COLORS["accent"],
                                       fg="white",
                                       font=("Segoe UI", 10, "bold"),
                                       padx=25,
                                       pady=8,
                                       relief="flat",
                                       cursor="hand2")
        self.process_button.pack(side=tk.LEFT)
        
        # Add hover effect
        def on_enter(e):
            self.process_button['bg'] = "#2980B9"
        
        def on_leave(e):
            self.process_button['bg'] = COLORS["accent"]
        
        self.process_button.bind("<Enter>", on_enter)
        self.process_button.bind("<Leave>", on_leave)
        
        # Output Section
        output_card = tk.Frame(main_container, bg="white", relief="flat", bd=1)
        output_card.pack(fill=tk.BOTH, expand=True)
        
        output_header = tk.Frame(output_card, bg=COLORS["secondary"])
        output_header.pack(fill=tk.X)
        
        tk.Label(output_header, text="📊 Results", 
                font=("Segoe UI", 12, "bold"),
                fg="white",
                bg=COLORS["secondary"]).pack(anchor=tk.W, padx=15, pady=8)
        
        self.text_output = scrolledtext.ScrolledText(output_card, 
                                                    height=12, 
                                                    wrap=tk.WORD,
                                                    font=("Consolas", 10),
                                                    bg="#F8F9FA",
                                                    relief="solid",
                                                    bd=1,
                                                    padx=10,
                                                    pady=10)
        self.text_output.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Status Bar
        status_bar = tk.Frame(root, bg=COLORS["dark"], height=25)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)
        
        status_label = tk.Label(status_bar, 
                               text="Ready • NLP Toolkit v1.0", 
                               font=("Segoe UI", 8),
                               fg="#BDC3C7",
                               bg=COLORS["dark"])
        status_label.pack(side=tk.LEFT, padx=10)
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure combobox style
        style.configure('TCombobox',
                       fieldbackground=COLORS["light"],
                       background="white",
                       foreground=COLORS["text"],
                       borderwidth=1,
                       relief="solid")
        
        style.map('TCombobox',
                 fieldbackground=[('readonly', 'white')],
                 selectbackground=[('readonly', COLORS["accent"])],
                 selectforeground=[('readonly', 'white')])
    
    def process_text(self):
        corpus = self.text_input.get("1.0", tk.END).strip()
        
        if not corpus:
            messagebox.showwarning("Input Required", "Please enter some text to process!")
            return
        
        operation = self.operation_var.get()
        self.text_output.delete("1.0", tk.END)
        
        try:
            # Tokenize text
            tokens = word_tokenize(corpus.lower())
            
            if operation == "Vocabulary":
                vocab = sorted(set(tokens))
                result = f"📚 Total unique words: {len(vocab)}\n"
                result += "─" * 50 + "\n\n"
                result += ", ".join(vocab)
                self.text_output.insert(tk.END, result)
            
            elif operation == "Stemming":
                result = "🔤 Stemming Results\n"
                result += "─" * 50 + "\n"
                result += "Original → Stemmed\n" + "─" * 30 + "\n"
                for word in tokens:
                    stemmed = stemmer.stem(word)
                    result += f"{word:15} → {stemmed}\n"
                self.text_output.insert(tk.END, result)
            
            elif operation == "Lemmatization":
                result = "🌿 Lemmatization Results\n"
                result += "─" * 50 + "\n"
                result += "Original → Lemmatized\n" + "─" * 30 + "\n"
                for word in tokens:
                    lemmatized = lemmatizer.lemmatize(word)
                    result += f"{word:15} → {lemmatized}\n"
                self.text_output.insert(tk.END, result)
            
            elif operation == "Stop Words":
                filtered = [word for word in tokens if word not in stop_words]
                result = "🚫 Stop Words Removal\n"
                result += "─" * 50 + "\n"
                result += f"Original tokens: {len(tokens)}\n"
                result += f"After removing stop words: {len(filtered)}\n"
                result += f"Stop words removed: {len(tokens) - len(filtered)}\n\n"
                result += "Filtered text:\n" + "─" * 30 + "\n"
                result += ", ".join(filtered)
                self.text_output.insert(tk.END, result)
            
            elif operation == "Tokenization":
                result = "✂️ Tokenization Results\n"
                result += "─" * 50 + "\n"
                result += "WORD TOKENS:\n" + "─" * 30 + "\n"
                result += ", ".join(tokens)
                result += f"\n\nTotal word tokens: {len(tokens)}\n\n"
                
                sentences = sent_tokenize(corpus)
                result += "SENTENCE TOKENS:\n" + "─" * 30 + "\n"
                for i, sent in enumerate(sentences, 1):
                    result += f"{i:2}. {sent}\n"
                self.text_output.insert(tk.END, result)
            
            elif operation == "POS Tagging":
                pos_tags = pos_tag(word_tokenize(corpus))
                result = "🏷️ Part-of-Speech Tagging\n"
                result += "─" * 50 + "\n"
                result += "Word → POS Tag\n" + "─" * 30 + "\n"
                for word, tag in pos_tags:
                    result += f"{word:15} → {tag}\n"
                self.text_output.insert(tk.END, result)
            
            elif operation == "Bag of Words (BoW)":
                vectorizer = CountVectorizer()
                bow = vectorizer.fit_transform([corpus])
                feature_names = vectorizer.get_feature_names_out()
                
                result = "🎒 Bag of Words (Word Frequencies)\n"
                result += "─" * 50 + "\n"
                word_freq = list(zip(feature_names, bow.toarray()[0]))
                word_freq.sort(key=lambda x: x[1], reverse=True)
                
                for word, count in word_freq:
                    result += f"{word:15} : {count:2}\n"
                self.text_output.insert(tk.END, result)
            
            elif operation == "TF-IDF":
                vectorizer = TfidfVectorizer()
                tfidf = vectorizer.fit_transform([corpus])
                feature_names = vectorizer.get_feature_names_out()
                
                result = "📈 TF-IDF Scores\n"
                result += "─" * 50 + "\n"
                word_scores = list(zip(feature_names, tfidf.toarray()[0]))
                word_scores.sort(key=lambda x: x[1], reverse=True)
                
                for word, score in word_scores:
                    result += f"{word:15} : {score:.4f}\n"
                self.text_output.insert(tk.END, result)
        
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred while processing:\n{str(e)}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = NLPApp(root)
    root.mainloop()