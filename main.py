import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from spacy.cli import download
from collections import Counter
import matplotlib.pyplot as plt
from textblob import TextBlob

# Загружаем необходимые ресурсы NLTK
nltk.download('vader_lexicon')
nltk.download('punkt')

# Инициализация анализатора настроений NLTK
sia = SentimentIntensityAnalyzer()

# Инициализация SpaCy с проверкой и загрузкой модели
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Downloading now...")
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# Функция для анализа текста с использованием NLTK
def analyze_text_nltk():
    text = text_area.get("1.0", tk.END)
    if not text.strip():
        messagebox.showwarning("Warning", "Please enter some text to analyze.")
        return

    scores = sia.polarity_scores(text)

    result = f"NLTK Sentiment Analysis:\n\n" \
             f"Positive: {scores['pos']}\n" \
             f"Neutral: {scores['neu']}\n" \
             f"Negative: {scores['neg']}\n" \
             f"Compound: {scores['compound']}"

    result_area.config(state=tk.NORMAL)
    result_area.delete("1.0", tk.END)
    result_area.insert(tk.END, result)
    result_area.config(state=tk.DISABLED)


# Функция для анализа текста с использованием SpaCy
def analyze_text_spacy():
    text = text_area.get("1.0", tk.END)
    if not text.strip():
        messagebox.showwarning("Warning", "Please enter some text to analyze.")
        return

    # Используем TextBlob для анализа настроений
    blob = TextBlob(text)
    sentiment = blob.sentiment

    result = f"SpaCy Sentiment Analysis (using TextBlob):\n\n" \
             f"Polarity: {sentiment.polarity}\n" \
             f"Subjectivity: {sentiment.subjectivity}"

    result_area.config(state=tk.NORMAL)
    result_area.insert(tk.END, '\n\n' + result)
    result_area.config(state=tk.DISABLED)


# Функция для загрузки текста из файла
def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            text_area.delete("1.0", tk.END)
            text_area.insert(tk.END, content)


# Функция для сохранения результатов в файл
def save_results():
    result_text = result_area.get("1.0", tk.END)
    if not result_text.strip():
        messagebox.showwarning("Warning", "There are no results to save.")
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(result_text)
        messagebox.showinfo("Success", "Results saved successfully.")


# Функция для очистки текста и результатов
def clear_text_and_results():
    text_area.delete("1.0", tk.END)
    result_area.config(state=tk.NORMAL)
    result_area.delete("1.0", tk.END)
    result_area.config(state=tk.DISABLED)


# Функция для отображения статистики по словам
def show_word_stats():
    text = text_area.get("1.0", tk.END)
    if not text.strip():
        messagebox.showwarning("Warning", "Please enter some text to analyze.")
        return

    # Токенизация текста и подсчет частотности слов
    words = nltk.word_tokenize(text)
    word_freq = Counter(words)
    common_words = word_freq.most_common(10)

    # Разделение слов и их частот для построения графика
    words, counts = zip(*common_words)

    # Построение графика
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts, color='blue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 Words')
    plt.show()


# Основное окно
root = tk.Tk()
root.title("Employee Psychotype Determination System")
root.geometry("800x700")
# Создание меню
menu = tk.Menu(root)
root.config(menu=menu)
file_menu = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Load Text File", command=load_file)
file_menu.add_command(label="Save Results", command=save_results)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

analysis_menu = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Analysis", menu=analysis_menu)
analysis_menu.add_command(label="Analyze with NLTK", command=analyze_text_nltk)
analysis_menu.add_command(label="Analyze with SpaCy", command=analyze_text_spacy)
analysis_menu.add_command(label="Show Word Statistics", command=show_word_stats)
analysis_menu.add_separator()
analysis_menu.add_command(label="Clear Text and Results", command=clear_text_and_results)

# Метка
label = tk.Label(root, text="Enter the text for psychotype analysis:")
label.pack(pady=10)

# Текстовое поле для ввода текста
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=15)
text_area.pack(pady=10)

# Текстовое поле для вывода результатов
result_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10, state=tk.DISABLED)
result_area.pack(pady=10)

# Создание кнопок
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

btn_analyze_nltk = tk.Button(button_frame, text="Analyze with NLTK", command=analyze_text_nltk)
btn_analyze_nltk.grid(row=0, column=0, padx=5)

btn_analyze_spacy = tk.Button(button_frame, text="Analyze with SpaCy", command=analyze_text_spacy)
btn_analyze_spacy.grid(row=0, column=1, padx=5)

btn_show_stats = tk.Button(button_frame, text="Show Word Statistics", command=show_word_stats)
btn_show_stats.grid(row=0, column=2, padx=5)

btn_clear = tk.Button(button_frame, text="Clear Text and Results", command=clear_text_and_results)
btn_clear.grid(row=0, column=3, padx=5)

# Запуск основного цикла обработки событий
root.mainloop()