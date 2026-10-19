# EmotionAnalyzer

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-NLTK_·_TextBlob_·_spaCy-5C6BC0)

Настольное приложение на Tkinter для анализа тональности английских текстов. Вставьте текст или загрузите файл — программа покажет, насколько он позитивный, негативный или нейтральный.

> Работает только с текстами на английском языке.

## Возможности

- **Анализ через NLTK (VADER)** — доли positive / neutral / negative и итоговый compound-балл
- **Анализ через TextBlob** — полярность и субъективность
- **Статистика слов** — частотный анализ текста с графиком (matplotlib)
- Загрузка текста из файла, сохранение результатов, очистка полей

## Как читать результаты

**Полярность (Polarity)** — от −1 (максимально негативный текст) до 1 (максимально позитивный); 0 — нейтральный.

**Субъективность (Subjectivity)** — от 0 (полностью объективный) до 1 (полностью субъективный).

## Запуск

```bash
git clone https://github.com/Wert01/EmotionAnalyzer.git
cd EmotionAnalyzer
pip install nltk spacy textblob matplotlib
python main.py
```

При первом запуске автоматически скачиваются ресурсы NLTK и модель spaCy `en_core_web_sm`.
