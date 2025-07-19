# ПОЛНАЯ ПРЕДОБРАБОТКА ДАННЫХ

import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
from tqdm import tqdm

# НАСТРОЙКИ
LANGUAGE = "russian"

# ЗАГРУЗКА РЕСУРСОВ 
print("Загрузка NLP моделей...")
nltk.download("stopwords", quiet=True)
nlp = spacy.load("ru_core_news_sm")
stop_words = stopwords.words(LANGUAGE)

# ФУНКЦИИ ПРЕДОБРАБОТКИ
def full_preprocess_text(text: str) -> str:
    """Полный цикл очистки текста."""
    if not isinstance(text, str):
        return ""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    words = [word for word in text.split() if word not in stop_words and not word.isdigit()]
    doc = nlp(" ".join(words))
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)

# ОСНОВНОЙ СКРИПТ
tqdm.pandas()

print("1. Загрузка исходных данных...")
df = pd.read_parquet('data/hackaton_train_types_recom.parquet')
print(f"Исходный размер данных: {df.shape}")

# Удаление дубликатов 
print(f"Найдено дубликатов: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"Размер после удаления дубликатов: {df.shape}")

# Удаление колонки main_photo, которую мы не используем 
df.drop(columns=['main_photo'], inplace=True)
print("Колонка 'main_photo' удалена.")

# Заполнение NaN в категориальных колонках
print("Заполнение пропусков (NaN)...")
categorical_cols_to_fill = ['category_l2', 'category_l4']
for col in categorical_cols_to_fill:
    df[col] = df[col].fillna('unknown')

# Предобработка текстовых колонок
text_cols_to_process = ["name", "type", "category_l2", "category_l4"]
for column in text_cols_to_process:
    print(f"Обработка колонки '{column}'...")
    processed_column_name = f"{column}_processed"
    df[processed_column_name] = df[column].progress_apply(full_preprocess_text)
    
    # Заменяем пустые строки, которые могли образоваться после очистки
    df[processed_column_name] = df[processed_column_name].apply(lambda x: x if x.strip() else 'unknown')

# 6. Сохранение чистого файла
output_filename = 'clean_train_final.csv'
df.to_csv(output_filename, index=False, encoding='utf-8')

print(f"\nГотово! Финальный чистый файл сохранен как '{output_filename}'")
print("\nИтоговая информация о датафрейме:")
df.info()