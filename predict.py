import pandas as pd
import numpy as np
from joblib import load
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings('ignore')

def handle_unseen_labels(series, encoder):
    """
    Преобразует метки с помощью обученного энкодера,
    обрабатывая новые значения, которых не было в обучении.
    """
    # Находим уникальные значения, которые энкодер "знает"
    known_labels = set(encoder.classes_)
    # Заменяем все "незнакомые" метки на 'unknown'
    # .astype(object) нужен на случай, если в series есть числа
    return series.astype(object).apply(lambda x: x if x in known_labels else 'unknown')

def make_predictions(test_filepath='clean_test_final.csv',
                     artifacts_filepath='final_model_with_artifacts.joblib',
                     output_filepath='submission.csv'):
    """
    Загружает модель и артефакты, обрабатывает тестовые данные и создает файл с предсказаниями.
    """
    print("--- 1. Загрузка модели и всех артефактов ---")
    try:
        artifacts = load(artifacts_filepath)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл с артефактами '{artifacts_filepath}' не найден!")
        print("Сначала запустите ноутбук с обучением, чтобы создать этот файл.")
        return

    model = artifacts['model']
    le_l2 = artifacts['le_l2']
    le_l4 = artifacts['le_l4']
    te_maps = artifacts['target_encoding_maps']
    feature_order = artifacts['feature_order']
    
    print("--- 2. Загрузка и обработка тестовых данных ---")
    df_test = pd.read_csv(test_filepath)
    
    # --- 2.1 Генерация признаков (все те же шаги, что и при обучении) ---
    print("Генерация признаков...")
    
    # Заполнение пропусков
    for col in ['category_l2', 'category_l4']:
        df_test[col] = df_test[col].fillna('unknown')
        
    # Текстовые признаки
    df_test['name_char_len'] = df_test['name'].str.len()
    df_test['type_char_len'] = df_test['type'].str.len()
    df_test['name_word_count'] = df_test['name'].str.split().str.len()
    df_test['type_word_count'] = df_test['type'].str.split().str.len()
    df_test['char_len_ratio'] = df_test['name_char_len'] / (df_test['type_char_len'] + 1)
    df_test['word_count_ratio'] = df_test['name_word_count'] / (df_test['type_word_count'] + 1)
    df_test['name_has_digit'] = df_test['name'].str.contains(r'\d').astype(int)
    
    # Эмбеддинги
    print("Генерация эмбеддингов (может занять время)...")
    model_emb = SentenceTransformer("ai-forever/sbert_large_nlu_ru") 
    name_embeddings_test = model_emb.encode(df_test['name'].astype(str).tolist(), show_progress_bar=True)
    type_embeddings_test = model_emb.encode(df_test['type'].astype(str).tolist(), show_progress_bar=True)
    
    df_test['cosine_similarity'] = np.einsum('ij,ij->i', name_embeddings_test, type_embeddings_test) / (
        np.linalg.norm(name_embeddings_test, axis=1) * np.linalg.norm(type_embeddings_test, axis=1)
    )

    # --- 2.2 Сборка финального набора признаков X_test ---
    print("Сборка датафрейма с признаками...")
    
    # Создаем DataFrame X_test, куда будем собирать все признаки
    X_test = pd.DataFrame(index=df_test.index)
    
    # Категориальные признаки (применяем .transform() от обученных энкодеров)
    # Используем нашу функцию для безопасного кодирования
    X_test['category_l2_enc'] = le_l2.transform(handle_unseen_labels(df_test['category_l2'], le_l2))
    X_test['category_l4_enc'] = le_l4.transform(handle_unseen_labels(df_test['category_l4'], le_l4))
    X_test['is_markup'] = df_test['is_markup'].astype('category').cat.codes

    # Target Encoding (применяем сохраненные карты)
    X_test['te_l2'] = X_test['category_l2_enc'].map(te_maps['map_l2']).fillna(te_maps['global_mean'])
    X_test['te_l4'] = X_test['category_l4_enc'].map(te_maps['map_l4']).fillna(te_maps['global_mean'])
    
    # Текстовые признаки
    text_features = [
        'name_char_len', 'type_char_len', 'name_word_count', 'type_word_count',
        'char_len_ratio', 'word_count_ratio', 'name_has_digit', 'cosine_similarity'
    ]
    
    # Объединяем все в один датафрейм
    X_test_final = pd.concat([
        df_test[text_features],
        pd.DataFrame(name_embeddings_test, columns=[f'name_emb_{i}' for i in range(name_embeddings_test.shape[1])]),
        pd.DataFrame(type_embeddings_test, columns=[f'type_emb_{i}' for i in range(type_embeddings_test.shape[1])]),
        X_test
    ], axis=1)
    
    # Приводим колонки к правильному порядку, который ожидает модель
    X_test_final = X_test_final[feature_order]

    # --- 3. Получение предсказаний ---
    print("Получение предсказаний...")
    predictions = model.predict_proba(X_test_final)[:, 1]

    # --- 4. Формирование файла для отправки ---
    submission = pd.DataFrame({'id': df_test.index, 'target': predictions})
    submission.to_csv(output_filepath, index=False)
    print(f"\n✅ Готово! Файл с предсказаниями '{output_filepath}' успешно создан.")

if __name__ == '__main__':
    make_predictions()
