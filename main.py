import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
import os
import re
import gc
import json
import gzip

stemmer = SnowballStemmer("russian")

# токенизация 
def tokenize(text):
    if not isinstance(text, str):
        return []
    tokens = re.findall(r'\b[а-яa-z0-9]+\b', text.lower())
    return [stemmer.stem(t) for t in tokens]


# Коэффициент насыщения частоты теримна
BM25_K1 = 1.2

# Штраф за длину документа
BM25_B = 0.4

# Размер первичного пула ретривала
TOP_K_CANDIDATES = 1500

# Файл кэша предобработанных документов
CACHE_FILE = "item_corpus.jsonl.gz"

# Потоковая токенизация (если кэш еще не собран) для ускорения алгоритма
if not os.path.exists(CACHE_FILE):
    print("Токенизация корпуса и запись в gzip...")

    # Берём только важные колонки
    items_dataframe = pd.read_parquet(
        "benchmark_items.parquet",
        columns=["item_title_raw", "item_infm_params_text", "item_description_raw"]
    )

    # Открытие сжатого потока на запись
    with gzip.open(CACHE_FILE, "wt", encoding="utf-8") as f_out:
        for idx in tqdm(range(len(items_dataframe)), desc="Tokenizing to disk"):

            # Извлекаем поля и взвешиваем текстовые зоны
            title = str(items_dataframe.at[idx, "item_title_raw"] or "")
            params = str(items_dataframe.at[idx, "item_infm_params_text"] or "")
            description = str(items_dataframe.at[idx, "item_description_raw"] or "")
            full_text = f"{title} {title} {title} {params} {params} {description}"
            
            # Токенизация, сериализация и потоковая запись
            tokens = tokenize(full_text)
            f_out.write(json.dumps(tokens, ensure_ascii=False) + "\n")
    
    del items_dataframe
    gc.collect()

# Загрузка метаданных
print("Загрузка метаданных...")
items_meta = pd.read_parquet(
    "benchmark_items.parquet",
    columns=["item_id", "item_category_id", "item_location_id"]
)
queries_dataframe = pd.read_parquet(
    "benchmark_queries.parquet",
    columns=["query_id", "search_query", "search_infm_params_text", "search_category", "search_location_id"]
)

# Получаем id объектов
item_ids = items_meta["item_id"].values
item_cats_ids = items_meta["item_category_id"].values
item_locs_ids = items_meta["item_location_id"].values

del items_meta
gc.collect()

print("Загрузка корпуса в память под BM25...")
item_corpus = []

# десериализации и быстрой загрузки предобработанного 
# корпуса в оперативную память перед передачей в индекс BM25
with gzip.open(CACHE_FILE, "rt", encoding="utf-8") as f_in:
    for line in tqdm(f_in, total=len(item_ids), desc="Loading cache"):
        item_corpus.append(json.loads(line))

# строит инвертированный поисковый индекс по алгоритму BM25Okapi
print(f"Построение BM25 индекса (k1={BM25_K1}, b={BM25_B})...")
bm25 = BM25Okapi(item_corpus, k1=BM25_K1, b=BM25_B)

del item_corpus
gc.collect()

# Инференс с исправленным строгим приоритетом
print("Генерация сабмита...")
answers = []

for _, row in tqdm(queries_dataframe.iterrows(), total=len(queries_dataframe), desc="ranking"):
    
    # сборка взвешенного текста запроса
    q_text = str(row["search_query"] or "") + " " + str(row["search_query"] or "") + " " + str(row["search_infm_params_text"] or "")
    
    # токенизация и стемминг
    q_tokens = tokenize(q_text)

    # извлекаем контекст пользователя
    q_cat = row["search_category"]
    q_loc = row["search_location_id"]

    # проверка от пустых запросов
    if not q_tokens:
        answers.append(" ".join(item_ids[:50]))
        continue

    # первичный отбор кандидатов (вычисляем релевантность запроса 
    # и приводим к рабочему виду индексы объявлений)
    scores = bm25.get_scores(q_tokens)
    top_idxs = np.argsort(scores)[-TOP_K_CANDIDATES:][::-1]

    # сборка признаков
    candidates_scores = scores[top_idxs]
    candidates_cats = item_cats_ids[top_idxs]
    candidates_locs = item_locs_ids[top_idxs]

    # формируем бинарные флаги соответствия
    if pd.isna(q_cat) or q_cat == 0:
        cat_match = np.ones(len(candidates_cats), dtype=int)
    else:
        cat_match = (candidates_cats == q_cat).astype(int)

    loc_match = (candidates_locs == q_loc).astype(int)
    
    # Строгая лексикографическая сортировка:
    # 1. Приоритет совпадению города (loc_match)
    # 2. Приоритет совпадению категории (cat_match)
    # 3. Текстовый скор BM25 (cand_scores)
    sort_order = np.lexsort((candidates_scores, cat_match, loc_match))[::-1]

    # формируем топ и сохраняем
    final_top_idx = top_idxs[sort_order[:50]]
    answers.append(" ".join(item_ids[final_top_idx]))

# Сохранение
submission = pd.DataFrame({
    "query_id": queries_dataframe["query_id"].astype(str),
    "answer": answers
})
submission.to_csv("answer_final.csv", index=False)
print("Файл answer_final.csv успешно сформирован!")