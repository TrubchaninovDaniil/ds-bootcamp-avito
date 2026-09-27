import pandas as pd
import numpy as np

def run_sanity_checks(sub_path="answer_final.csv", queries_path="benchmark_queries.parquet", items_path="benchmark_items.parquet"):
    print("Загрузка данных для проверки...")
    
    try:
        sub = pd.read_csv(sub_path)
        queries = pd.read_parquet(queries_path, columns=["query_id", "search_category", "search_location_id"])
        items = pd.read_parquet(items_path, columns=["item_id", "item_category_id", "item_location_id"])
    except FileNotFoundError as e:
        print(f"Ошибка загрузки файлов: {e}")
        return

    print("\n --- БАЗОВЫЕ ПРОВЕРКИ ФОРМАТА ---")
    
    # Проверка количества запросов
    queries_count_match = len(sub) == len(queries)
    print(f"Всего запросов в сабмите: {len(sub)} | Совпадает с исходником: {queries_count_match}")

    # Проверка на NaN
    has_nans = sub['answer'].isna().any()
    print(f"(NaN): {has_nans}")

    # Проверка длины выдачи и дубликатов
    lengths = []
    is_unique = []
    top1_items = []

    for ans in sub['answer']:
        items_list = str(ans).split()
        lengths.append(len(items_list))
        is_unique.append(len(set(items_list)) == len(items_list))
        top1_items.append(items_list[0] if items_list else None)

    avg_len = np.mean(lengths)
    print(f"Среднее число кандидатов на запрос: {avg_len:.1f} (ожидается 50.0)")
    print(f"Все кандидаты уникальны внутри каждого запроса: {all(is_unique)}")

    print("\n--- ПРОВЕРКА БИЗНЕС-ЛОГИКИ ---")
    
    # Подготовка датафреймов для джоина
    sub['top_1_item'] = top1_items
    sub['query_id'] = sub['query_id'].astype(str)
    queries['query_id'] = queries['query_id'].astype(str)
    items['item_id'] = items['item_id'].astype(str)

    # Собираем единую таблицу для первого кандидата
    merged = sub[['query_id', 'top_1_item']].merge(queries, on='query_id', how='left')
    merged = merged.merge(items, left_on='top_1_item', right_on='item_id', how='left')

    # Метрика совпадения локации (Geo Match Rate)
    # Считаем как часто город топ-1 товара совпадает с городом запроса
    geo_matches = (merged['search_location_id'] == merged['item_location_id']).mean()
    
    # Метрика совпадения категории (Category Match Rate)
    # Исключаем запросы, где категория не была передана
    valid_cats = merged[merged['search_category'].notna() & (merged['search_category'] != 0)]
    cat_matches = (valid_cats['search_category'] == valid_cats['item_category_id']).mean()

    print(f"Geo Match: {geo_matches:.1%}")
    print(f"Category Match: {cat_matches:.1%}")
    
    print("\n--- ИТОГ ---")
    if queries_count_match and not has_nans and avg_len == 50.0 and all(is_unique):
        print("Проверки пройденны успешно. Файл полностью валиден.")
    else:
        print("Были обнаружены проблемы в формате!")

if __name__ == "__main__":
    run_sanity_checks()