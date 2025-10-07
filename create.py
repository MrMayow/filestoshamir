#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
from collections import defaultdict

def read_results_txt(path: Path):
    """
    Читает все непустые строки из results.txt.
    Возвращает список строк.
    """
    values = []
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                s = line.strip()
                if s != '':
                    values.append(s)
    except Exception:
        return []
    return values

def collect_data(top_dirs):
    """
    Собирает данные из непосредственных подкаталогов каждого топ-каталога.
    Возвращает:
      - set всех уникальных имён подкаталогов (для формирования строк),
      - dict: { top_name -> dict { subfolder_name -> list(values) } }
    """
    all_subnames = set()
    per_top = {}

    for top in top_dirs:
        top_path = Path(top).resolve()
        if not top_path.is_dir():
            continue
        top_name = top_path.name

        grid = defaultdict(list)
        for sub in top_path.iterdir():
            if not sub.is_dir():
                continue
            results_path = sub / 'results.txt'
            if results_path.is_file():
                vals = read_results_txt(results_path)
                # если файл пустой, оставим пустой список => не будет строк для этой пары
                if vals:
                    grid[sub.name].extend(vals)
                else:
                    # Хотим отразить факт наличия, но без значений — добавим пустую строку
                    grid[sub.name].append('')
                all_subnames.add(sub.name)
        per_top[top_name] = grid

    return all_subnames, per_top

def expand_rows(all_subnames, per_top, top_order):
    """
    Создаёт строки для финального DataFrame:
    - Первый столбец: 'Folder' — имя вложенной папки (подкаталога).
    - Далее по столбцу на каждый top (в порядке top_order).
    Правило множественных значений:
      - Если у подкаталога в каком-то top несколько значений, создаются отдельные строки
        только по этому столбцу; остальные столбцы заполняются пустыми, чтобы избежать
        искусственного декартова произведения.
    """
    rows = []
    for subname in sorted(all_subnames):
        # Сначала найдём максимум числа значений среди топов для этого subname
        counts = []
        per_col_values = []
        for top_name in top_order:
            vals = per_top.get(top_name, {}).get(subname, [])
            per_col_values.append(vals)
            counts.append(len(vals))
        max_rows = max(counts) if counts else 1

        # Если нигде нет значений вообще, всё равно сделаем одну строку с пустыми
        if max_rows == 0:
            max_rows = 1

        for i in range(max_rows):
            row = {'Folder': subname}
            for top_name, vals in zip(top_order, per_col_values):
                row[top_name] = vals[i] if i < len(vals) else ''
            rows.append(row)
    return rows

def main():
    # УКАЖИТЕ пути к 4 верхним папкам (порядок определит порядок столбцов)
    top_dirs = [
        'PATH/TO/TOP1',
        'PATH/TO/TOP2',
        'PATH/TO/TOP3',
        'PATH/TO/TOP4',
    ]

    all_subnames, per_top = collect_data(top_dirs)

    # Подписи столбцов — имена 4 верхних папок (в заданном порядке)
    top_order = []
    for p in top_dirs:
        pp = Path(p).resolve()
        if pp.is_dir():
            top_order.append(pp.name)

    if not top_order:
        print('Не найдено ни одного валидного верхнего каталога.')
        return

    # Строим строки
    rows = expand_rows(all_subnames, per_top, top_order)
    if not rows:
        # если вообще нет подкаталогов с results.txt, дадим пустую таблицу с нужными столбцами
        cols = ['Folder'] + top_order
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(rows, columns=['Folder'] + top_order)

    out_file = 'results_table.xlsx'
    with pd.ExcelWriter(out_file) as writer:
        df.to_excel(writer, index=False, sheet_name='Data')

    print(f'Готово: сохранено в {out_file}')

if __name__ == '__main__':
    main()
