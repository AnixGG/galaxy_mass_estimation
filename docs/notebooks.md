# Навигатор по ноутбукам

[Проект](../README.md) · [Результаты](results.md) · [Воспроизведение](reproduction.md)

Все пути определяются от корня репозитория. Папка запуска Jupyter может быть корнем проекта или любой вложенной папкой.

## Основной маршрут

1. [Итоговый анализ](../notebooks/evaluation.ipynb) — графики и метрики из включённых в репозиторий массивов; без обучения и исходных карт.
2. [FNO-регрессоры](../src/galaxy_mass_estimation/models/fno.py) — модели `ModelFNO1` и `ModelFNO2`.
3. [Спектральные слои](../src/galaxy_mass_estimation/models/spectral.py) — `SpectralConv2d` и `FNO2d`.
4. [Датасет](../src/galaxy_mass_estimation/data.py) — изображение, красное смещение и целевая log-масса.

## Подготовка данных

| Ноутбук | Назначение |
| :--- | :--- |
| [01_prepare_cutouts](../notebooks/data/01_prepare_cutouts.ipynb) | Полный проход по каталогу и сохранение вырезок |
| [02_augment_clusters](../notebooks/data/02_augment_clusters.ipynb) | Повороты и отражения массивных скоплений |
| [03_split_dataset](../notebooks/data/03_split_dataset.ipynb) | Квантильное разбиение и проверка распределений |
| [prepare_cutouts_resume](../notebooks/data/prepare_cutouts_resume.ipynb) | Дополнительный вариант продолжения выгрузки с записи 5000; содержит сохранение меток |

Это исследовательские ноутбуки; номера задают порядок этапов, но не означают готовый автоматический пайплайн. В частности, полный шаг нормализации отсутствует, а перед повторным разбиением требуется определить `y`. Подробности — в [воспроизведении](reproduction.md).

## Обучение

| Модель | Raw | Normalized | Normalized + augmented |
| :--- | :--- | :--- | :--- |
| FNO base | [запуск](../notebooks/training/fno_base_raw.ipynb) | [запуск](../notebooks/training/fno_base_normalized.ipynb) | — |
| FNO wide | [запуск](../notebooks/training/fno_wide_raw.ipynb) | [запуск](../notebooks/training/fno_wide_normalized.ipynb) | [запуск](../notebooks/training/fno_wide_augmented.ipynb) |
| FNO less | [запуск](../notebooks/training/fno_less_raw.ipynb) | [запуск](../notebooks/training/fno_less_normalized.ipynb) | [запуск](../notebooks/training/fno_less_augmented.ipynb) |
| FNO large | — | — | [запуск](../notebooks/training/fno_large_augmented.ipynb) |
| CNN | — | — | [запуск](../notebooks/training/cnn_augmented.ipynb) |

Архитектуры, параметры, seed и вычисления исходного обучения сохранены. Обновлены импорты, пути и блоки сохранения: каждый запуск пишет один комплект файлов в свою папку `results/`.

## Ранние эксперименты

- [cnn_baseline](../notebooks/exploration/cnn_baseline.ipynb) — собственный CSV-загрузчик и нелогарифмированные метки.
- [cnn_augmented](../notebooks/exploration/cnn_augmented.ipynb) — ранний вариант с `images_augm/`.
- [cnn_residual](../notebooks/exploration/cnn_residual.ipynb) — остаточная CNN с отдельной веткой красного смещения.

Они сохранены как содержательные варианты исследования. Две копии с одинаковым исполняемым кодом и пустой ноутбук удалены. Эти ранние варианты не включены в основную сравнительную таблицу.

## Справочные материалы

- [ACT+Planck maps](../notebooks/reference/act_maps.ipynb) — чтение и визуализация карт, примеры вырезок.
- [Cluster catalogs](../notebooks/reference/cluster_catalogs.ipynb) — загрузка каталога и сохранение `clusters_selected.csv`.
- [Planck / HEALPix](../notebooks/reference/planck_maps.ipynb) — справочные примеры сферических карт.

Внешние файлы загружаются в `data/raw/`. Справочные ноутбуки содержат учебные примеры и требуют ручного выбора исходных данных.
