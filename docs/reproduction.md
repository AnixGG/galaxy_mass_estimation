# Просмотр и воспроизведение

[Проект](../README.md) · [Ноутбуки](notebooks.md) · [Результаты](results.md)

## Сохранённые результаты

Из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib jupyterlab
python -m jupyterlab
```

Откройте [notebooks/evaluation.ipynb](../notebooks/evaluation.ipynb). Ему достаточно массивов из `results/`. Пути автоматически определяются от корня проекта, поэтому не зависят от того, из какой папки открыт ноутбук. Установка PyTorch, GPU и исходные FITS-карты для этого сценария не нужны.

Пересчёт табличных метрик без Jupyter:

```bash
python scripts/summarize_results.py
```

Скрипт проверяет формы и конечность значений, вычисляет попарные MAE и RMSE и обновляет [results/metrics.json](../results/metrics.json).

## Пакет и окружение

Требуется Python 3.10 или новее. Установка из корня репозитория:

```bash
python -m pip install -r requirements.txt
```

Это editable-установка пакета `galaxy_mass_estimation` с зависимостями для ноутбуков и данных. Минимальная установка модели: `python -m pip install -e .`; для анализа: `python -m pip install -e '.[analysis]'`.

Версии исходного облачного окружения не зафиксированы. PyTorch должен поддерживать выбранное устройство; в исходном обучении выбирается CUDA при наличии, иначе CPU.

```python
from galaxy_mass_estimation.data import ACTClusterDataset
from galaxy_mass_estimation.models.fno import ModelFNO1, ModelFNO2
```

## Размещение данных

Каталог и частотные карты размещаются в `data/raw/`:

```text
clusters_selected.csv
act-planck_dr4dr6_coadd_AA_daynight_f090_map.fits
act-planck_dr4dr6_coadd_AA_daynight_f150_map.fits
act-planck_dr6.02_coadd_AA_daynight_f220_map.fits
```

В CSV используются `RADeg`, `decDeg`, `redshift`, `theta500Arcmin`, `M500c`, `M500cUncorr`. [Справочный ноутбук каталогов](../notebooks/reference/cluster_catalogs.ipynb) сохраняет CSV сразу в эту папку. Загрузки в справочных ноутбуках также направлены в `data/raw/`.

Подготовленные массивы находятся в `data/processed/`:

```text
data/processed/
├── images/                 # Исходные вырезки {idx}.npy
├── images_normalized/      # Нормализованные вырезки
├── images_augm/            # Аугментированные вырезки
├── images_norm_augm/       # Нормализованные аугментированные вырезки
├── redshifts.npy
├── logM500c.npy
├── logM500cUncorr.npy
├── redshifts_augm.npy
├── logM500c_augm.npy
└── logM500cUncorr_augm.npy
```

Обе папки с большими данными исключены из Git. Индексы включены в репозиторий: `data/splits/raw/` используется для raw и normalized, `data/splits/augmented/` — для расширенных запусков.

Загрузчик возвращает `(img, scalars, target)` в `float32`: изображение, красное смещение в массиве длины 1 и скалярный `logM500cUncorr`. FNO-регрессоры настроены на `148 × 148`: перед головой зафиксирован размер `21904`.

## Особенности исходного исследования

- **Вырезки.** `01_prepare_cutouts.ipynb` обрабатывает полный каталог. `prepare_cutouts_resume.ipynb` — отдельный вариант продолжения с `df[5000:]` и индексами `+5000`; запускать оба подряд не требуется. В полном варианте сохранение скалярных меток закомментировано; рабочий блок находится в варианте продолжения.
- **Нормализация.** Обучение ожидает готовые `images_normalized/` и `images_norm_augm/`. Полного шага их построения в исходной работе нет; метод нельзя восстановить по одним предсказаниям.
- **Аугментация.** Создаются повороты и отражения для верхних 15% скоплений по массе. В загрузчике сохранены граница `idx <= 9976` и имя `{idx + 1}.npy` для нормализованных дополнительных изображений. Соответствие файлов и меток нужно проверить на своём датасете.
- **Разбиение.** Перед пересозданием индексов в `03_split_dataset.ipynb` требуется определить `y` из выбранных меток. Готовые индексы можно использовать без пересоздания. Аугментированное разбиение не группирует копии исходного скопления.
- **Обучение.** Архитектуры и численные параметры сохранены. Выходные папки создаются автоматически. Веса моделей не включены; сохранены только предсказания, метки и loss. Особенности CNN и использования test описаны в [результатах](results.md).

Реструктуризация устраняет зависимость от старых абсолютных путей, но не меняет методику исходного исследования.

## Иллюстрации README

Научные графики строятся Matplotlib из массивов `results/`; HTML задаёт типографику, подписи и компоновку. Все подписи иллюстраций — на английском. PNG экспортируются из локального HTML через Playwright/Chromium, без внешних шрифтов и сетевых ресурсов:

```bash
python -m pip install -e '.[analysis,dev]'
python -m playwright install chromium
python scripts/render_readme.py
```

Только построение HTML и графиков: `python scripts/render_readme.py --html-only`. Исходники и изображения находятся в [docs/assets/](assets/).

## Проверка кода

После установки пакета:

```bash
python -m unittest discover -s tests -v
python -m black --check src scripts tests
```

Black входит в группу зависимостей `dev`. Проверки охватывают FNO-прогон, спектральные градиенты, варианты загрузки изображений и сохранённые численные артефакты.
