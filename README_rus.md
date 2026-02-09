# NSM-Visualizer

Локальный интерактивный визуализатор (Dash + Plotly, Python) для сценариев монетарной политики Zcash:
- статус-кво по фактическим данным цепи;
- сценарии ZIP 233 / ZIP 234 / ZIP 235;
- сценарии с burn/reissue и выбором сценариев будущей активности для ZIP 235 после tip.

## 1. Что показывает приложение

Основной график отображает:
- `Status quo inflation (%/year)` (зелёная линия): базовая траектория без сценарных изменений.
- `Scenario inflation (%/year)` (оранжевая линия): сценарий на основе выбранных опций.

Диагностика справа показывает:
- фактический `max_height_csv`;
- запрошенные и эффективные границы окна (`start_height`, `end_height`);
- какие ZIP-опции реально включены;
- burn/reissue баланс;
- контрольные точки (scenario audit).

## 2. Инварианты модели

- До текущего tip (`max_height_csv`) используются факты из `data/blocks_full_with_types.csv`.
- Исторические факты не переписываются (subsidy, fees, tx_count, issued supply, timestamps).
- Сценарные эффекты применяются только с `NSM activation height` и позже.
- Если входные блоки некорректны (`start_height > end_height`), тяжёлый расчёт не запускается.
- Для неполного CSV код работает через safe reindex/ffill и не должен падать на missing heights.

## 3. ZIP-логика в интерфейсе

`Enable Funds Removal From Circulation (ZIP 233)`:
- включает класс burn-механик (сам по себе ничего не сжигает).

`Remove 60% of Transaction Fees (ZIP 235)`:
- зависит от ZIP 233;
- сжигает `floor(fees_zat * ratio)` начиная с NSM-активации;
- использует фактические fee из CSV до tip;
- после tip автоматически применяет выбранный сценарий будущей активности (без отдельного чекбокса).

`Apply Issuance Smoothing (ZIP 234)`:
- включает сглаженную траекторию субсидии после NSM-активации;
- до активации равна статус-кво.

`Reissue Burned Amount in Future Subsidies (ZIP 234)`:
- доступно только при `ZIP 233 + ZIP 234` и если есть активный источник burn;
- сожжённый объём добавляется в пул reissue и возвращается в будущие субсидии.

`One-time Sprout burn` и `Apply voluntary burns`:
- зависят от ZIP 233;
- уменьшают circulating supply (level-effect).

## 4. Формула инфляции в графике

Инфляция считается по блоку как:

`inflation = subsidy_per_block * blocks_per_year / circulating_supply * 100`

где:
- до Blossom используется 150 сек/блок, после Blossom 75 сек/блок;
- `circulating = issued - burned`.

## 5. Структура проекта

- `viz/app.py` — UI, callbacks, валидация, построение графика, diagnostics.
- `viz/io.py` — загрузка YAML и CSV.
- `viz/model.py` — чистая модель (утилиты/вспомогательные расчёты).
- `config/config.yaml` — конфигурация цепи и параметров сценариев.
- `data/blocks_full_with_types.csv` — базовые фактические данные.
- `data/events_burn.csv` — добровольные burn-события.
- `data/update_fees_last_million.py` — обновление `fees_zat` через RPC.
- `backend/app.py` — вспомогательный FastAPI backend (snapshot/RPC health endpoints).

## 6. Конфигурация `config/config.yaml`

Минимальная рабочая схема:

```yaml
meta:
  name: "Zcash NSM / Burns Visualizer"
  version: 1

paths:
  blocks_csv: "data/blocks_full_with_types.csv"
  events_burn_csv: "data/events_burn.csv"

defaults:
  nsm_activation_height: 3566401
  horizon_end_height: 6400000
  display_start_date_utc: "2019-01-01T00:00:00Z"

issuance:
  zip234:
    numerator: 4126
    denominator: 10000000000

burns:
  fee_burn:
    ratio: 0.60

sprout:
  snapshot:
    pool_sprout_zat: 2548104009822
    measured_height: 3233397
    measured_time_utc: "2026-02-08T00:00:00Z"

future_activity:
  anchor_blocks: 200000
  default_profile: "linear"
  default_preset: "base"
  default_logistic_k: 0.80
  presets:
    conservative:
      linear_k: 0.05
      exp_k: 0.04
    base:
      linear_k: 0.10
      exp_k: 0.08
    aggressive:
      linear_k: 0.20
      exp_k: 0.14

ui:
  plot:
    max_points: 20000
    height_vh: 72
  y_axis:
    legacy_full_top_min: 52.0
    adaptive_padding_ratio: 0.15
    adaptive_padding_min: 0.02
  markers:
    line_width: 1
    line_dash: "dash"
    line_color: "#2f4f6f"
    opacity: 0.75
    label_font_size: 11
    label_xshift: 2
    label_yshift: -2

model:
  activation_rules:
    enforce_from_nsm: true
```

Пояснения по ключам:
- `paths.*` — пути к CSV.
- `defaults.nsm_activation_height` — базовая высота активации NSM, напрямую используемая как дефолт в UI.
- `defaults.horizon_end_height` — дефолтный horizon в UI.
- `defaults.display_start_date_utc` — дата якоря для дефолтного начала отображения.
- `issuance.zip234.*` — параметры сглаживания и reissue-распределения.
- `burns.fee_burn.ratio` — дефолт для fee burn в UI.
- `sprout.snapshot.*` — дефолтные значения для Sprout burn.
- `future_activity.anchor_blocks` — окно калибровки fee/tx около tip.
- `future_activity.presets.*` — пользовательские пресеты роста активности.
- `ui.plot.*` — высота графика и лимит downsampling.
- `ui.y_axis.*` — правила масштабирования по оси Y.
- `ui.markers.*` — стиль вертикальных маркеров и позиционирование подписей.
- `model.activation_rules.enforce_from_nsm` — принудительное начало сценарных эффектов от NSM-активации.

Mainnet-константы зафиксированы в коде (не меняются через YAML):
- `zatoshis_per_zec = 100_000_000`
- `max_money_zat = 2_100_000_000_000_000`
- `second_halving_height = 2_726_400`
- `halving_interval_blocks = 1_680_000` (post-Blossom)

## 7. Быстрый старт (локально)

Требования:
- Python 3.11+ (рекомендуется 3.12/3.13).

Установка:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Запуск Dash UI:

```bash
python -m viz.app
```

Открыть:
- `http://127.0.0.1:8050`

Опциональный backend (если нужен API health/snapshot):

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

## 8. Деплой

Вариант без контейнеров:
- процесс 1: Dash (`python -m viz.app`) на внутреннем порту;
- процесс 2: FastAPI (`uvicorn backend.app:app ...`) при необходимости;
- внешний reverse proxy (Nginx/Caddy) на 80/443.

Рекомендации:
- хранить `config/config.yaml` и `data/*.csv` как persistent volume;
- запускать процессы под `systemd` или supervisor;
- отключить debug для production.

## 9. Обновление `fees_zat` в CSV

Скрипт:
- `data/update_fees_last_million.py`

Пример (последний 1 млн блоков):

```bash
python data/update_fees_last_million.py --rpc-url http://127.0.0.1:8232 --last-blocks 1000000
```

Полезные флаги:
- `--start-height`, `--end-height` — ручной диапазон.
- `--rpc-user`, `--rpc-pass` — если RPC с auth.
- `--no-backup` — не создавать `.bak` (обычно лучше не использовать).

Важно:
- `changed=0` в начале диапазона нормально, если старые `fees_zat` уже совпадают.
- реальная скорость обработки in-range блоков смотрится по `in_range_rate_segment`.

## 10. Частые причины «нет дельты»

- `NSM activation height` выше tip и future fee модель фактически off.
- Включен только ZIP 233 без источника burn (fee/sprout/voluntary).
- Окно просмотра не захватывает участок после NSM-активации.
- В выбранном окне эффект слишком мал, нужна адаптивная шкала (start/end ближе к интересующему периоду).

## 11. Проверка после изменений

Минимум:

```bash
python -m py_compile viz/app.py viz/io.py viz/model.py data/update_fees_last_million.py
```

И вручную в UI:
- проверить статус-кво (все чекбоксы off);
- включить по очереди ZIP 234, ZIP 233+235, reissue;
- сверить diagnostics (`burned total`, `reissued total`, `max observed delta`).
