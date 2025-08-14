# Clinical Movement Analysis System

Система для анализа клинических движений на основе видео с использованием YOLOv8 Pose и количественной оценки параметров движения.

## 📌 Возможности

- **Трекинг позы**: Определение ключевых точек тела с помощью YOLOv8 Pose
- **Анализ движений**:
  - Расчет амплитуды, частоты и ритма движений
  - Оценка ухудшения параметров во времени
  - 5-балльная клиническая оценка
- **Визуализация**:
  - Графики траекторий движений
  - Автоматическое обнаружение пиков активности
  - Разделение на временные сегменты

## 🛠 Установка

1. Клонируйте репозиторий:
   git clone https://github.com/yourusername/clinical-movement-analysis.git
   cd clinical-movement-analysis
Установите зависимости:

pip install -r requirements.txt


🚀 Использование
1. Анализ видеофайла (полный пайплайн)
python main.py --input input_video.mp4 --output-csv pose_data.csv --output-report report.txt

2. Только трекинг позы
python pose_tracker.py --input input_video.mp4 --output tracking_results.csv

3. Только клинический анализ
python movement_analysis.py --input tracking_results.csv --output analysis_report.txt

📊 Пример вывода
График движения стоп:
https://docs/movement_plot_example.png

Клинический отчет:

КЛИНИЧЕСКИЙ ОТЧЕТ О ВЫПОЛНЕНИИ ФУНКЦИОНАЛЬНОЙ ПРОБЫ
============================================================
Файл данных: tracking_results.csv
Длительность теста: 12.3 сек

ЛЕВАЯ НОГА:
----------------------------------------
Клиническая оценка: 3/4
Описание: Легкие нарушения: незначительное снижение амплитуды/ритма

Детализация:
  - Оценка амплитуды: 3/4
  - Оценка частоты: 4/4
  - Оценка устойчивости: 2/4

📂 Структура проекта
text
clinical-movement-analysis/
├── main.py               # Главный скрипт для полного пайплайна
├── pose_tracker.py       # Трекинг позы с YOLOv8
├── movement_analysis.py  # Анализ клинических параметров
├── requirements.txt      # Зависимости
├── README.md             # Документация
