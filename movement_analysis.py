import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from typing import Dict, Optional, Tuple

class ClinicalMovementAnalyzer:
    def __init__(self, csv_path: str, fps: int = 30):
        """
        Инициализация клинического анализатора движения
        
        :param csv_path: путь к CSV файлу с данными ключевых точек
        :param fps: частота кадров видео
        """
        self.csv_path = csv_path
        self.fps = fps
        self.df = None
        self.left_foot_y = None
        self.right_foot_y = None
        self.left_metrics = None
        self.right_metrics = None
        self.left_segments = None
        self.right_segments = None
        
        # Параметры для обнаружения пиков (можно настраивать)
        self.peak_params = {
            'prominence': 1,    # Уменьшил для большей чувствительности
            'width': 2,         # Минимальная ширина пика в кадрах
            'height': None,     # Без ограничения по высоте
            'distance': 3       # Минимальное расстояние между пиками в кадрах
        }
        
    def load_data(self) -> bool:
        """Загрузка и предварительная обработка данных"""
        try:
            self.df = pd.read_csv(self.csv_path)
            
            # Извлечение координат для стоп (ключевые точки 15 и 16 в YOLO Pose)
            if 'kp_15_y' in self.df.columns:
                self.left_foot_y = self._clean_keypoints(self.df['kp_15_y'].values)
            if 'kp_16_y' in self.df.columns:
                self.right_foot_y = self._clean_keypoints(self.df['kp_16_y'].values)
                
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _clean_keypoints(self, foot_y: np.ndarray) -> np.ndarray:
        """Очистка и интерполяция данных ключевых точек"""
        if foot_y is None or len(foot_y) == 0:
            return None
            
        foot_y = np.array(foot_y, dtype=np.float32)
        
        # Интерполяция пропущенных значений
        nan_indices = np.isnan(foot_y)
        if np.any(nan_indices):
            foot_y[nan_indices] = np.interp(
                np.where(nan_indices)[0],
                np.where(~nan_indices)[0],
                foot_y[~nan_indices]
            )
            
        # Сглаживание (фильтр Савицкого-Голея)
        if len(foot_y) > 11:
            foot_y = savgol_filter(foot_y, window_length=11, polyorder=2)
            
        return foot_y
    
    def _find_peaks(self, signal: np.ndarray) -> np.ndarray:
        """Обнаружение пиков с текущими параметрами"""
        if signal is None or len(signal) < 3:
            return np.array([])
            
        peaks, _ = find_peaks(signal, **self.peak_params)
        return peaks
    
    def _analyze_segments(self, foot_y: np.ndarray) -> Dict[str, float]:
        """Анализ сегментов выполнения теста (начало, середина, конец)"""
        if foot_y is None or len(foot_y) < 3:
            return None
            
        segments = {}
        segment_len = len(foot_y) // 3
        
        for i, segment in enumerate(['start', 'middle', 'end']):
            start_idx = i * segment_len
            end_idx = (i + 1) * segment_len if i < 2 else len(foot_y)
            
            segment_data = foot_y[start_idx:end_idx]
            vel = np.diff(segment_data)
            acc = np.diff(vel)
            
            peaks = self._find_peaks(segment_data)
            
            segments[f'{segment}_amplitude'] = np.max(segment_data) - np.min(segment_data)
            segments[f'{segment}_frequency'] = len(peaks) / (len(segment_data)/self.fps)
            segments[f'{segment}_smoothness'] = np.mean(np.abs(acc))
            
            # Расчет ритма
            zero_crossings = np.where(np.diff(np.sign(vel)))[0]
            if len(zero_crossings) > 2:
                intervals = np.diff(zero_crossings)
                segments[f'{segment}_rhythm'] = 1/np.std(intervals) if np.std(intervals) > 0 else 0
            else:
                segments[f'{segment}_rhythm'] = 0
            
        return segments
    
    def _calculate_clinical_score(self, metrics: Dict, segments: Dict) -> Tuple[int, Dict]:
        """Расчет клинической оценки по 5-балльной шкале"""
        if not metrics or not segments:
            return 0, {}
            
        # Базовые метрики (4=худший результат, 0=лучший)
        # Амплитуда: чем меньше амплитуда, тем хуже результат
        if metrics['amplitude'] < 10:
            amplitude_score = 4  # Очень плохая амплитуда
        elif metrics['amplitude'] < 20:
            amplitude_score = 3  # Плохая амплитуда
        elif metrics['amplitude'] < 40:
            amplitude_score = 2  # Умеренная амплитуда
        elif metrics['amplitude'] < 60:
            amplitude_score = 1  # Хорошая амплитуда
        else:
            amplitude_score = 0  # Отличная амплитуда
            
        # Частота: чем меньше частота, тем хуже результат
        if metrics['frequency'] < 0.5:
            frequency_score = 4  # Очень плохая частота
        elif metrics['frequency'] < 1.0:
            frequency_score = 3  # Плохая частота
        elif metrics['frequency'] < 2.0:
            frequency_score = 2  # Умеренная частота
        elif metrics['frequency'] < 3.0:
            frequency_score = 1  # Хорошая частота
        else:
            frequency_score = 0  # Отличная частота
        
        # Оценка ухудшения к концу теста (4=сильное ухудшение, 0=без ухудшения)
        decline_score = 0
        if segments['end_amplitude'] < 0.1 * segments['start_amplitude']:
            decline_score = 4  # Сильное ухудшение
        elif segments['end_amplitude'] < 0.3 * segments['start_amplitude']:
            decline_score = 3  # Значительное ухудшение
        elif segments['end_amplitude'] < 0.5 * segments['start_amplitude']:
            decline_score = 2  # Умеренное ухудшение
        elif segments['end_amplitude'] < 0.7 * segments['start_amplitude']:
            decline_score = 1  # Легкое ухудшение
        
        # Общая оценка (среднее с округлением)
        total_score = round((amplitude_score + frequency_score + decline_score) / 3)
        
        # Интерпретация (по требованиям: 4=наибольший уровень декомпенсации, 0=минимальный)
        score_description = {
            4: "Наибольший уровень неврологической декомпенсации: грубые нарушения движения",
            3: "Выраженные нарушения: значительное ухудшение к концу теста",
            2: "Умеренные нарушения: заметное снижение параметров",
            1: "Легкие нарушения: незначительное снижение амплитуды/ритма",
            0: "Минимальный уровень: сохраненная амплитуда и ритм на всем протяжении теста"
        }
        
        details = {
            'amplitude_score': amplitude_score,
            'frequency_score': frequency_score,
            'decline_score': decline_score,
            'description': score_description.get(total_score, "Не определено")
        }
        
        return total_score, details
    
    def analyze(self) -> bool:
        """Основной метод анализа"""
        if not self.load_data():
            return False
            
        # Анализ сегментов для каждой стопы
        self.left_segments = self._analyze_segments(self.left_foot_y)
        self.right_segments = self._analyze_segments(self.right_foot_y)
        
        # Расчет основных метрик
        self.left_metrics = self._calculate_metrics(self.left_foot_y) if self.left_foot_y is not None else None
        self.right_metrics = self._calculate_metrics(self.right_foot_y) if self.right_foot_y is not None else None
        
        return True
    
    def _calculate_metrics(self, foot_y: np.ndarray) -> Optional[Dict]:
        """Расчет основных метрик движения"""
        if foot_y is None or len(foot_y) < 3:
            return None
            
        # Находим пики (удары) с текущими параметрами
        peaks = self._find_peaks(foot_y)
        valleys = self._find_peaks(-foot_y)
        
        # Расчет временных параметров
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks) / self.fps
            rhythm_std = np.std(peak_intervals)
        else:
            rhythm_std = 0
            
        # Основные метрики
        metrics = {
            'amplitude': np.max(foot_y) - np.min(foot_y),
            'frequency': len(peaks) / (len(foot_y)/self.fps),
            'rhythm_std': rhythm_std,
            'peaks_count': len(peaks),
            'movement_duration': len(foot_y) / self.fps
        }
        
        return metrics
    
    def plot_results(self):
        """Визуализация результатов анализа"""
        if self.left_foot_y is None and self.right_foot_y is None:
            print("Нет данных для визуализации")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Временная ось в секундах
        time_axis = np.arange(len(self.left_foot_y if self.left_foot_y is not None else self.right_foot_y)) / self.fps
        
        if self.left_foot_y is not None:
            plt.plot(time_axis, self.left_foot_y, label='Левая нога', color='blue')
            peaks = self._find_peaks(self.left_foot_y)
            plt.scatter(peaks/self.fps, self.left_foot_y[peaks], color='red', marker='x', label='Пики (левая)')
            
        if self.right_foot_y is not None:
            plt.plot(time_axis, self.right_foot_y, label='Правая нога', color='green')
            peaks = self._find_peaks(self.right_foot_y)
            plt.scatter(peaks/self.fps, self.right_foot_y[peaks], color='orange', marker='x', label='Пики (правая)')
            
        plt.title('Движение стоп во времени (обнаружение пиков)')
        plt.xlabel('Время (сек)')
        plt.ylabel('Y-координата стопы')
        plt.legend()
        plt.grid(True)
        
        # Разделение на сегменты
        if len(time_axis) > 0:
            segment_len = len(time_axis) // 3
            for i in range(1, 3):
                plt.axvline(x=time_axis[i*segment_len], color='gray', linestyle='--', alpha=0.5)
                plt.text(time_axis[i*segment_len], plt.ylim()[1], 
                        ['Начало', 'Середина', 'Конец'][i], 
                        ha='center', va='bottom', backgroundcolor='white')
        
        plt.tight_layout()
        plt.show()
    
    def get_clinical_assessment(self) -> Dict:
        """Получить клиническую оценку по 5-балльной шкале"""
        left_score, left_details = (0, {}) if self.left_metrics is None else self._calculate_clinical_score(self.left_metrics, self.left_segments or {})
        right_score, right_details = (0, {}) if self.right_metrics is None else self._calculate_clinical_score(self.right_metrics, self.right_segments or {})
        
        return {
            'left_foot': {
                'score': left_score,
                'details': left_details
            },
            'right_foot': {
                'score': right_score,
                'details': right_details
            },
            'test_duration': len(self.df) / self.fps if self.df is not None else 0
        }
    
    def get_segment_analysis(self) -> Dict:
        """Получить детальный анализ сегментов"""
        analysis = {}
        
        for foot_name, segments in [('left_foot', self.left_segments), ('right_foot', self.right_segments)]:
            if segments:
                analysis[foot_name] = {
                    'start_amplitude': segments.get('start_amplitude', 0),
                    'middle_amplitude': segments.get('middle_amplitude', 0),
                    'end_amplitude': segments.get('end_amplitude', 0),
                    'start_frequency': segments.get('start_frequency', 0),
                    'middle_frequency': segments.get('middle_frequency', 0),
                    'end_frequency': segments.get('end_frequency', 0),
                    'amplitude_decline': (segments.get('start_amplitude', 0) - segments.get('end_amplitude', 0)) / max(segments.get('start_amplitude', 1), 1),
                    'frequency_decline': (segments.get('start_frequency', 0) - segments.get('end_frequency', 0)) / max(segments.get('start_frequency', 1), 1)
                }
            else:
                analysis[foot_name] = None
                
        return analysis

    def generate_report(self, output_path: str = None) -> str:
        """Генерация клинического отчета"""
        assessment = self.get_clinical_assessment()
        segment_analysis = self.get_segment_analysis()
        report = []
        
        report.append("КЛИНИЧЕСКИЙ ОТЧЕТ О ВЫПОЛНЕНИИ ФУНКЦИОНАЛЬНОЙ ПРОБЫ")
        report.append("="*60)
        report.append(f"Файл данных: {self.csv_path}")
        report.append(f"Длительность теста: {assessment['test_duration']:.1f} сек")
        report.append("\n")
        
        for foot in ['left_foot', 'right_foot']:
            report.append(f"{'ЛЕВАЯ НОГА' if foot == 'left_foot' else 'ПРАВАЯ НОГА'}:")
            report.append("-"*40)
            
            foot_data = assessment[foot]
            if foot_data['score'] is not None:
                report.append(f"Клиническая оценка: {foot_data['score']}/4")
                report.append(f"Описание: {foot_data['details']['description']}")
                report.append("\nДетализация:")
                report.append(f"  - Оценка амплитуды: {foot_data['details']['amplitude_score']}/4")
                report.append(f"  - Оценка частоты: {foot_data['details']['frequency_score']}/4")
                report.append(f"  - Оценка устойчивости: {foot_data['details']['decline_score']}/4")
                
                metrics = self.left_metrics if foot == 'left_foot' else self.right_metrics
                if metrics:
                    report.append("\nМетрики:")
                    report.append(f"  - Средняя амплитуда: {metrics['amplitude']:.1f} px")
                    report.append(f"  - Частота движений: {metrics['frequency']:.2f} Гц")
                    report.append(f"  - Количество ударов: {metrics['peaks_count']}")
                    
                    # Анализ сегментов
                    if segment_analysis[foot]:
                        seg = segment_analysis[foot]
                        report.append("\nАнализ сегментов:")
                        report.append(f"  - Амплитуда (начало): {seg['start_amplitude']:.1f} px")
                        report.append(f"  - Амплитуда (середина): {seg['middle_amplitude']:.1f} px")
                        report.append(f"  - Амплитуда (конец): {seg['end_amplitude']:.1f} px")
                        report.append(f"  - Снижение амплитуды: {seg['amplitude_decline']*100:.1f}%")
                        report.append(f"  - Частота (начало): {seg['start_frequency']:.2f} Гц")
                        report.append(f"  - Частота (середина): {seg['middle_frequency']:.2f} Гц")
                        report.append(f"  - Частота (конец): {seg['end_frequency']:.2f} Гц")
                        report.append(f"  - Снижение частоты: {seg['frequency_decline']*100:.1f}%")
            else:
                report.append("Данные отсутствуют или недостаточны для анализа")
            
            report.append("\n")
        
        full_report = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_report)
        
        return full_report

def analyze_clinical_movement(csv_path: str, show_plots: bool = True, report_path: str = None) -> bool:
    """
    Анализ клинического движения по данным из CSV файла
    
    :param csv_path: путь к CSV файлу с данными
    :param show_plots: показывать графики (True/False)
    :param report_path: путь для сохранения отчета (None - не сохранять)
    :return: True если анализ успешен, False если есть ошибки
    """
    analyzer = ClinicalMovementAnalyzer(csv_path)
    if analyzer.analyze():
        report = analyzer.generate_report(report_path)
        print(report)
        
        if show_plots:
            analyzer.plot_results()
            
        return True
    return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Клинический анализ движения стоп')
    parser.add_argument('--input', required=True, help='Путь к входному CSV файлу')
    parser.add_argument('--output', help='Путь для сохранения отчета')
    parser.add_argument('--no-plots', action='store_true', help='Отключить визуализацию')
    
    args = parser.parse_args()
    
    analyze_clinical_movement(
        csv_path=args.input,
        show_plots=not args.no_plots,
        report_path=args.output
    )