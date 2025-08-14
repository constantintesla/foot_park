import argparse
from pose_tracker import process_video
from movement_analysis import analyze_clinical_movement

def main():
    parser = argparse.ArgumentParser(description="Full clinical movement analysis pipeline")
    parser.add_argument('--input', required=True, help="Path to input video file")
    parser.add_argument('--output-csv', default="tracking_results.csv", help="Output CSV for pose data")
    parser.add_argument('--output-report', default="clinical_report.txt", help="Output clinical report")
    parser.add_argument('--show', action='store_true', help="Show real-time visualization")
    
    args = parser.parse_args()
    
    # Шаг 1: Трекинг позы
    process_video(args.input, args.output_csv, args.show)
    
    # Шаг 2: Клинический анализ
    analyze_clinical_movement(
        csv_path=args.output_csv,
        show_plots=args.show,
        report_path=args.output_report
    )

if __name__ == "__main__":
    main()