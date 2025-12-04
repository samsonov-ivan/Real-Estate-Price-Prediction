"""Generate model comparison report from trained models."""

from src.trainer import ModelTrainer


def main() -> None:
    """Generate and save model comparison report"""
    trainer = ModelTrainer()
    
    if not trainer.results:
        print("No trained models found. Run main.py first to train models")
        return
    
    output_dir = 'reports'
    trainer.generate_report(output_dir=output_dir)
    print(f"Report generated in '{output_dir}/' directory")


if __name__ == "__main__":
    main()
