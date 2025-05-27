import json
import pandas as pd
from pathlib import Path
import logging
import multiprocessing as mp
from collections import Counter
from config import Config, setup_logging
from core import DICOMHandler, ImageQualityAnalyzer, StaticPreprocessor, AdaptivePreprocessor, ImageEvaluator, VisualizationManager

def process_image(args):
    """Process a single image for parallel execution."""
    image_id, image, static_preprocessor, adaptive_preprocessor, evaluator, visualizer = args
    try:
        static_result = static_preprocessor.preprocess(image, image_id)
        adaptive_result, metrics, classification = adaptive_preprocessor.preprocess(image, image_id)
        evaluation = evaluator.evaluate(image, static_result, adaptive_result, image_id)
        visualizer.plot_comparison(image, static_result, adaptive_result, metrics, classification, image_id)
        return image_id, {
            'original': image,
            'static': static_result,
            'adaptive': adaptive_result,
            'metrics': metrics,
            'classification': classification,
            'evaluation': evaluation
        }
    except Exception as e:
        logging.error(f"Error processing {image_id}: {e}")
        return image_id, None

class AdaptiveIOPAPreprocessor:
    """Main pipeline class."""
    def __init__(self, config=Config):
        self.config = config
        self.data_dir = config.DATA_DIR
        self.output_dir = config.OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)
        
        setup_logging(config.LOG_FILE)
        self.dicom_handler = DICOMHandler(config.SUPPORTED_FORMATS)
        self.quality_analyzer = ImageQualityAnalyzer()
        self.static_preprocessor = StaticPreprocessor()
        self.adaptive_preprocessor = AdaptivePreprocessor(self.quality_analyzer, config.THRESHOLDS)
        self.evaluator = ImageEvaluator()
        self.visualizer = VisualizationManager(self.output_dir, config)
        
        self.images = {}
        self.metadata = {}
        self.quality_metrics = {}
        self.preprocessing_results = {}
        self.evaluation_results = []
        logging.info("Initialized pipeline")
    
    def load_dataset(self):
        """Load images from data directory."""
        logging.info(f"Loading dataset from {self.data_dir}")
        if not self.data_dir.exists():
            logging.error(f"Data directory {self.data_dir} does not exist")
            return False
        
        image_files = []
        for ext in self.config.SUPPORTED_FORMATS:
            image_files.extend(self.data_dir.glob(f'*{ext}'))
            image_files.extend(self.data_dir.glob(f'*{ext.upper()}'))
        
        formats = Counter(file_path.suffix.lower() for file_path in image_files)
        logging.info(f"Image formats found: {dict(formats)}")
        
        for file_path in sorted(image_files):
            image_id = file_path.stem
            image, metadata = self.dicom_handler.read_file(file_path)
            if image is not None:
                self.images[image_id] = image
                self.metadata[image_id] = metadata
                logging.info(f"Loaded {image_id}")
            else:
                logging.warning(f"Failed to load {image_id}")
        
        logging.info(f"Loaded {len(self.images)} images")
        return len(self.images) > 0
    
    def analyze_quality(self):
        """Analyze quality metrics."""
        logging.info("Analyzing quality")
        for image_id, image in self.images.items():
            self.quality_metrics[image_id] = self.quality_analyzer.analyze(image, image_id)
        
        with open(self.output_dir / 'quality_metrics.json', 'w') as f:
            json.dump({
                k: {
                    'brightness': float(v['brightness']),
                    'contrast_rms': float(v['contrast']['rms_contrast']),
                    'sharpness': float(v['sharpness']['laplacian_variance']),
                    'noise': float(v['noise']['wavelet_noise_estimate'])
                } for k, v in self.quality_metrics.items()
            }, f, indent=2)
        logging.info("Saved quality metrics")
    
    def run_preprocessing(self):
        """Run preprocessing in parallel."""
        logging.info("Running preprocessing")
        pool = mp.Pool(processes=mp.cpu_count())
        tasks = [(image_id, image, self.static_preprocessor, self.adaptive_preprocessor, self.evaluator, self.visualizer)
                 for image_id, image in self.images.items()]
        
        results = pool.map(process_image, tasks)
        pool.close()
        pool.join()
        
        for image_id, result in results:
            if result:
                self.preprocessing_results[image_id] = result
                self.evaluation_results.append(result['evaluation'])
                logging.info(f"Processed {image_id}")
            else:
                logging.warning(f"Failed to process {image_id}")
    
    def report_results(self):
        """Generate evaluation report."""
        logging.info("Generating report")
        self.visualizer.plot_results(self.evaluation_results)
        
        df = pd.DataFrame(self.evaluation_results)
        df.to_csv(self.output_dir / 'evaluation_results.csv', index=False)
        
        summary = {
            'count': len(df),
            'static_ssim': float(df['static_ssim'].mean()),
            'adaptive_ssim': float(df['adaptive_ssim'].mean()),
            'static_edge': float(df['static_edge_preservation'].mean()),
            'adaptive_edge': float(df['adaptive_edge_preservation'].mean())
        }
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print(f"Images processed: {len(self.images)}")
        print(f"Adaptive SSIM: {summary['adaptive_ssim']:.3f} (Static: {summary['static_ssim']:.3f})")
        print(f"Adaptive Edge: {summary['adaptive_edge']:.4f} (Static: {summary['static_edge']:.4f})")
        print("="*50)
    
    def run(self):
        """Execute full pipeline."""
        logging.info("Starting pipeline")
        print("Starting Adaptive IOPA Preprocessing Pipeline")
        
        if not self.load_dataset():
            print(f"No images found in {self.data_dir}")
            return False
        
        self.analyze_quality()
        self.run_preprocessing()
        self.report_results()
        
        print(f"Results saved to: {self.output_dir}")
        return True

def main():
    """Main function."""
    missing_libs = []
    try:
        import pylibjpeg
    except ImportError:
        missing_libs.append("pylibjpeg")
    try:
        import python_gdcm
    except ImportError:
        missing_libs.append("python-gdcm")
    
    if missing_libs:
        warning = f"Missing libraries: {', '.join(missing_libs)}. RVG files may fail, but other formats will process. Install using 'pip install {' '.join(missing_libs)}'."
        logging.warning(warning)
        print(f"\nWARNING: {warning}")
    else:
        logging.info("pylibjpeg and python-gdcm installed")
    
    pipeline = AdaptiveIOPAPreprocessor()
    if pipeline.run():
        print("\n🎉 Pipeline completed successfully")
    else:
        print("\n✗ Pipeline failed, see preprocessing.log")

if __name__ == "__main__":
    main()