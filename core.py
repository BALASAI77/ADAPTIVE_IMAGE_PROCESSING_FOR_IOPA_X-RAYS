import numpy as np
import cv2
import pydicom
from PIL import Image
from skimage import exposure, restoration
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import logging
from config import Config, normalize_to_8bit, calculate_gradient_map

class DICOMHandler:
    """Handles DICOM and RVG file reading."""
    def __init__(self, supported_formats):
        self.supported_formats = supported_formats
        logging.info("Initialized DICOMHandler")
    
    def read_file(self, filepath):
        """Read DICOM, RVG, or image file."""
        try:
            if filepath.suffix.lower() in ['.dcm', '.rvg']:
                dicom_data = pydicom.dcmread(filepath, force=True)
                image = dicom_data.pixel_array
                if hasattr(dicom_data, 'PhotometricInterpretation') and dicom_data.PhotometricInterpretation == 'MONOCHROME1':
                    image = np.max(image) - image
                image = normalize_to_8bit(image)
                metadata = {'PhotometricInterpretation': getattr(dicom_data, 'PhotometricInterpretation', 'Unknown')}
                logging.info(f"Loaded DICOM/RVG: {filepath}")
            else:
                image = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    image = np.array(Image.open(filepath).convert('L'))
                image = normalize_to_8bit(image)
                metadata = {}
                logging.info(f"Loaded image: {filepath}")
            return image, metadata
        except Exception as e:
            logging.error(f"Error reading {filepath}: {e}")
            return None, None

class ImageQualityAnalyzer:
    """Analyzes image quality metrics."""
    def __init__(self):
        self.metrics_cache = {}
        logging.info("Initialized ImageQualityAnalyzer")
    
    def analyze(self, image, image_id=None):
        """Compute brightness, contrast, sharpness, and noise."""
        if image_id and image_id in self.metrics_cache:
            return self.metrics_cache[image_id]
        
        brightness = float(np.mean(image))
        contrast_rms = float(np.std(image))
        contrast_michelson = float((np.max(image) - np.min(image)) / (np.max(image) + np.min(image) + 1e-10))
        
        if image.ndim > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = normalize_to_8bit(image)
        
        laplacian_var = float(cv2.Laplacian(image, cv2.CV_64F).var())
        
        try:
            gaussian = cv2.GaussianBlur(image, (3,3), 0)
            noise_wavelet = float(np.median(np.abs(image.astype(float) - gaussian)) / 0.6745)
        except:
            noise_wavelet = float(np.std(image) * 0.1)
        
        metrics = {
            'brightness': brightness,
            'contrast': {'rms_contrast': contrast_rms, 'michelson_contrast': contrast_michelson},
            'sharpness': {'laplacian_variance': laplacian_var},
            'noise': {'wavelet_noise_estimate': noise_wavelet}
        }
        
        if image_id:
            self.metrics_cache[image_id] = metrics
        logging.info(f"Metrics for {image_id}: {metrics}")
        return metrics

class StaticPreprocessor:
    """Static preprocessing pipeline."""
    def __init__(self):
        self.name = "Static"
        logging.info("Initialized StaticPreprocessor")
    
    def preprocess(self, image, image_id=None):
        """Apply static preprocessing."""
        processed = cv2.equalizeHist(image)
        processed = cv2.GaussianBlur(processed, (3,3), 0)
        return np.clip(processed, 0, 255).astype(np.uint8)

class AdaptivePreprocessor:
    """Adaptive preprocessing pipeline."""
    def __init__(self, quality_analyzer, thresholds):
        self.quality_analyzer = quality_analyzer
        self.thresholds = thresholds
        self.name = "Adaptive"
        logging.info("Initialized AdaptivePreprocessor")
    
    def classify_quality(self, metrics):
        """Classify image quality."""
        def classify(value, thresh):
            if value < thresh['low']:
                return 'low'
            elif value > thresh['high']:
                return 'high'
            return 'medium'
        
        classification = {
            'brightness_level': classify(metrics['brightness'], self.thresholds['brightness']),
            'contrast_level': classify(metrics['contrast']['rms_contrast'], self.thresholds['contrast']),
            'sharpness_level': classify(metrics['sharpness']['laplacian_variance'], self.thresholds['sharpness']),
            'noise_level': classify(metrics['noise']['wavelet_noise_estimate'], self.thresholds['noise'])
        }
        return classification
    
    def preprocess(self, image, image_id=None):
        """Apply adaptive preprocessing with balanced adjustments."""
        metrics = self.quality_analyzer.analyze(image, image_id)
        classification = self.classify_quality(metrics)
        logging.info(f"Classification for {image_id}: {classification}")
        
        original = image.copy()
        processed = image.astype(np.float32)
        steps_applied = []
        
        # Default preprocessing: milder CLAHE and sharpening
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        processed = clahe.apply(processed.astype(np.uint8)).astype(np.float32)
        steps_applied.append("default_clahe_clip2.0")
        
        # Check SSIM after CLAHE
        processed_temp = np.clip(processed, 0, 255).astype(np.uint8)
        ssim_temp = float(ssim(original, processed_temp, data_range=255))
        if ssim_temp < 0.7:
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
            processed = clahe.apply(original.astype(np.uint8)).astype(np.float32)
            steps_applied.append("adjusted_clahe_clip1.5")
        
        gaussian = cv2.GaussianBlur(processed, (0,0), 1.0)
        processed = cv2.addWeighted(processed, 1.2, gaussian, -0.2, 0)
        steps_applied.append("default_sharpen_alpha1.2")
        
        # Additional contrast enhancement
        if classification['contrast_level'] == 'low':
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            processed = clahe.apply(processed.astype(np.uint8)).astype(np.float32)
            steps_applied.append("contrast_low_clahe_clip3.0")
        elif classification['contrast_level'] == 'high':
            processed = exposure.adjust_gamma(processed/255.0, gamma=0.8) * 255
            steps_applied.append("contrast_high_gamma0.8")
        
        # Noise reduction with edge preservation
        if classification['noise_level'] == 'high':
            processed = restoration.denoise_wavelet(processed/255.0, sigma=0.01, wavelet='db1') * 255
            steps_applied.append("noise_high_wavelet_sigma0.01")
        elif classification['noise_level'] == 'medium':
            processed = cv2.bilateralFilter(processed.astype(np.uint8), 5, 75, 75).astype(np.float32)
            steps_applied.append("noise_medium_bilateral_d5_sigma75")
        
        # Sharpening with moderation
        if classification['sharpness_level'] == 'low':
            gaussian = cv2.GaussianBlur(processed, (0,0), 1.5)
            processed = cv2.addWeighted(processed, 1.5, gaussian, -0.5, 0)
            steps_applied.append("sharpness_low_alpha1.5")
        elif classification['sharpness_level'] == 'medium':
            gaussian = cv2.GaussianBlur(processed, (0,0), 1.2)
            processed = cv2.addWeighted(processed, 1.3, gaussian, -0.3, 0)
            steps_applied.append("sharpness_medium_alpha1.3")
        
        processed = np.clip(processed, 0, 255).astype(np.uint8)
        
        # Final SSIM check to ensure quality
        final_ssim = float(ssim(original, processed, data_range=255))
        if final_ssim < 0.6:
            logging.warning(f"SSIM too low ({final_ssim:.4f}) for {image_id}, reverting to milder preprocessing")
            processed = original.astype(np.float32)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
            processed = clahe.apply(processed.astype(np.uint8)).astype(np.float32)
            gaussian = cv2.GaussianBlur(processed, (0,0), 1.0)
            processed = cv2.addWeighted(processed, 1.1, gaussian, -0.1, 0)
            processed = np.clip(processed, 0, 255).astype(np.uint8)
            steps_applied.append("reverted_mild_clahe1.5_sharpen1.1")
        
        # Log metrics before/after
        mse_check = np.mean((image - processed) ** 2)
        ssim_check = float(ssim(image, processed, data_range=255))
        edges_original = cv2.Canny(image, 50, 150)
        edges_processed = cv2.Canny(processed, 50, 150)
        edge_check = float(np.mean(edges_original == edges_processed))
        
        logging.info(f"Preprocessing for {image_id}: steps={steps_applied}, MSE={mse_check:.4f}, SSIM={ssim_check:.4f}, Edge={edge_check:.4f}")
        if mse_check == 0:
            logging.warning(f"Zero MSE for {image_id}: no preprocessing applied despite steps {steps_applied}")
        if ssim_check < 0.5:
            logging.warning(f"Low SSIM {ssim_check:.4f} for {image_id}: possible over-processing")
        
        return processed, metrics, classification

class ImageEvaluator:
    """Evaluates preprocessing results."""
    def __init__(self):
        self.results = []
        logging.info("Initialized ImageEvaluator")
    
    def evaluate(self, original, static_result, adaptive_result, image_id):
        """Compute evaluation metrics."""
        mse_static = np.mean((original - static_result) ** 2)
        mse_adaptive = np.mean((original - adaptive_result) ** 2)
        logging.info(f"MSE for {image_id}: static={mse_static:.4f}, adaptive={mse_adaptive:.4f}")
        
        psnr_static = float('inf') if mse_static == 0 else float(20 * np.log10(255.0 / np.sqrt(mse_static)))
        psnr_adaptive = float('inf') if mse_adaptive == 0 else float(20 * np.log10(255.0 / np.sqrt(mse_adaptive)))
        
        if mse_adaptive == 0:
            logging.warning(f"Adaptive MSE is 0 for {image_id}, indicating no preprocessing applied")
        
        ssim_static = float(ssim(original, static_result, data_range=255))
        ssim_adaptive = float(ssim(original, adaptive_result, data_range=255))
        
        edges_original = cv2.Canny(original, 50, 150)
        edges_static = cv2.Canny(static_result, 50, 150)
        edges_adaptive = cv2.Canny(adaptive_result, 50, 150)
        edge_static = float(np.mean(edges_original == edges_static))
        edge_adaptive = float(np.mean(edges_original == edges_adaptive))
        
        result = {
            'image_id': image_id,
            'static_psnr': psnr_static,
            'adaptive_psnr': psnr_adaptive,
            'static_ssim': ssim_static,
            'adaptive_ssim': ssim_adaptive,
            'static_edge_preservation': edge_static,
            'adaptive_edge_preservation': edge_adaptive
        }
        self.results.append(result)
        logging.info(f"Evaluated {image_id}: {result}")
        return result

class VisualizationManager:
    """Handles visualizations."""
    def __init__(self, output_dir, config):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.show_plots = config.VISUALIZATION['show_plots']
        self.dpi = config.VISUALIZATION['dpi']
        logging.info("Initialized VisualizationManager")
    
    def plot_metrics(self, all_metrics):
        """Plot quality metrics distribution."""
        df = pd.DataFrame([{
            'brightness': m['brightness'],
            'contrast': m['contrast']['rms_contrast'],
            'sharpness': m['sharpness']['laplacian_variance'],
            'noise': m['noise']['wavelet_noise_estimate']
        } for m in all_metrics.values()])
        
        fig = go.Figure()
        for col in df.columns:
            fig.add_trace(go.Histogram(x=df[col], name=col.capitalize(), nbinsx=20, opacity=0.5))
        fig.update_layout(title='Quality Metrics Distribution', barmode='overlay')
        fig.write_html(self.output_dir / 'metrics.html')
        if self.show_plots:
            fig.show()
    
    def plot_comparison(self, original, static_result, adaptive_result, metrics, classification, image_id):
        """Create comparison visualization with histograms of pixel intensities."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Comparison - {image_id}')
        
        # Top row: Original, Static, Adaptive images
        axes[0,0].imshow(original, cmap='gray')
        axes[0,0].set_title('Original')
        axes[0,0].axis('off')
        axes[0,1].imshow(static_result, cmap='gray')
        axes[0,1].set_title('Static')
        axes[0,1].axis('off')
        axes[0,2].imshow(adaptive_result, cmap='gray')
        axes[0,2].set_title('Adaptive')
        axes[0,2].axis('off')
        
        # Bottom row: Histograms of pixel intensities
        axes[1,0].hist(original.ravel(), bins=256, range=(0, 255), color='gray', alpha=0.7)
        axes[1,0].set_title('Original Histogram')
        axes[1,0].set_xlim(0, 255)
        axes[1,1].hist(static_result.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.7)
        axes[1,1].set_title('Static Histogram')
        axes[1,1].set_xlim(0, 255)
        axes[1,2].hist(adaptive_result.ravel(), bins=256, range=(0, 255), color='red', alpha=0.7)
        axes[1,2].set_title('Adaptive Histogram')
        axes[1,2].set_xlim(0, 255)
        
        metrics_text = f"Brightness: {metrics['brightness']:.1f} ({classification['brightness_level']})\n" \
                       f"Contrast: {metrics['contrast']['rms_contrast']:.1f} ({classification['contrast_level']})\n" \
                       f"Sharpness: {metrics['sharpness']['laplacian_variance']:.1f} ({classification['sharpness_level']})\n" \
                       f"Noise: {metrics['noise']['wavelet_noise_estimate']:.1f} ({classification['noise_level']})"
        fig.text(0.02, 0.02, metrics_text, fontsize=8)
        
        plt.savefig(self.output_dir / f'comparison_{image_id}.png', dpi=self.dpi, bbox_inches='tight')
        if self.show_plots:
            plt.show()
        plt.close()
    
    def plot_results(self, evaluation_results):
        """Plot evaluation results."""
        df = pd.DataFrame(evaluation_results)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['image_id'], y=df['static_ssim'], name='Static SSIM', marker_color='blue', opacity=0.4))
        fig.add_trace(go.Bar(x=df['image_id'], y=df['adaptive_ssim'], name='Adaptive SSIM', marker_color='red', opacity=0.4))
        fig.update_layout(title='SSIM Comparison', barmode='group')
        fig.write_html(self.output_dir / 'ssim.html')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['image_id'], y=['static_edge_preservation'], name='Static Edge', marker_color='blue', opacity=0.4))
        fig.add_trace(go.Bar(x=df['image_id'], y=['adaptive_edge_preservation'], name='Adaptive Edge', marker_color='red', opacity=0.4))
        fig.update_layout(title='Edge Preservation Comparison', barmode='group')
        fig.write_html(self.output_dir / 'edge.html')
        
        if self.show_plots:
            fig.show()