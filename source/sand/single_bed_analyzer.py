import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy import stats, ndimage
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.signal import find_peaks
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage import measure
from skimage.filters import gabor_kernel
import mahotas as mh

class SandPatternAnalyzer:
    def __init__(self, image_paths):
        """
        Initialize with a list of image paths instead of a folder
        
        Args:
            image_paths (list): Lista de rutas a las imágenes a analizar
        """
        self.image_paths = image_paths
        self.images = []
        self.average_pattern = None
        self.variance_map = None
        self.height, self.width = None, None
        
    def load_images(self, enhance_contrast=True):
        """Load and preprocess the specified images."""
        image_arrays = []
        
        for image_path in self.image_paths:
            # Load image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Error loading image: {image_path}")
                continue
                
            if enhance_contrast:
                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img = clahe.apply(img)
            
            if self.height is None:
                self.height, self.width = img.shape
            elif img.shape != (self.height, self.width):
                img = cv2.resize(img, (self.width, self.height))
            
            # Normalize image
            img = img / 255.0
            image_arrays.append(img)
            
        self.images = np.array(image_arrays)
        print(f"Loaded {len(self.images)} images successfully")
        
        if len(self.images) == 0:
            raise ValueError("No images were loaded successfully")

    def generate_heatmap(self):
        """Generate average pattern heatmap and variance map."""
        if len(self.images) == 0:
            raise ValueError("No images loaded. Please load images first.")
            
        self.average_pattern = np.mean(self.images, axis=0)
        self.variance_map = np.var(self.images, axis=0)
        
        return self.average_pattern, self.variance_map

    def analyze_texture(self, image):
        """Perform detailed texture analysis using GLCM."""
        img_uint8 = (image * 255).astype(np.uint8)
        
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(img_uint8, distances, angles, 256, symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        energy = graycoprops(glcm, 'energy').mean()
        
        return {
            'contrast': contrast,
            'dissimilarity': dissimilarity,
            'homogeneity': homogeneity,
            'correlation': correlation,
            'energy': energy
        }

    def analyze_local_patterns(self, image, window_size=50):
        """Analyze patterns in local windows across the image."""
        local_features = np.zeros_like(image)
        padded_image = np.pad(image, window_size//2)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = padded_image[i:i+window_size, j:j+window_size]
                local_features[i,j] = np.std(window)
        
        return local_features

    def detect_detailed_edges(self, image):
        """Enhanced edge detection using multiple methods."""
        img = (image * 255).astype(np.uint8)
        
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        
        canny = cv2.Canny(img, 50, 150)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        
        return sobel, canny, laplacian

    def find_pattern_periodicity(self):
        """Analyze pattern periodicity in both directions."""
        if len(self.images) == 0:
            raise ValueError("No images loaded. Please load images first.")
            
        horizontal_profile = np.mean(self.average_pattern, axis=0)
        h_peaks, _ = find_peaks(horizontal_profile, distance=20)
        
        vertical_profile = np.mean(self.average_pattern, axis=1)
        v_peaks, _ = find_peaks(vertical_profile, distance=20)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        ax1.plot(horizontal_profile)
        ax1.plot(h_peaks, horizontal_profile[h_peaks], "x")
        ax1.set_title('Horizontal Pattern Periodicity')
        
        ax2.plot(vertical_profile)
        ax2.plot(v_peaks, vertical_profile[v_peaks], "x")
        ax2.set_title('Vertical Pattern Periodicity')
        
        plt.tight_layout()
        return fig, h_peaks, v_peaks

    def analyze_patterns_detailed(self):
        """Perform comprehensive detailed pattern analysis."""
        if len(self.images) == 0:
            raise ValueError("No images loaded. Please load images first.")
            
        fig = plt.figure(figsize=(20, 20))
        
        # 1. Enhanced Average Pattern
        avg_pattern, var_map = self.generate_heatmap()
        plt.subplot(331)
        sns.heatmap(avg_pattern, cmap='viridis')
        plt.title('Enhanced Average Pattern')
        
        # 2. Local Pattern Analysis
        local_patterns = self.analyze_local_patterns(avg_pattern)
        plt.subplot(332)
        sns.heatmap(local_patterns, cmap='hot')
        plt.title('Local Pattern Variation')
        
        # 3. Multiple Edge Detection Methods
        sobel, canny, laplacian = self.detect_detailed_edges(avg_pattern)
        plt.subplot(333)
        plt.imshow(sobel, cmap='magma')
        plt.title('Sobel Edge Detection')
        
        plt.subplot(334)
        plt.imshow(canny, cmap='magma')
        plt.title('Canny Edge Detection')
        
        plt.subplot(335)
        plt.imshow(laplacian, cmap='magma')
        plt.title('Laplacian Edge Detection')
        
        # 4. Pattern Direction Analysis
        sobelx = cv2.Sobel(avg_pattern, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(avg_pattern, cv2.CV_64F, 0, 1, ksize=3)
        gradient_direction = np.arctan2(sobely, sobelx)
        plt.subplot(336)
        plt.imshow(gradient_direction, cmap='hsv')
        plt.title('Pattern Direction Map')
        
        # 5. Texture Analysis
        texture_features = self.analyze_texture(avg_pattern)
        plt.subplot(337)
        plt.bar(texture_features.keys(), texture_features.values())
        plt.xticks(rotation=45)
        plt.title('Texture Features')
        
        # 6. Fine Detail Enhancement
        detail_enhanced = cv2.detailEnhance(
            cv2.cvtColor((avg_pattern * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
            sigma_s=10,
            sigma_r=0.15
        )
        plt.subplot(338)
        plt.imshow(cv2.cvtColor(detail_enhanced, cv2.COLOR_BGR2RGB))
        plt.title('Enhanced Fine Details')
        
        # 7. Pattern Scale Analysis
        scales = np.array([ndimage.gaussian_filter(avg_pattern, sigma=s).var() 
                          for s in range(1, 11)])
        plt.subplot(339)
        plt.plot(range(1, 11), scales)
        plt.title('Pattern Scale Analysis')
        
        plt.tight_layout()
        return fig

    def save_analysis(self, output_folder):
        """Save all analysis results."""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        # Generate and save detailed pattern analysis
        fig_analysis = self.analyze_patterns_detailed()
        fig_analysis.savefig(os.path.join(output_folder, 'detailed_pattern_analysis.png'), dpi=300)
        plt.close(fig_analysis)
        
        # Generate and save periodicity analysis
        fig_periodicity, _, _ = self.find_pattern_periodicity()
        fig_periodicity.savefig(os.path.join(output_folder, 'pattern_periodicity.png'), dpi=300)
        plt.close(fig_periodicity)
        
        # Save high-resolution maps
        fig_maps = plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(self.average_pattern, cmap='viridis')
        plt.title('Average Pattern')
        plt.subplot(132)
        plt.imshow(self.variance_map, cmap='hot')
        plt.title('Variance Map')
        plt.subplot(133)
        local_patterns = self.analyze_local_patterns(self.average_pattern)
        plt.imshow(local_patterns, cmap='magma')
        plt.title('Local Pattern Variation')
        plt.savefig(os.path.join(output_folder, 'high_res_pattern_maps.png'), dpi=300)
        plt.close(fig_maps)

# Ejemplo de uso
if __name__ == "__main__":
    # Obtener el directorio actual donde se encuentra el script
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    
    # Lista de rutas de imágenes relativas
    image_paths = [
        os.path.join(directorio_actual, "dataset", "light", "morning", "2_2024-02-22-15-15-04.jpg"),
        os.path.join(directorio_actual, "dataset", "light", "morning", "2_2024-02-22-15-50-04.jpg"),
        os.path.join(directorio_actual, "dataset", "light", "morning", "2_2024-02-22-16-05-04.jpg"),
        os.path.join(directorio_actual, "dataset", "light", "morning", "2_2024-02-22-16-30-04.jpg"),
        os.path.join(directorio_actual, "dataset", "light", "morning", "2_2024-02-22-16-40-04.jpg")
    ]
    
    # Crear la carpeta de salida de manera relativa
    output_folder = os.path.join(directorio_actual, "resultados_analisis")
    
    # Crear el analizador
    analyzer = SandPatternAnalyzer(image_paths)
    
    # Cargar y procesar las imágenes
    analyzer.load_images()
    
    # Guardar los resultados
    analyzer.save_analysis(output_folder)