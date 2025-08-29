import sys
import time
import logging
import argparse
import threading
from pathlib import Path
import concurrent.futures
from typing import List, Optional

# Bibliotecas de processamento de imagem
try:
	import numpy as np
	import cv2
	from PIL import Image, ImageFile
	from rembg import remove, new_session
except ImportError as e:
	print(f"Erro: Biblioteca necessária não encontrada: {e}")
	print("Execute: pip install opencv-python numpy pillow rembg")
	sys.exit(1)

# Permitir imagens truncadas
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------
# Configuração de Logging
# -----------------------------
def setup_logging(verbose: bool = False) -> logging.Logger:
	log_format = "%(asctime)s - %(levelname)s - %(message)s"
	log_level = logging.DEBUG if verbose else logging.INFO

	logging.basicConfig(
		level=log_level,
		format=log_format,
		handlers=[
			logging.FileHandler('processamento_imagens.log', encoding='utf-8'),
			logging.StreamHandler(sys.stdout)
		]
	)
	logger = logging.getLogger(__name__)
	logger.info("Sistema de logging inicializado")
	return logger

# -----------------------------
# Varredura de Arquivos
# -----------------------------
class FileScanner:
	SUPPORTED_EXTENSIONS = {
		'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.heic', '.heif', '.gif', '.svg'
	}

	def __init__(self, logger: logging.Logger):
		self.logger = logger

	def find_images(self, root_dir: str) -> List[Path]:
		root_path = Path(root_dir).resolve()
		self.logger.info(f"Iniciando busca de imagens em: {root_path}")
		if not root_path.exists():
			self.logger.error(f"Diretório não encontrado: {root_path}")
			return []
		files = [f for f in root_path.rglob("*") if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS]
		self.logger.info(f"Encontrados {len(files)} arquivos de imagem")
		return sorted(files)

# -----------------------------
# Detecção de Documento
# -----------------------------
class DocumentDetector:
	def __init__(self, logger: logging.Logger):
		self.logger = logger

	def is_text_document(self, image: np.ndarray, threshold: float = 0.7) -> bool:
		try:
			if len(image.shape) == 3:
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			else:
				gray = image

			hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
			total_pixels = gray.shape[0] * gray.shape[1]

			light_pixels = np.sum(hist[200:]) / total_pixels
			dark_pixels = np.sum(hist[:50]) / total_pixels

			return light_pixels > threshold and dark_pixels < 0.1
		except Exception as e:
			self.logger.error(f"Erro na detecção de documento textual: {e}")
			return False

	def detect_document_type(self, image: np.ndarray) -> str:
		try:
			h, w = image.shape[:2]
			aspect_ratio = w / h

			if self.is_text_document(image):
				if 0.65 <= aspect_ratio <= 0.75:
					return "A4_text_portrait"
				elif 1.3 <= aspect_ratio <= 1.5:
					return "A4_text_landscape"
				else:
					return "text_document"

			if 0.70 <= aspect_ratio <= 0.72:
				return "A4_portrait"
			elif 1.40 <= aspect_ratio <= 1.43:
				return "A4_landscape"
			elif 0.75 <= aspect_ratio <= 0.85:
				return "letter"
			elif aspect_ratio > 2.5:
				return "banner"
			elif aspect_ratio < 0.5:
				return "receipt"
			else:
				return "custom"
		except Exception as e:
			self.logger.error(f"Erro na detecção de tipo de documento: {e}")
			return "unknown"

# -----------------------------
# Processamento de Imagem
# -----------------------------
class ImageProcessor:
	def __init__(self, logger: logging.Logger, rembg_model: str = 'u2net'):
		self.logger = logger
		self.file_lock = threading.Lock()
		self.stats_lock = threading.Lock()
		self.document_detector = DocumentDetector(logger)
		self.stats = {'processed': 0, 'errors': 0, 'skipped': 0}

		try:
			self.rembg_session = new_session(rembg_model)
			self.logger.info(f"Modelo rembg '{rembg_model}' carregado com sucesso")
		except Exception as e:
			self.logger.warning(f"Erro ao carregar modelo rembg: {e}. Usando padrão")
			self.rembg_session = None

	# -----------------------------
	# Funções básicas
	# -----------------------------
	def load_image(self, path: Path) -> Optional[np.ndarray]:
		try:
			if path.suffix.lower() in ['.heic', '.heif']:
				pil_img = Image.open(path)
				img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
			else:
				img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

			if img is None:
				self.logger.error(f"Falha ao decodificar: {path}")
				return None

			if len(img.shape) == 3:
				if img.shape[2] == 4:
					img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
				elif img.shape[2] == 3:
					img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			return img
		except Exception as e:
			self.logger.error(f"Erro ao carregar {path}: {e}")
			return None

	def remove_background(self, image: np.ndarray) -> np.ndarray:
		try:
			rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			pil_img = Image.fromarray(rgb)
			result = remove(pil_img, session=self.rembg_session) if self.rembg_session else remove(pil_img)
			arr = np.array(result)
			return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA) if arr.shape[2] == 4 else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
		except Exception as e:
			self.logger.error(f"Erro na remoção de fundo: {e}")
			return image

	def save_image(self, image: np.ndarray, path: Path) -> bool:
		try:
			path.parent.mkdir(parents=True, exist_ok=True)
			with self.file_lock:
				success = cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
			if not success:
				self.logger.error(f"Falha ao salvar: {path}")
			return success
		except Exception as e:
			self.logger.error(f"Erro ao salvar {path}: {e}")
			return False

	# -----------------------------
	# Detecção e corte de documentos
	# -----------------------------
	def find_document_contour(self, image: np.ndarray, doc_type: str = "") -> Optional[np.ndarray]:
		try:
			if doc_type and "text" in doc_type:
				return self._find_text_document_contour(image)
			else:
				return self._find_general_document_contour(image)
		except Exception as e:
			self.logger.error(f"Erro na detecção de contorno: {e}")
			return None

	def _find_text_document_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
		try:
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
			kernel = np.ones((5, 5), np.uint8)
			closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
			contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			if not contours:
				return None
			contours = sorted(contours, key=cv2.contourArea, reverse=True)
			for contour in contours[:3]:
				area = cv2.contourArea(contour)
				if area < (image.shape[0] * image.shape[1] * 0.1):
					continue
				peri = cv2.arcLength(contour, True)
				approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
				if len(approx) == 4:
					return approx
				else:
					return contour
			return None
		except Exception as e:
			self.logger.error(f"Erro na detecção de contorno textual: {e}")
			return None

	def _find_general_document_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
		try:
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			gray = cv2.equalizeHist(gray)
			blurred = cv2.GaussianBlur(gray, (7, 7), 0)
			edges = cv2.Canny(blurred, 30, 100)
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
			edges = cv2.dilate(edges, kernel, iterations=2)
			edges = cv2.erode(edges, kernel, iterations=1)
			contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			if not contours:
				return None
			contour = max(contours, key=cv2.contourArea)
			peri = cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
			return approx if len(approx) == 4 else contour
		except Exception as e:
			self.logger.error(f"Erro na detecção de contorno geral: {e}")
			return None

	def auto_crop_document(self, image: np.ndarray, doc_type: str = "termo") -> np.ndarray:
		try:
			h, w = image.shape[:2]

			if doc_type in ["A4_portrait", "A4_landscape"]:
				self.logger.info(f"Documento A4 detectado ({doc_type}), cortando nas bordas")
				margin = 2
				return image[margin:h-margin, margin:w-margin]

			contour = self.find_document_contour(image, doc_type)
			if contour is None:
				self.logger.warning("Nenhum contorno encontrado, usando original")
				return image

			if len(contour) == 4:
				pts = contour.reshape(4, 2).astype("float32")
				area = cv2.contourArea(contour)
				if area < (h * w * 0.1):
					self.logger.warning("Contorno muito pequeno, usando original")
					return image

				s = pts.sum(axis=1)
				rect = np.zeros((4, 2), dtype="float32")
				rect[0] = pts[np.argmin(s)]
				rect[2] = pts[np.argmax(s)]
				diff = np.diff(pts, axis=1)
				rect[1] = pts[np.argmin(diff)]
				rect[3] = pts[np.argmax(diff)]

				(tl, tr, br, bl) = rect
				widthA = np.linalg.norm(br - bl)
				widthB = np.linalg.norm(tr - tl)
				maxWidth = max(int(widthA), int(widthB))
				heightA = np.linalg.norm(tr - br)
				heightB = np.linalg.norm(tl - bl)
				maxHeight = max(int(heightA), int(heightB))

				dst = np.array([
					[0, 0],
					[maxWidth - 1, 0],
					[maxWidth - 1, maxHeight - 1],
					[0, maxHeight - 1]
				], dtype="float32")

				M = cv2.getPerspectiveTransform(rect, dst)
				warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

				if "text" in doc_type:
					border_size = 20
					warped = cv2.copyMakeBorder(
						warped, border_size, border_size, border_size, border_size,
						cv2.BORDER_CONSTANT, value=[255, 255, 255]
					)
				return warped
			else:
				x, y, w_box, h_box = cv2.boundingRect(contour)
				margin = 10
				x = max(0, x - margin)
				y = max(0, y - margin)
				w_box = min(image.shape[1] - x, w_box + 2 * margin)
				h_box = min(image.shape[0] - y, h_box + 2 * margin)
				return image[y:y+h_box, x:x+w_box]

		except Exception as e:
			self.logger.error(f"Erro no recorte do documento: {e}")
			return image

	# -----------------------------
	# Processamento batch
	# -----------------------------
	def process_single_image(self, path: Path, output_base: Path, input_base: Path) -> bool:
		try:
			start = time.time()
			self.logger.info(f"Processando: {path.name}")
			img = self.load_image(path)
			if img is None:
				with self.stats_lock:
					self.stats['errors'] += 1
				return False

			doc_type = self.document_detector.detect_document_type(img)
			self.logger.debug(f"Tipo detectado: {doc_type}")

			if "text" in doc_type:
				no_bg = img
			else:
				no_bg = self.remove_background(img)

			final = self.auto_crop_document(no_bg, doc_type)

			# -----------------------------------
			# RESTAURAR A COR ORIGINAL
			# -----------------------------------
			if final.shape[2] == 4:
				final = cv2.cvtColor(final, cv2.COLOR_BGRA2BGR)
			final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

			relative_path = path.resolve().relative_to(input_base.resolve())
			out_path = output_base / relative_path.with_suffix('.png')
			success = self.save_image(final, out_path)

			with self.stats_lock:
				self.stats['processed' if success else 'errors'] += 1

			self.logger.info(f"✓ {path.name} ({time.time()-start:.2f}s) - Tipo: {doc_type}")
			return success
		except Exception as e:
			self.logger.error(f"Erro ao processar {path}: {e}")
			with self.stats_lock:
				self.stats['errors'] += 1
			return False

	def process_images_batch(self, image_files: List[Path], output_dir: str, input_dir: str, max_workers: int = 4) -> dict:
		output_base = Path(output_dir).resolve()
		input_base = Path(input_dir).resolve()
		with self.stats_lock:
			self.stats = {'processed': 0, 'errors': 0, 'skipped': 0}

		start_time = time.time()
		self.logger.info(f"🚀 Iniciando processamento de {len(image_files)} imagens com {max_workers} threads")

		with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
			futures = [executor.submit(self.process_single_image, f, output_base, input_base) for f in image_files]
			for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
				try:
					fut.result()
					if i % 10 == 0 or i == len(image_files):
						with self.stats_lock:
							self.logger.info(
								f"📊 Progresso: {i}/{len(image_files)} "
								f"(✅ {self.stats['processed']} | ❌ {self.stats['errors']})"
							)
				except Exception as e:
					self.logger.error(f"💥 Erro em thread: {e}")

		total_time = time.time() - start_time
		with self.stats_lock:
				stats = self.stats.copy()
		stats['total_time'] = total_time
		stats['images_per_second'] = len(image_files) / total_time if total_time > 0 else 0

		self.logger.info("📦 Processamento finalizado")
		self.logger.info(f"   ✅ Processadas: {stats['processed']}")
		self.logger.info(f"   ❌ Erros: {stats['errors']}")
		self.logger.info(f"   ⏱️ Tempo total: {stats['total_time']:.2f}s")
		self.logger.info(f"   ⚡ Velocidade: {stats['images_per_second']:.2f} imgs/s")

		return stats

# -----------------------------
# Função Principal
# -----------------------------
def main():
	parser = argparse.ArgumentParser(
		description='Processamento automático de imagens',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument('-i','--input-dir', default='.', help='Diretório de entrada')
	parser.add_argument('-o','--output-dir', default='./saida_da_foto', help='Diretório de saída')
	parser.add_argument('-w','--max-workers', type=int, default=4, help='Threads paralelas')
	parser.add_argument('-v','--verbose', action='store_true', help='Logging detalhado')
	parser.add_argument('--rembg-model', default='u2net', choices=['u2net','u2netp','silueta'], help='Modelo rembg')
	args = parser.parse_args()

	logger = setup_logging(args.verbose)
	logger.info("="*60)
	logger.info("PROCESSADOR AUTOMÁTICO DE IMAGENS")
	logger.info("="*60)

	file_scanner = FileScanner(logger)
	image_processor = ImageProcessor(logger, args.rembg_model)
	image_files = file_scanner.find_images(args.input_dir)

	if not image_files:
		logger.warning("Nenhuma imagem encontrada!")
		return

	print(f"\n🚀 Processando {len(image_files)} imagens em: {args.input_dir}")
	stats = image_processor.process_images_batch(image_files, args.output_dir, args.input_dir, args.max_workers)

	logger.info("="*60)
	logger.info(f"✅ Processadas: {stats['processed']}")
	logger.info(f"❌ Erros: {stats['errors']}")
	logger.info(f"⏱️ Tempo total: {stats['total_time']:.2f}s")
	logger.info(f"Velocidade: {stats['images_per_second']:.2f} imgs/s")
	print(f"\n🎉 Processamento concluído! Saída em: {args.output_dir}")

if __name__ == "__main__":
	main()
