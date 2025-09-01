import sys
import time
import logging
import argparse
import threading
from pathlib import Path
import concurrent.futures
from typing import List, Optional, Dict, Any
import numpy as np

# Bibliotecas de processamento de imagem
try:
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
			logging.FileHandler('imagens.log', encoding='utf-8'),
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
		'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'
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
# Analisador Inteligente de Imagens
# -----------------------------
class ImageAnalyzer:
	def __init__(self, logger: logging.Logger):
		self.logger = logger

	def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
		"""
		Analisa a imagem e retorna características para decidir o melhor processamento
		"""
		try:
			results = {
				'type': 'unknown',
				'is_text_document': False,
				'has_clear_background': False,
				'contrast_level': 'medium',
				'recommended_method': 'standard'
			}

			if len(image.shape) == 3:
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			else:
				gray = image

			# Calcular métricas
			height, width = gray.shape
			hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
			total_pixels = height * width

			# Percentuais de intensidade
			dark_pixels = np.sum(hist[:50]) / total_pixels    # 0-50 (escuro)
			mid_pixels = np.sum(hist[50:200]) / total_pixels  # 50-200 (médio)
			light_pixels = np.sum(hist[200:]) / total_pixels  # 200-255 (claro)

			# Contraste
			contrast = np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0

			# Detectar se é documento textual
			is_text_like = (dark_pixels > 0.08 and light_pixels > 0.4 and contrast > 0.4)

			# Verificar fundo uniforme
			background_uniform = (light_pixels > 0.6 and dark_pixels < 0.1)

			# Determinar tipo
			if is_text_like:
				results['type'] = 'text_document'
				results['is_text_document'] = True
				results['recommended_method'] = 'text_optimized'
			elif background_uniform:
				results['type'] = 'clean_background'
				results['has_clear_background'] = True
				results['recommended_method'] = 'aggressive'
			else:
				results['type'] = 'complex_image'
				results['recommended_method'] = 'standard'

			# Nível de contraste
			if contrast > 0.6:
				results['contrast_level'] = 'high'
			elif contrast < 0.3:
				results['contrast_level'] = 'low'

			self.logger.debug(f"Análise: {results}")
			return results

		except Exception as e:
			self.logger.error(f"Erro na análise da imagem: {e}")
			return {'type': 'error', 'recommended_method': 'standard'}

# -----------------------------
# Processamento de Imagem para PDF
# -----------------------------
class PDFProcessor:
	def __init__(self, logger: logging.Logger, rembg_model: str = 'u2net'):
		self.logger = logger
		self.file_lock = threading.Lock()
		self.stats_lock = threading.Lock()
		self.analyzer = ImageAnalyzer(logger)
		self.stats = {
			'processed': 0,
			'errors': 0,
			'text_documents': 0,
			'clean_backgrounds': 0,
			'complex_images': 0
		}

		try:
			self.rembg_session = new_session(rembg_model)
			self.logger.info(f"Modelo rembg '{rembg_model}' carregado com sucesso")
		except Exception as e:
			self.logger.warning(f"Erro ao carregar modelo rembg: {e}. Usando padrão")
			self.rembg_session = None

	def load_image(self, path: Path) -> Optional[np.ndarray]:
		try:
			img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
			if img is None:
				self.logger.error(f"Falha ao decodificar: {path}")
				return None
			return img
		except Exception as e:
			self.logger.error(f"Erro ao carregar {path}: {e}")
			return None

	def remove_background_standard(self, image: np.ndarray) -> np.ndarray:
		"""Remoção de fundo padrão"""
		try:
			rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			pil_img = Image.fromarray(rgb)
			result = remove(pil_img, session=self.rembg_session)
			arr = np.array(result)

			if arr.shape[2] == 4:
				return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
			else:
				return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

		except Exception as e:
			self.logger.error(f"Erro na remoção de fundo padrão: {e}")
			return image

	def remove_background_aggressive(self, image: np.ndarray) -> np.ndarray:
		"""Remoção mais agressiva para fundos limpos"""
		try:
			# Primeiro remoção normal
			rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			pil_img = Image.fromarray(rgb)
			result = remove(pil_img, session=self.rembg_session)
			arr = np.array(result)

			# Aplicar processamento adicional para fundos limpos
			if arr.shape[2] == 4:
				alpha = arr[:, :, 3]
				# Tornar transparência mais definida
				_, alpha_processed = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)
				arr[:, :, 3] = alpha_processed

			return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)

		except Exception as e:
			self.logger.error(f"Erro na remoção agressiva: {e}")
			return self.remove_background_standard(image)

	def process_text_document(self, image: np.ndarray) -> np.ndarray:
		"""Processamento otimizado para documentos textuais"""
		try:
			# Para documentos textuais, melhorar a imagem em vez de remover fundo
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			# Aplicar CLAHE para melhorar contraste
			clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			enhanced = clahe.apply(gray)

			# Redução de ruído
			denoised = cv2.medianBlur(enhanced, 3)

			# Converter de volta para BGR
			result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

			self.logger.info("📄 Documento textual - Aplicando enhancement em vez de remoção de fundo")
			return result

		except Exception as e:
			self.logger.error(f"Erro no processamento de texto: {e}")
			return image

	def remove_background(self, image: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
		"""Seleciona o método de remoção baseado na análise"""
		method = analysis['recommended_method']

		if method == 'text_optimized':
			return self.process_text_document(image)
		elif method == 'aggressive':
			return self.remove_background_aggressive(image)
		else:
			return self.remove_background_standard(image)

	def save_as_pdf(self, image: np.ndarray, output_path: Path) -> bool:
		try:
			output_path.parent.mkdir(parents=True, exist_ok=True)

			if image.shape[2] == 4:
				image_rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
				pil_img = Image.fromarray(image_rgba, 'RGBA')
				background = Image.new('RGB', pil_img.size, (255, 255, 255))
				background.paste(pil_img, mask=pil_img.split()[3])
				final_image = background
			else:
				image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				pil_img = Image.fromarray(image_rgb, 'RGB')
				final_image = pil_img

			final_image.save(output_path, "PDF", resolution=100.0, title="Documento processado")
			return True

		except Exception as e:
			self.logger.error(f"Erro ao salvar PDF {output_path}: {e}")
			return False

	def process_single_image(self, input_path: Path, output_dir: Path) -> bool:
		try:
			start = time.time()
			self.logger.info(f"Processando: {input_path.name}")

			img = self.load_image(input_path)
			if img is None:
				with self.stats_lock:
					self.stats['errors'] += 1
				return False

			# Analisar imagem para decidir o melhor método
			analysis = self.analyzer.analyze_image(img)
			self.logger.info(f"Tipo detectado: {analysis['type']} - Método: {analysis['recommended_method']}")

			# Aplicar o método apropriado
			processed_img = self.remove_background(img, analysis)

			# Atualizar estatísticas
			with self.stats_lock:
				if analysis['type'] == 'text_document':
					self.stats['text_documents'] += 1
				elif analysis['type'] == 'clean_background':
					self.stats['clean_backgrounds'] += 1
				elif analysis['type'] == 'complex_image':
					self.stats['complex_images'] += 1

			# Salvar
			output_filename = f"{input_path.stem}_processado.pdf"
			output_path = output_dir / output_filename
			success = self.save_as_pdf(processed_img, output_path)

			with self.stats_lock:
				self.stats['processed' if success else 'errors'] += 1

			self.logger.info(f"✓ {input_path.name} -> {output_filename} ({time.time()-start:.2f}s)")
			return success

		except Exception as e:
			self.logger.error(f"Erro ao processar {input_path}: {e}")
			with self.stats_lock:
				self.stats['errors'] += 1
			return False

	def process_images_to_pdf(self, image_files: List[Path], output_dir: str, max_workers: int = 4) -> dict:
		output_path = Path(output_dir).resolve()
		output_path.mkdir(parents=True, exist_ok=True)

		with self.stats_lock:
			self.stats = {'processed': 0, 'errors': 0, 'text_documents': 0, 'clean_backgrounds': 0, 'complex_images': 0}

		start_time = time.time()
		self.logger.info(f"🚀 Iniciando processamento inteligente de {len(image_files)} imagens")

		with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
			futures = [
				executor.submit(self.process_single_image, img_file, output_path)
				for img_file in image_files
			]

			for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
				try:
					fut.result()
					if i % 5 == 0 or i == len(image_files):
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
		return stats

# -----------------------------
# Função Principal
# -----------------------------
def main():
	parser = argparse.ArgumentParser(
		description='Remover fundo de imagens inteligentemente e salvar como PDF',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument('-i', '--input-dir', default='.', help='Diretório de entrada')
	parser.add_argument('-o', '--output-dir', default='./output', help='Diretório de saída')
	parser.add_argument('-w', '--max-workers', type=int, default=4, help='Threads paralelas')
	parser.add_argument('-v', '--verbose', action='store_true', help='Logging detalhado')
	parser.add_argument('--rembg-model', default='u2net', choices=['u2net', 'u2netp', 'silueta'], help='Modelo rembg')

	args = parser.parse_args()

	logger = setup_logging(args.verbose)
	logger.info("="*60)
	logger.info("PROCESSADOR INTELIGENTE DE IMAGENS")
	logger.info("="*60)

	file_scanner = FileScanner(logger)
	image_files = file_scanner.find_images(args.input_dir)

	if not image_files:
		logger.warning("Nenhuma imagem encontrada!")
		return

	print(f"\n🚀 Processando {len(image_files)} imagens de: {args.input_dir}")
	print(f"📁 Saída em: {args.output_dir}")
	print("🤖 Modo inteligente ativado - detectando tipos de imagem automaticamente")
	print("⏳ Isso pode levar alguns minutos...\n")

	pdf_processor = PDFProcessor(logger, args.rembg_model)
	stats = pdf_processor.process_images_to_pdf(image_files, args.output_dir, args.max_workers)

	logger.info("="*60)
	logger.info("RESUMO DO PROCESSAMENTO INTELIGENTE")
	logger.info("="*60)
	logger.info(f"✅ PDFs criados com sucesso: {stats['processed']}")
	logger.info(f"📄 Documentos textuais: {stats['text_documents']}")
	logger.info(f"✨ Fundos limpos: {stats['clean_backgrounds']}")
	logger.info(f"🎨 Imagens complexas: {stats['complex_images']}")
	logger.info(f"❌ Erros: {stats['errors']}")
	logger.info(f"⏱️ Tempo total: {stats['total_time']:.2f}s")
	logger.info(f"⚡ Velocidade: {stats['images_per_second']:.2f} imgs/s")

	print(f"\n🎉 Processamento inteligente concluído!")
	print(f"📊 Estatísticas:")
	print(f"   📄 Documentos textuais: {stats['text_documents']}")
	print(f"   ✨ Fundos limpos: {stats['clean_backgrounds']}")
	print(f"   🎨 Imagens complexas: {stats['complex_images']}")
	print(f"   ❌ Erros: {stats['errors']}")
	print(f"⏰ Tempo total: {stats['total_time']:.2f} segundos")

if __name__ == "__main__":
	main()
