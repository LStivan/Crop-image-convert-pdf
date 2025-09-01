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
# Processamento de Imagem para PDF
# -----------------------------
class PDFProcessor:
	def __init__(self, logger: logging.Logger, rembg_model: str = 'u2net'):
		self.logger = logger
		self.file_lock = threading.Lock()
		self.stats_lock = threading.Lock()
		self.stats = {'processed': 0, 'errors': 0}

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

	def remove_background(self, image: np.ndarray) -> np.ndarray:
		try:
			# Converter BGR para RGB para o rembg
			rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			pil_img = Image.fromarray(rgb)

			# Remover fundo
			result = remove(pil_img, session=self.rembg_session) if self.rembg_session else remove(pil_img)

			# Converter de volta para numpy array
			arr = np.array(result)

			# Manter transparência se disponível
			if arr.shape[2] == 4:
				return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
			else:
				return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

		except Exception as e:
			self.logger.error(f"Erro na remoção de fundo: {e}")
			return image

	def save_as_pdf(self, image: np.ndarray, output_path: Path) -> bool:
		try:
			output_path.parent.mkdir(parents=True, exist_ok=True)

			# Converter numpy array para PIL Image
			if image.shape[2] == 4:  # BGRA (com transparência)
				# Converter BGRA para RGBA
				image_rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
				pil_img = Image.fromarray(image_rgba, 'RGBA')

				# Criar fundo branco para imagens transparentes
				background = Image.new('RGB', pil_img.size, (255, 255, 255))
				background.paste(pil_img, mask=pil_img.split()[3])  # Usar canal alpha como máscara
				final_image = background
			else:  # BGR (sem transparência)
				# Converter BGR para RGB
				image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				pil_img = Image.fromarray(image_rgb, 'RGB')
				final_image = pil_img

			# Salvar como PDF
			final_image.save(output_path, "PDF", resolution=100.0, title="Documento sem fundo")

			return True

		except Exception as e:
				self.logger.error(f"Erro ao salvar PDF {output_path}: {e}")
				return False

	def process_single_image(self, input_path: Path, output_dir: Path) -> bool:
		try:
				start = time.time()
				self.logger.info(f"Processando: {input_path.name}")

				# Carregar imagem
				img = self.load_image(input_path)
				if img is None:
					with self.stats_lock:
						self.stats['errors'] += 1
					return False

				# Remover fundo
				img_sem_fundo = self.remove_background(img)

				# Criar nome do arquivo de saída
				output_filename = f"{input_path.stem}_sem_fundo.pdf"
				output_path = output_dir / output_filename

				# Salvar como PDF
				success = self.save_as_pdf(img_sem_fundo, output_path)

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
			self.stats = {'processed': 0, 'errors': 0}

		start_time = time.time()
		self.logger.info(f"🚀 Iniciando processamento de {len(image_files)} imagens para PDF")

		# Processar imagens em paralelo
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
		description='Remover fundo de imagens e salvar como PDF individuais',
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
	logger.info("REMOVEDOR DE FUNDO - IMAGENS PARA PDF")
	logger.info("="*60)

	# Encontrar imagens
	file_scanner = FileScanner(logger)
	image_files = file_scanner.find_images(args.input_dir)

	if not image_files:
		logger.warning("Nenhuma imagem encontrada!")
		return

	print(f"\n🚀 Processando {len(image_files)} imagens de: {args.input_dir}")
	print(f"📁 Saída em: {args.output_dir}")
	print("⏳ Isso pode levar alguns minutos...\n")

	# Processar imagens
	pdf_processor = PDFProcessor(logger, args.rembg_model)
	stats = pdf_processor.process_images_to_pdf(image_files, args.output_dir, args.max_workers)

	# Resultados
	logger.info("="*60)
	logger.info("RESUMO DO PROCESSAMENTO")
	logger.info("="*60)
	logger.info(f"✅ PDFs criados com sucesso: {stats['processed']}")
	logger.info(f"❌ Erros: {stats['errors']}")
	logger.info(f"⏱️ Tempo total: {stats['total_time']:.2f}s")
	logger.info(f"⚡ Velocidade: {stats['images_per_second']:.2f} imgs/s")

	print(f"\n🎉 Processamento concluído!")
	print(f"📄 {stats['processed']} PDFs criados em: {args.output_dir}")
	print(f"⏰ Tempo total: {stats['total_time']:.2f} segundos")

if __name__ == "__main__":
	main()
