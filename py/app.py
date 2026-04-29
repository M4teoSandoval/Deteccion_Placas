from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from ultralytics import YOLO
import easyocr
import logging
import re
import io

try:
    from PIL import Image
    PIL_DISPONIBLE = True
except ImportError:
    PIL_DISPONIBLE = False

# -----------------------
# LOGS
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yolo-plates")

app = FastAPI()

logger.info("🔹 Cargando modelo YOLOv8 desde best.pt ...")
model = YOLO("best.pt")
logger.info("✅ Modelo YOLOv8 cargado correctamente.")

logger.info("🔹 Inicializando EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)
logger.info("✅ EasyOCR listo.")

# -----------------------
# PATRONES VÁLIDOS
# -----------------------
PATRON_CARRO = re.compile(r'^[A-Z]{3}[0-9]{3}$')       # ABC123
PATRON_MOTO  = re.compile(r'^[A-Z]{3}[0-9]{2}[A-Z]$')  # ABC12D

# Textos decorativos que aparecen en placas colombianas — NO son matrícula
TEXTOS_DECORATIVOS = {
    'BOGOTA', 'BOG07A', 'COLOMBIA', 'STAFE', 'MEDELLIN', 'CALI',
    'BARRANQUILLA', 'CARTAGENA', 'CUNDINAMARCA', 'ANTIOQUIA',
    'BOYACA', 'TOLIMA', 'HUILA', 'NARINO', 'CAUCA', 'VALLE',
    'TRANSIT', 'TRANSITO', 'POLICIA', 'EJERCITO', 'ARMADA',
    'REPUBLICA', 'REPUBLIC',
}

def es_texto_decorativo(texto):
    texto = re.sub(r'[^A-Z]', '', texto.upper())
    return texto in TEXTOS_DECORATIVOS or len(texto) > 8

def es_placa(texto):
    if es_texto_decorativo(texto):
        return False
    return PATRON_CARRO.match(texto) or PATRON_MOTO.match(texto)

# -----------------------
# DETECCIÓN DE FORMATO POR MAGIC BYTES
# -----------------------
def detectar_formato(data: bytes) -> str:
    if data[:3] == b'\xff\xd8\xff':
        return 'jpeg'
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'webp'
    if data[4:12] in (b'ftypheic', b'ftypheix', b'ftypheim',
                      b'ftyphevx', b'ftypmif1', b'ftypmsf1'):
        return 'heic'
    return 'desconocido'

# -----------------------
# DECODIFICACIÓN ROBUSTA
# -----------------------
def decodificar_imagen(contents: bytes):
    formato = detectar_formato(contents)
    logger.info(f"  📄 Formato detectado: {formato} ({len(contents)} bytes)")

    # Intento 1: OpenCV (JPG, PNG, BMP)
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is not None:
        logger.info(f"  ✅ Decodificada con OpenCV ({img.shape[1]}x{img.shape[0]})")
        return img

    logger.warning(f"  ⚠️ OpenCV falló con '{formato}', intentando Pillow...")

    # Intento 2: Pillow (WebP, GIF, etc.)
    if PIL_DISPONIBLE:
        try:
            pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            logger.info(f"  ✅ Decodificada con Pillow ({img.shape[1]}x{img.shape[0]})")
            return img
        except Exception as e:
            logger.warning(f"  ⚠️ Pillow falló: {e}")

    # Intento 3: HEIC con pillow-heif
    if formato == 'heic':
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
            pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            logger.info(f"  ✅ Decodificada con pillow-heif ({img.shape[1]}x{img.shape[0]})")
            return img
        except ImportError:
            logger.error("  ❌ HEIC sin soporte. Instala: pip install pillow-heif")
        except Exception as e:
            logger.error(f"  ❌ pillow-heif falló: {e}")

    return None

# -----------------------
# CORRECCIÓN OCR POSICIONAL
# Posiciones 0,1,2 → letras  (corrige dígitos que parecen letras)
# Posiciones 3,4,5 → números (corrige letras que parecen dígitos)
# -----------------------
def corregir_ocr(texto):
    texto = texto.upper().strip()
    texto = re.sub(r'[^A-Z0-9]', '', texto)

    if len(texto) == 6:
        LETRA_A_NUMERO = {
            'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'E': '3',
            'S': '5', 'G': '6', 'T': '7', 'B': '8', 'Q': '0'
        }
        NUMERO_A_LETRA = {
            '0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G'
        }
        lista = list(texto)
        for i in range(3):
            if lista[i].isdigit():
                lista[i] = NUMERO_A_LETRA.get(lista[i], lista[i])
        for i in range(3, 6):
            if lista[i].isalpha():
                lista[i] = LETRA_A_NUMERO.get(lista[i], lista[i])
        texto = ''.join(lista)

    return texto

# -----------------------
# SCORES POR TIPO DE DETECCIÓN
# Lectura directa >> ventana fragmento solo >> par exacto >> subcadena de par
# -----------------------
SCORE_DIRECTO      = 1.0   # fragmento leído limpio de 6 chars
SCORE_VENTANA_SOLO = 0.5   # ventana sobre fragmento solo con ruido (ej: IDVH13I)
SCORE_PAR_EXACTO   = 0.7   # dos fragmentos concatenados dan exactamente 6 chars
SCORE_SUB_PAR      = 0.3   # subcadena de ventana sobre par — máxima penalización

# -----------------------
# RECONSTRUCCIÓN DESDE FRAGMENTOS
# -----------------------
def reconstruir_desde_fragmentos(fragmentos_con_conf):
    """
    Intenta reconstruir la placa desde fragmentos OCR usando 4 estrategias
    en orden de confianza decreciente:

    1. Fragmento directo de 6 chars                    (score × 1.0)
    2. Ventana deslizante sobre fragmento solo 7+ chars (score × 0.5)
       → cubre 'IDVH13I' → DVH13I en [1:7]
    3. Par de fragmentos que da exactamente 6 chars     (avg × 0.7)
    4. Ventana sobre par complementario (letras+nums)   (avg × 0.3)

    Devuelve lista de (placa, score, motivo) ordenada de mayor a menor.
    """
    candidatos = {}

    # Filtrar decorativos antes de combinar
    fragmentos = []
    for texto, conf in fragmentos_con_conf:
        limpio = re.sub(r'[^A-Z0-9]', '', texto.upper().strip())
        if limpio and not es_texto_decorativo(limpio):
            fragmentos.append((limpio, conf))

    if not fragmentos:
        return []

    def registrar(placa, score, motivo):
        if placa not in candidatos or candidatos[placa][0] < score:
            candidatos[placa] = (score, motivo)

    def mayoria_letras(s):
        return sum(c.isalpha() for c in s) > len(s) / 2

    def mayoria_numeros(s):
        return sum(c.isdigit() for c in s) > len(s) / 2

    # --- Estrategia 1: fragmento directo ---
    for f, conf in fragmentos:
        candidato = corregir_ocr(f)
        if len(candidato) == 6 and es_placa(candidato):
            score = conf * SCORE_DIRECTO
            registrar(candidato, score, f"directo '{f}'")
            logger.info(f"    ✅ Directo: '{f}' → {candidato} (score={score:.2f})")

    # --- Estrategia 2: ventana sobre fragmento solo con ruido en bordes ---
    # Cubre casos como 'IDVH13I' (7 chars) donde la placa real es DVH13I
    for f, conf in fragmentos:
        if len(f) > 6:
            score_base = conf * SCORE_VENTANA_SOLO
            for start in range(len(f) - 5):
                sub = f[start:start+6]
                candidato = corregir_ocr(sub)
                if len(candidato) == 6 and es_placa(candidato):
                    registrar(candidato, score_base, f"ventana_solo '{f}'[{start}:{start+6}]")
                    logger.info(f"    ✅ Ventana solo: '{f}'[{start}:{start+6}] → {candidato} (score={score_base:.2f})")

    # --- Estrategia 3 y 4: pares de fragmentos ---
    for i in range(len(fragmentos)):
        for j in range(len(fragmentos)):
            if i == j:
                continue
            fa, conf_a = fragmentos[i]
            fb, conf_b = fragmentos[j]
            conf_avg = (conf_a + conf_b) / 2
            concat = fa + fb

            # Par exacto de 6 chars
            candidato = corregir_ocr(concat)
            if len(candidato) == 6 and es_placa(candidato):
                score = conf_avg * SCORE_PAR_EXACTO
                registrar(candidato, score, f"par '{fa}'+'{fb}'")
                logger.info(f"    ✅ Par exacto: '{concat}' → {candidato} (score={score:.2f})")
                continue

            # Ventana sobre par solo si son complementarios (letras+números)
            if (mayoria_letras(fa) and mayoria_numeros(fb)) or \
               (mayoria_numeros(fa) and mayoria_letras(fb)):
                for start in range(len(concat) - 5):
                    sub = concat[start:start+6]
                    candidato_sub = corregir_ocr(sub)
                    if len(candidato_sub) == 6 and es_placa(candidato_sub):
                        score = conf_avg * SCORE_SUB_PAR
                        registrar(candidato_sub, score, f"sub_par '{fa}'+'{fb}'[{start}:{start+6}]")
                        logger.info(f"    ✅ Sub par: '{concat}'[{start}:{start+6}] → {candidato_sub} (score={score:.2f})")

    return sorted(
        [(p, s, m) for p, (s, m) in candidatos.items()],
        key=lambda x: x[1],
        reverse=True
    )

# -----------------------
# RUTA BASE
# -----------------------
@app.get("/")
def home():
    return {"message": "API de placas funcionando 🚀"}

# -----------------------
# PREPROCESAMIENTO
# -----------------------
TARGET_W     = 400
SCORE_MINIMO = 0.30

def preparar_para_ocr(img_bgr):
    h, w = img_bgr.shape[:2]
    if w != TARGET_W:
        factor = TARGET_W / w
        new_h = max(1, int(h * factor))
        img_bgr = cv2.resize(img_bgr, (TARGET_W, new_h), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    thresh = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return [
        ("color",     img_bgr),
        ("gray_blur", gray_blur),
        ("clahe",     clahe),
        ("thresh",    thresh),
    ]

# -----------------------
# OCR SOBRE REGIÓN
# -----------------------
def ocr_sobre_region(region_bgr):
    """
    Corre OCR en 4 variantes de preprocesamiento.
    Acumula candidatos con score global y devuelve solo el mejor
    que supere SCORE_MINIMO.
    """
    variantes = preparar_para_ocr(region_bgr)
    global_scores = {}  # placa → (score, motivo)

    def registrar_global(placa, score, motivo):
        if placa not in global_scores or global_scores[placa][0] < score:
            global_scores[placa] = (score, motivo)

    for nombre, variante in variantes:
        try:
            resultados = reader.readtext(variante, detail=1)
            if not resultados:
                logger.info(f"    [{nombre}] OCR no leyó nada")
                continue

            fragmentos_con_conf = []

            for (_, text, conf) in resultados:
                texto_limpio = corregir_ocr(text)
                es_deco = es_texto_decorativo(
                    re.sub(r'[^A-Z0-9]', '', text.upper())
                )
                tag = "🎨 decorativo" if es_deco else ""
                logger.info(f"    [{nombre}] OCR='{text}' → '{texto_limpio}' (conf={conf:.2f}) {tag}")

                if not es_deco:
                    fragmentos_con_conf.append((text, conf))

                # Intento directo si no es decorativo
                if not es_deco and len(texto_limpio) == 6 and es_placa(texto_limpio):
                    score = conf * SCORE_DIRECTO
                    registrar_global(texto_limpio, score, f"[{nombre}] directo")
                    logger.info(f"      ✅ DIRECTO: {texto_limpio} (score={score:.2f})")

            # Reconstrucción desde fragmentos
            reconstruidas = reconstruir_desde_fragmentos(fragmentos_con_conf)
            for placa, score, motivo in reconstruidas:
                registrar_global(placa, score, f"[{nombre}] {motivo}")

        except Exception as e:
            logger.warning(f"    [{nombre}] ⚠️ Error en OCR: {e}")
            continue

    if not global_scores:
        return []

    ranking = sorted(global_scores.items(), key=lambda x: x[1][0], reverse=True)
    logger.info(f"    📊 Ranking: {[(p, round(s, 2), m) for p, (s, m) in ranking[:5]]}")

    validos = [(p, s, m) for p, (s, m) in ranking if s >= SCORE_MINIMO]
    if not validos:
        p, (s, m) = ranking[0]
        logger.warning(f"    ⚠️ Sin candidatos sobre score={SCORE_MINIMO}. Mejor: '{p}' ({s:.2f})")
        return []

    mejor_placa, mejor_score, mejor_motivo = validos[0]
    logger.info(f"    🏆 Elegida: {mejor_placa} (score={mejor_score:.2f}, via {mejor_motivo})")
    return [mejor_placa]

# -----------------------
# PREDICCIÓN
# -----------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        logger.info("📩 Petición recibida en /predict/")
        logger.info(f"  📄 filename='{file.filename}' content_type='{file.content_type}'")

        contents = await file.read()

        if not contents:
            return JSONResponse(
                status_code=400,
                content={"error": "El archivo recibido está vacío."}
            )

        img = decodificar_imagen(contents)

        if img is None:
            formato = detectar_formato(contents)
            msg = f"No se pudo decodificar la imagen (formato: '{formato}'). "
            if formato == 'heic':
                msg += "Instala soporte HEIC: pip install pillow-heif"
            elif formato == 'webp':
                msg += "Instala soporte WebP: pip install Pillow"
            else:
                msg += "Envía la imagen en formato JPG o PNG."
            logger.error(f"  ❌ {msg}")
            return JSONResponse(status_code=400, content={"error": msg})

        logger.info("🧠 Procesando imagen con YOLOv8...")
        results = model(img, conf=0.15)

        placas_detectadas = []
        cajas_encontradas = 0

        for result in results:
            for box in result.boxes:
                cajas_encontradas += 1
                conf_yolo = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                h_img, w_img = img.shape[:2]
                margen = 8
                x1 = max(0, x1 - margen)
                y1 = max(0, y1 - margen)
                x2 = min(w_img, x2 + margen)
                y2 = min(h_img, y2 + margen)

                placa_roi = img[y1:y2, x1:x2]
                if placa_roi.size == 0:
                    logger.warning(f"  ⚠️ Región vacía en caja #{cajas_encontradas}, saltando.")
                    continue

                logger.info(
                    f"  📦 Caja #{cajas_encontradas} "
                    f"(conf YOLO: {conf_yolo:.2f}) → "
                    f"[{x1},{y1},{x2},{y2}] ({x2-x1}x{y2-y1}px)"
                )

                encontrados = ocr_sobre_region(placa_roi)
                placas_detectadas.extend(encontrados)

        logger.info(f"  Cajas YOLO encontradas: {cajas_encontradas}")

        # Fallback: si YOLO no detectó nada, OCR sobre imagen completa
        fallback_usado = False
        if cajas_encontradas == 0:
            logger.warning("⚠️ YOLO no detectó cajas. Intentando OCR en imagen completa...")
            encontrados = ocr_sobre_region(img)
            placas_detectadas.extend(encontrados)
            fallback_usado = True

        placas_unicas = list(dict.fromkeys(placas_detectadas))
        logger.info(f"✅ Placas detectadas: {placas_unicas}")

        return JSONResponse({
            "placas": placas_unicas,
            "cajas_yolo": cajas_encontradas,
            "fallback_usado": fallback_usado
        })

    except Exception as e:
        logger.error(f"❌ Error inesperado: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
