"""API Flask y página local para clasificar una imagen a la vez."""

from pathlib import Path
import sys

# Permite arrancar con python src/web/app.py desde cualquier directorio.
ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from src.web.inference import MODELS, InvalidImageError, get_predictor


HOST = "127.0.0.1"
PORT = 8000
PAGE_ROUTE = "/"
PREDICT_ROUTE = "/api/predict"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def create_app():
    """Configura Flask; cada modelo se carga al recibir su primera predicción.

    Returns:
        Flask: Aplicación con la interfaz y el endpoint POST /api/predict.

    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 11 * 1024 * 1024
    app.json.ensure_ascii = False
    @app.get(PAGE_ROUTE)
    def index():
        return render_template("index.html", models=MODELS)

    @app.post(PREDICT_ROUTE)
    def predict():
        model_id = request.form.get("model", "efficientnet")
        if model_id not in MODELS:
            return jsonify(error="Selecciona un modelo válido."), 400
        files = request.files.getlist("file")
        if len(files) != 1 or not files[0].filename or set(request.files) != {"file"}:
            return jsonify(error="Selecciona una sola imagen en el campo file."), 400
        upload = files[0]
        try:
            if Path(upload.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
                return jsonify(error="Usa una imagen JPEG, PNG, WebP o BMP."), 415
            content = upload.read(MAX_IMAGE_BYTES + 1)
            if len(content) > MAX_IMAGE_BYTES:
                return jsonify(error="La imagen supera el límite de 10 MiB."), 413
            if not content:
                return jsonify(error="El archivo está vacío. Selecciona otra imagen."), 400
            predictor = get_predictor(model_id)
            return jsonify(model=model_id, top_three=predictor.predict(content))
        except InvalidImageError as error:
            return jsonify(error=str(error)), 400
        except FileNotFoundError as error:
            return jsonify(error=str(error)), 503
        finally:
            upload.close()

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(error):
        return jsonify(error="La solicitud es demasiado grande. Usa una imagen de hasta 10 MiB."), 413

    @app.errorhandler(Exception)
    def unexpected_error(error):
        if isinstance(error, HTTPException):
            if request.path.startswith("/api/"):
                return jsonify(error=error.description), error.code
            return error
        app.logger.exception("Error al procesar la solicitud")
        return jsonify(error="No se pudo generar la predicción. Intenta de nuevo."), 500

    return app


if __name__ == "__main__":
    create_app().run(host=HOST, port=PORT, debug=False, use_reloader=False)
