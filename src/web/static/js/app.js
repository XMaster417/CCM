"use strict";

const form = document.getElementById("prediction-form");
const input = document.getElementById("image-input");
const modelSelect = document.getElementById("model-select");
const zone = document.getElementById("drop-zone");
const preview = document.getElementById("image-preview");
const placeholder = document.getElementById("upload-placeholder");
const changeImage = document.getElementById("change-image");
const fileDetails = document.getElementById("file-details");
const button = document.getElementById("predict-button");
const statusMessage = document.getElementById("status-message");
const errorMessage = document.getElementById("error-message");
const resultsPanel = document.getElementById("results-panel");
const resultsPlaceholder = document.getElementById("results-placeholder");
const resultsContent = document.getElementById("results-content");
const predictionList = document.getElementById("prediction-list");
const maxImageBytes = 10 * 1024 * 1024;
const probabilityFormatter = new Intl.NumberFormat("es-MX", {
  style: "percent", minimumFractionDigits: 2, maximumFractionDigits: 2,
});
let selectedFile = null;
let previewUrl = null;
let busy = false;

function clearResults() {
  resultsContent.hidden = true;
  resultsPlaceholder.hidden = false;
  predictionList.replaceChildren();
  document.getElementById("winning-class").textContent = "";
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
  statusMessage.textContent = "Selecciona otra imagen o vuelve a intentarlo.";
}

function resetSelection() {
  selectedFile = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  preview.onload = null;
  preview.onerror = null;
  preview.removeAttribute("src");
  preview.hidden = true;
  placeholder.hidden = false;
  changeImage.hidden = true;
  fileDetails.hidden = true;
  button.disabled = true;
  input.value = "";
}

function selectFiles(files) {
  if (busy) return;
  clearResults();
  resetSelection();
  errorMessage.hidden = true;
  if (files.length !== 1) {
    showError("Selecciona una sola imagen a la vez.");
    return;
  }
  const file = files[0];
  if (!/\.(jpe?g|png|webp|bmp)$/i.test(file.name)) {
    showError("Usa una imagen JPEG, PNG, WebP o BMP.");
    return;
  }
  if (!file.size || file.size > maxImageBytes) {
    showError(file.size ? "La imagen supera el límite de 10 MiB." : "El archivo está vacío.");
    return;
  }
  previewUrl = URL.createObjectURL(file);
  const currentUrl = previewUrl;
  statusMessage.textContent = "Preparando la vista previa…";
  preview.onload = () => {
    if (previewUrl !== currentUrl) return;
    selectedFile = file;
    placeholder.hidden = true;
    preview.hidden = false;
    changeImage.hidden = false;
    fileDetails.hidden = false;
    document.getElementById("file-name").textContent = file.name;
    document.getElementById("file-size").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MiB`;
    button.disabled = false;
    statusMessage.textContent = "Imagen lista. Pulsa Predecir para continuar.";
  };
  preview.onerror = () => {
    if (previewUrl !== currentUrl) return;
    resetSelection();
    showError("No se pudo leer la imagen. Selecciona un archivo válido.");
  };
  preview.src = previewUrl;
}

function setBusy(value) {
  busy = value;
  input.disabled = value;
  modelSelect.disabled = value;
  button.disabled = value || !selectedFile;
  button.classList.toggle("processing", value);
  zone.setAttribute("aria-disabled", String(value));
  zone.tabIndex = value ? -1 : 0;
  resultsPanel.setAttribute("aria-busy", String(value));
  document.getElementById("button-label").textContent = value ? "Analizando imagen…" : "Predecir";
  document.getElementById("button-arrow").hidden = value;
  document.getElementById("button-spinner").hidden = !value;
}

function renderPredictions(predictions) {
  if (!Array.isArray(predictions) || predictions.length !== 3 || predictions.some(
    (item) => typeof item.class_name !== "string" || !Number.isFinite(item.probability)
      || item.probability < 0 || item.probability > 1,
  )) throw new Error("El servidor devolvió una respuesta inválida. Intenta de nuevo.");

  predictionList.replaceChildren();
  predictions.forEach((prediction, index) => {
    const item = document.createElement("li");
    item.className = "prediction-item";
    const heading = document.createElement("div");
    heading.className = "prediction-heading";
    const rank = document.createElement("span");
    rank.className = "prediction-rank";
    rank.textContent = `0${index + 1}`;
    const name = document.createElement("span");
    name.className = "prediction-name";
    name.textContent = prediction.class_name.replaceAll("_", " ");
    const probability = document.createElement("span");
    probability.className = "prediction-probability";
    probability.textContent = probabilityFormatter.format(prediction.probability);
    heading.append(rank, name, probability);
    const track = document.createElement("div");
    track.className = "probability-track";
    track.setAttribute("aria-hidden", "true");
    const fill = document.createElement("span");
    fill.className = "probability-fill";
    fill.style.width = `${prediction.probability * 100}%`;
    track.append(fill);
    item.append(heading, track);
    predictionList.append(item);
  });
  document.getElementById("winning-class").textContent = predictions[0].class_name.replaceAll("_", " ");
  resultsPlaceholder.hidden = true;
  resultsContent.hidden = false;
  statusMessage.textContent = `Predicción lista. Clase más probable: ${predictions[0].class_name.replaceAll("_", " ")}, ${probabilityFormatter.format(predictions[0].probability)}.`;
}

modelSelect.addEventListener("change", () => {
  clearResults();
  errorMessage.hidden = true;
  statusMessage.textContent = selectedFile
    ? "Modelo cambiado. Pulsa Predecir para analizar la imagen."
    : "Selecciona una imagen para comenzar.";
});

input.addEventListener("change", () => {
  // Copia la lista antes de limpiar el input para permitir seleccionar la misma imagen.
  const files = Array.from(input.files);
  if (files.length) selectFiles(files);
});
zone.addEventListener("click", (event) => { if (busy) event.preventDefault(); });
zone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    if (!busy) input.click();
  }
});
// Evita que el navegador abandone la página al soltar un archivo fuera del recuadro.
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());
zone.addEventListener("dragover", (event) => {
  event.preventDefault();
  if (!busy) zone.classList.add("dragging");
});
zone.addEventListener("dragleave", (event) => {
  if (!zone.contains(event.relatedTarget)) zone.classList.remove("dragging");
});
zone.addEventListener("drop", (event) => {
  event.preventDefault();
  zone.classList.remove("dragging");
  if (!busy) selectFiles(Array.from(event.dataTransfer.files));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile || busy) return;
  clearResults();
  errorMessage.hidden = true;
  setBusy(true);
  statusMessage.textContent = `${modelSelect.selectedOptions[0].textContent} está analizando tu imagen…`;
  const payload = new FormData();
  payload.append("file", selectedFile);
  payload.append("model", modelSelect.value);
  try {
    let response;
    try {
      response = await fetch(form.action, { method: "POST", body: payload });
    } catch {
      throw new Error("No se pudo conectar con el servidor. Comprueba que Flask siga ejecutándose e intenta de nuevo.");
    }
    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error(response.status === 413 ? "La imagen supera el tamaño permitido." : "No se pudo leer la respuesta del servidor. Intenta de nuevo.");
    }
    if (!response.ok) throw new Error(data.error || "No se pudo generar la predicción.");
    renderPredictions(data.top_three);
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
});

window.addEventListener("beforeunload", () => { if (previewUrl) URL.revokeObjectURL(previewUrl); });
