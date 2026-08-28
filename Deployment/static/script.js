// Wires the Stitch-designed BCI UI to the Flask backend.
// Handles: file selection (click or drag-drop) with visible feedback,
// calling /predict and /example, and toggling between idle / loading /
// success / rejected states.

const dropzone = document.getElementById('upload-dropzone');
const dropzoneText = dropzone.querySelector('p'); // "Drag and drop..." text, swapped for filename once chosen
const fileInput = document.getElementById('csv-file-input');
const predictButton = document.getElementById('predict-button');
const exampleButton = document.getElementById('example-button');

const idleState = document.getElementById('result-idle');
const loadingState = document.getElementById('loading-state');
const resultSuccess = document.getElementById('result-success');
const resultRejected = document.getElementById('result-rejected');

const predictionIcon = document.getElementById('prediction-icon');
const predictionText = document.getElementById('prediction-text');
const confidenceValue = document.getElementById('confidence-value');
const confidenceProgress = document.getElementById('confidence-progress');
const rejectionReason = document.getElementById('rejection-reason');

let selectedFile = null;

// --- Dropzone: click to browse (ignored if the click came from Predict button) ---
dropzone.addEventListener('click', (e) => {
  if (e.target === predictButton || predictButton.contains(e.target)) return;
  fileInput.click();
});

// --- Dropzone: drag and drop ---
dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('border-tertiary');
});
dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('border-tertiary');
});
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('border-tertiary');
  if (e.dataTransfer.files.length) {
    setSelectedFile(e.dataTransfer.files[0]);
  }
});

// --- File picked via browse dialog ---
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) {
    setSelectedFile(fileInput.files[0]);
  }
});

// --- Show the chosen filename so it's obvious a file was picked ---
function setSelectedFile(file) {
  selectedFile = file;
  dropzoneText.textContent = `Selected: ${file.name}`;
}

// --- Show only one of: idle / loading / success / rejected ---
function showState(state) {
  idleState.classList.add('hidden');
  loadingState.classList.add('hidden');
  resultSuccess.classList.add('hidden');
  resultRejected.classList.add('hidden');

  if (state === 'idle') idleState.classList.remove('hidden');
  if (state === 'loading') loadingState.classList.remove('hidden');
  if (state === 'success') resultSuccess.classList.remove('hidden');
  if (state === 'rejected') resultRejected.classList.remove('hidden');
}

// --- Populate the success card ---
function renderSuccess(data) {
  const pct = Math.round(data.confidence * 100);
  predictionText.textContent = data.prediction.toUpperCase();
  confidenceValue.textContent = `${pct}%`;
  confidenceProgress.style.width = `${pct}%`;

  // Material Symbols renders the icon based on its text content ("east"/"west" ligatures) —
  // point the arrow right for RIGHT, left for LEFT.
  const iconName = data.prediction.toLowerCase() === 'left' ? 'west' : 'east';
  predictionIcon.textContent = iconName;
  predictionIcon.setAttribute('data-icon', iconName);

  showState('success');
}

// --- Populate the rejected card ---
function renderRejected(data) {
  rejectionReason.textContent = data.rejection_reasons;
  showState('rejected');
}

// --- Predict button: upload the selected file ---
predictButton.addEventListener('click', async (e) => {
  e.stopPropagation(); // don't let the click bubble to the dropzone's browse handler
  if (!selectedFile) {
    alert('Please choose a CSV file first (click or drag one into the box).');
    return;
  }

  const formData = new FormData();
  formData.append('file', selectedFile);

  showState('loading');
  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) {
      alert(data.error);
      showState('idle');
      return;
    }
    if (data.is_valid) {
      renderSuccess(data);
    } else {
      renderRejected(data);
    }
  } catch (err) {
    alert('Request failed: ' + err.message);
    showState('idle');
  }
});

// --- Example button: run the pre-loaded clean trial ---
exampleButton.addEventListener('click', async () => {
  showState('loading');
  try {
    const res = await fetch('/example');
    const data = await res.json();
    renderSuccess(data); // /example is always a clean trial
  } catch (err) {
    alert('Request failed: ' + err.message);
    showState('idle');
  }
});