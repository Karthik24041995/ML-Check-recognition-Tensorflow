// Check Amount Recognition - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const uploadBox = document.getElementById('uploadBox');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const filePreview = document.getElementById('filePreview');
    const previewImage = document.getElementById('previewImage');
    const fileName = document.getElementById('fileName');
    const changeFileBtn = document.getElementById('changeFileBtn');
    const processBtn = document.getElementById('processBtn');
    const loading = document.getElementById('loading');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    const errorMessage = document.getElementById('errorMessage');
    const newCheckBtn = document.getElementById('newCheckBtn');
    const retryBtn = document.getElementById('retryBtn');

    let selectedFile = null;

    // Event Listeners
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    changeFileBtn.addEventListener('click', resetUpload);
    processBtn.addEventListener('click', processCheck);
    newCheckBtn.addEventListener('click', resetUpload);
    retryBtn.addEventListener('click', resetUpload);

    // Drag and Drop
    uploadBox.addEventListener('dragover', handleDragOver);
    uploadBox.addEventListener('dragleave', handleDragLeave);
    uploadBox.addEventListener('drop', handleDrop);
    uploadBox.addEventListener('click', () => fileInput.click());

    // Functions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            displayFile(file);
        }
    }

    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadBox.classList.add('dragover');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadBox.classList.remove('dragover');
    }

    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadBox.classList.remove('dragover');

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            displayFile(file);
        } else {
            showError('Please drop a valid image file');
        }
    }

    function displayFile(file) {
        selectedFile = file;
        
        // Read and display image
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            fileName.textContent = file.name;
            
            // Show preview, hide upload box
            uploadBox.style.display = 'none';
            filePreview.style.display = 'block';
            hideError();
            hideResults();
        };
        reader.readAsDataURL(file);
    }

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        uploadBox.style.display = 'block';
        filePreview.style.display = 'none';
        hideLoading();
        hideResults();
        hideError();
    }

    async function processCheck() {
        if (!selectedFile) {
            showError('Please select a file first');
            return;
        }

        // Show loading
        showLoading();
        hideResults();
        hideError();

        // Prepare form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            // Send to server
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            hideLoading();

            if (result.success) {
                displayResults(result);
            } else {
                showError(result.error || 'An error occurred during processing');
            }
        } catch (error) {
            hideLoading();
            showError('Network error: ' + error.message);
        }
    }

    function displayResults(result) {
        // Show results section
        resultsSection.style.display = 'block';
        filePreview.style.display = 'none';

        // Display amount
        const validation = result.validation;
        const amountCard = document.getElementById('amountCard');
        const amountValue = document.getElementById('amountValue');
        const confidenceValue = document.getElementById('confidenceValue');
        const progressFill = document.getElementById('progressFill');

        if (validation.is_valid) {
            amountCard.classList.remove('invalid');
            amountValue.textContent = validation.amount_formatted;
        } else {
            amountCard.classList.add('invalid');
            amountValue.textContent = validation.amount_formatted + ' ⚠️';
        }

        // Display confidence
        const avgConfidence = validation.confidence.average;
        confidenceValue.textContent = (avgConfidence * 100).toFixed(1) + '%';
        progressFill.style.width = (avgConfidence * 100) + '%';

        // Display validation items
        displayValidationItems(validation);

        // Display digit predictions
        displayDigits(result.predictions, result.confidences);

        // Display images
        document.getElementById('originalImage').src = result.original_image;
        document.getElementById('preprocessedImage').src = result.preprocessed_image;

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function displayValidationItems(validation) {
        const container = document.getElementById('validationItems');
        container.innerHTML = '';

        const validations = [
            {
                name: 'Confidence Check',
                valid: validation.validations.confidence_valid,
                message: `Average: ${(validation.confidence.average * 100).toFixed(1)}%`
            },
            {
                name: 'Format Check',
                valid: validation.validations.format_valid,
                message: validation.validations.format_valid ? 'Valid format' : 'Invalid format'
            },
            {
                name: 'Amount Range',
                valid: validation.validations.range_valid,
                message: validation.validations.range_valid ? 'Within limits' : 'Out of range'
            }
        ];

        validations.forEach(item => {
            const div = document.createElement('div');
            div.className = 'validation-item' + (item.valid ? '' : ' invalid');
            div.innerHTML = `
                <span class="validation-icon">${item.valid ? '✓' : '✗'}</span>
                <div class="validation-text">
                    <strong>${item.name}</strong>
                    <span>${item.message}</span>
                </div>
            `;
            container.appendChild(div);
        });

        // Show errors if any
        if (validation.errors && validation.errors.length > 0) {
            validation.errors.forEach(error => {
                const div = document.createElement('div');
                div.className = 'validation-item invalid';
                div.innerHTML = `
                    <span class="validation-icon">⚠️</span>
                    <div class="validation-text">
                        <strong>Error</strong>
                        <span>${error}</span>
                    </div>
                `;
                container.appendChild(div);
            });
        }

        // Show anomalies if any
        if (validation.anomalies && validation.anomalies.length > 0) {
            validation.anomalies.forEach(anomaly => {
                const div = document.createElement('div');
                div.className = 'validation-item warning';
                div.innerHTML = `
                    <span class="validation-icon">⚠️</span>
                    <div class="validation-text">
                        <strong>Anomaly</strong>
                        <span>${anomaly.message}</span>
                    </div>
                `;
                container.appendChild(div);
            });
        }
    }

    function displayDigits(predictions, confidences) {
        const container = document.getElementById('digitsContainer');
        container.innerHTML = '';

        predictions.forEach((digit, index) => {
            const confidence = confidences[index];
            const div = document.createElement('div');
            div.className = 'digit-item' + (confidence < 0.7 ? ' low-confidence' : '');
            div.innerHTML = `
                <div class="digit-number">${digit}</div>
                <div class="digit-confidence">${(confidence * 100).toFixed(1)}%</div>
            `;
            container.appendChild(div);
        });
    }

    function showLoading() {
        loading.style.display = 'block';
    }

    function hideLoading() {
        loading.style.display = 'none';
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorSection.style.display = 'block';
    }

    function hideError() {
        errorSection.style.display = 'none';
    }

    function hideResults() {
        resultsSection.style.display = 'none';
    }
});
