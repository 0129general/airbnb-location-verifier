document.addEventListener('DOMContentLoaded', function() {
    
    // Initialize map only if coordinates are available and not on quick results page
    // (Quick results page has its own map initialization)
    const mapContainer = document.getElementById('map');
    if (mapContainer && mapContainer.dataset.lat && mapContainer.dataset.lng) {
        const lat = parseFloat(mapContainer.dataset.lat);
        const lng = parseFloat(mapContainer.dataset.lng);
        
        if (!isNaN(lat) && !isNaN(lng)) {
            // Only initialize if not on quick results page
            if (!window.location.pathname.includes('/quick_results/')) {
                initMap(lat, lng);
            }
        }
    }
    
    // Handle form submission with loading state
    const urlForm = document.getElementById('url-form');
    if (urlForm) {
        urlForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Validate URL before submitting
            const urlInput = document.getElementById('airbnb-url');
            if (!validateUrl(urlInput.value)) {
                return false;
            }
            
            // Show loading overlay
            showLoadingOverlay();
            
            // Update loading status messages
            const statusMessages = [
                'Fetching Airbnb data...',
                'Analyzing property photos...',
                'Extracting location details...',
                'Verifying address information...'
            ];
            
            let messageIndex = 0;
            const statusInterval = setInterval(() => {
                if (messageIndex < statusMessages.length) {
                    updateLoadingStatus(statusMessages[messageIndex]);
                    messageIndex++;
                }
            }, 3000);
            
            // Submit the form
            setTimeout(() => {
                this.submit();
            }, 100);
        });
    }
    
    // Validate URL input
    const urlInput = document.getElementById('airbnb-url');
    if (urlInput) {
        urlInput.addEventListener('input', function() {
            validateUrl(this.value);
        });
        
        // Clear error message on focus
        urlInput.addEventListener('focus', function() {
            const feedback = document.getElementById('url-feedback');
            feedback.textContent = '';
            feedback.classList.remove('text-danger', 'text-success');
        });
    }
    
    // Copy address to clipboard
    const copyBtn = document.getElementById('copy-address');
    if (copyBtn) {
        copyBtn.addEventListener('click', function() {
            const address = document.getElementById('property-address').value;
            copyToClipboard(address);
            
            // Show tooltip
            this.setAttribute('data-bs-original-title', 'Copied!');
            const tooltip = bootstrap.Tooltip.getInstance(this);
            tooltip.show();
            
            // Reset tooltip after 2 seconds
            setTimeout(() => {
                this.setAttribute('data-bs-original-title', 'Copy to clipboard');
            }, 2000);
        });
        
        // Initialize tooltip
        new bootstrap.Tooltip(copyBtn);
    }
    
    // Initialize tooltips everywhere with custom HTML support
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl, {
        html: true,
        placement: tooltipTriggerEl.getAttribute('data-bs-placement') || 'top'
    }));
});

function initMap(lat, lng) {
    // Create a map centered at the property's location
    const map = L.map('map').setView([lat, lng], 15);
    
    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);
    
    // Add a marker at the property location
    const marker = L.marker([lat, lng]).addTo(map);
    
    // Add a circle to indicate the approximate area (for privacy)
    L.circle([lat, lng], {
        color: 'rgba(66, 133, 244, 0.6)',
        fillColor: 'rgba(66, 133, 244, 0.2)',
        fillOpacity: 0.5,
        radius: 200
    }).addTo(map);
    
    // Make sure the map renders correctly
    setTimeout(() => {
        map.invalidateSize();
    }, 100);
}

function validateUrl(url) {
    const submitBtn = document.getElementById('submit-btn');
    const feedback = document.getElementById('url-feedback');
    const urlInput = document.getElementById('airbnb-url');
    
    if (!url) {
        feedback.textContent = '';
        submitBtn.disabled = false;
        urlInput.classList.remove('is-valid', 'is-invalid');
        return true; // Allow empty for initial state
    }
    
    // Check for valid URL format
    try {
        const urlObj = new URL(url);
        
        // Check if it's an Airbnb domain
        const validDomains = ['airbnb.com', 'airbnb.co.uk', 'airbnb.ca', 'airbnb.com.au', 'airbnb.de', 'airbnb.fr', 'airbnb.es', 'airbnb.it'];
        const isAirbnb = validDomains.some(domain => urlObj.hostname.includes(domain));
        
        if (!isAirbnb) {
            feedback.textContent = 'Please enter a valid Airbnb URL (e.g., airbnb.com, airbnb.co.uk)';
            feedback.classList.remove('text-success');
            feedback.classList.add('text-danger');
            submitBtn.disabled = true;
            urlInput.classList.remove('is-valid');
            urlInput.classList.add('is-invalid');
            return false;
        }
        
        // Check if it's a listing URL
        if (!urlObj.pathname.includes('/rooms/')) {
            feedback.textContent = 'Please enter a specific Airbnb listing URL (should contain /rooms/)';
            feedback.classList.remove('text-success');
            feedback.classList.add('text-danger');
            submitBtn.disabled = true;
            urlInput.classList.remove('is-valid');
            urlInput.classList.add('is-invalid');
            return false;
        }
        
        // Valid URL
        feedback.textContent = 'Valid Airbnb listing URL';
        feedback.classList.remove('text-danger');
        feedback.classList.add('text-success');
        submitBtn.disabled = false;
        urlInput.classList.remove('is-invalid');
        urlInput.classList.add('is-valid');
        return true;
        
    } catch (e) {
        feedback.textContent = 'Please enter a valid URL';
        feedback.classList.remove('text-success');
        feedback.classList.add('text-danger');
        submitBtn.disabled = true;
        urlInput.classList.remove('is-valid');
        urlInput.classList.add('is-invalid');
        return false;
    }
}

function showLoadingOverlay() {
    // Create or show the loading overlay
    let overlay = document.getElementById('loadingOverlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'loadingOverlay';
        overlay.innerHTML = `
            <div class="loading-content">
                <div class="spinner-border spinner-border-lg text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div class="loading-text mt-3">
                    <h5>Processing...</h5>
                    <p class="mb-2">This may take 10-20 seconds</p>
                    <p class="loading-status text-muted">Initializing...</p>
                </div>
            </div>
        `;
        document.body.appendChild(overlay);
    }
    overlay.classList.add('show');
    
    // Disable the submit button
    const submitBtn = document.getElementById('submit-btn');
    if (submitBtn) {
        submitBtn.disabled = true;
    }
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.remove('show');
    }
    
    // Re-enable the submit button
    const submitBtn = document.getElementById('submit-btn');
    if (submitBtn) {
        submitBtn.disabled = false;
    }
}

function updateLoadingStatus(status) {
    const statusElement = document.querySelector('.loading-status');
    if (statusElement) {
        statusElement.textContent = status;
    }
}

function copyToClipboard(text) {
    // Create a temporary textarea element
    const textarea = document.createElement('textarea');
    textarea.value = text;
    document.body.appendChild(textarea);
    
    // Select and copy the text
    textarea.select();
    document.execCommand('copy');
    
    // Remove the temporary element
    document.body.removeChild(textarea);
}
