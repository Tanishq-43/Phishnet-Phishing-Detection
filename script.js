// URL Detection Form Handler
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('detectionForm');
    const results = document.getElementById('results');
    const loading = document.getElementById('loading');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultTitle = document.getElementById('resultTitle');
    const resultIcon = document.getElementById('resultIcon');
    const resultMessage = document.getElementById('resultMessage');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');
    const safeProb = document.getElementById('safeProb');
    const phishingProb = document.getElementById('phishingProb');
    const suspicionFactors = document.getElementById('suspicionFactors');
    const factorsList = document.getElementById('factorsList');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const urlInput = document.getElementById('urlInput');
        const url = urlInput.value.trim();
        
        if (!url) {
            alert('Please enter a URL');
            return;
        }

        // Show loading state
        loading.style.display = 'block';
        results.style.display = 'none';
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        
        try {
            const response = await fetch('/detect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `url=${encodeURIComponent(url)}`
            });
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            displayResults(data);
            
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            loading.style.display = 'none';
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze URL';
        }
    });

    function displayResults(data) {
        const isPhishing = data.is_phishing;
        const confidence = data.confidence * 100;
        
        // Update results based on classification
        if (isPhishing) {
            resultTitle.textContent = '🚨 Phishing Website Detected!';
            resultTitle.className = 'text-danger';
            resultIcon.innerHTML = '<i class="fas fa-exclamation-triangle text-danger"></i>';
            resultMessage.textContent = 'This website appears to be a phishing attempt. Do not enter any personal information.';
            results.querySelector('.card').classList.add('phishing-result');
            results.querySelector('.card').classList.remove('safe-result');
        } else {
            resultTitle.textContent = '✅ Safe Website';
            resultTitle.className = 'text-success';
            resultIcon.innerHTML = '<i class="fas fa-check-circle text-success"></i>';
            resultMessage.textContent = 'This website appears to be safe. However, always exercise caution online.';
            results.querySelector('.card').classList.add('safe-result');
            results.querySelector('.card').classList.remove('phishing-result');
        }
        
        // Update confidence bar
        confidenceBar.style.width = confidence + '%';
        confidenceBar.textContent = Math.round(confidence) + '%';
        confidenceBar.className = 'progress-bar ' + (isPhishing ? 'bg-danger' : 'bg-success');
        
        confidenceText.textContent = `Confidence: ${Math.round(confidence)}%`;
        
        // Update probabilities
        safeProb.textContent = Math.round(data.safe_probability * 100) + '%';
        phishingProb.textContent = Math.round(data.phishing_probability * 100) + '%';
        
        // Show suspicion factors if available
        if (data.suspicion_factors && data.suspicion_factors.length > 0) {
            factorsList.innerHTML = '';
            data.suspicion_factors.forEach(factor => {
                const badge = document.createElement('span');
                badge.className = 'badge bg-warning suspicion-badge me-1';
                badge.textContent = factor;
                factorsList.appendChild(badge);
            });
            suspicionFactors.style.display = 'block';
        } else {
            suspicionFactors.style.display = 'none';
        }
        
        // Show results
        results.style.display = 'block';
        
        // Scroll to results
        results.scrollIntoView({ behavior: 'smooth' });
    }

    // Add some sample URLs for quick testing
    // Update the sample URLs section in the JavaScript
    const sampleUrls = [
        "google.com",
        "www.github.com",
        "http://paypal-security-verify.com", 
        "https://facebook-login-secure.com",
        "192.168.1.1/login",
        "bit.ly/suspicious-link",
        "amazon-account-update.xyz",
        "microsoft-verify.tk"
    ];

    // Optional: Add quick test buttons
    const urlInput = document.getElementById('urlInput');
    const quickTestDiv = document.createElement('div');
    quickTestDiv.className = 'mt-3';
    quickTestDiv.innerHTML = `
        <small class="text-muted">Quick test:</small>
        <div class="btn-group btn-group-sm mt-1" role="group">
            ${sampleUrls.map(url => 
                `<button type="button" class="btn btn-outline-secondary" onclick="document.getElementById('urlInput').value='${url}'">${new URL(url).hostname}</button>`
            ).join('')}
        </div>
    
    `;
    
    
    form.appendChild(quickTestDiv);
    
});