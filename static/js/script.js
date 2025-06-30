// Portfolio Analyzer - Custom JavaScript
// Handles interactive features, form validation, and UI enhancements

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize form validations
    initializeFormValidations();
    
    // Initialize interactive features
    initializeInteractiveFeatures();
    
    // Initialize market data refresh
    initializeMarketDataRefresh();
    
    // Initialize accessibility features
    initializeAccessibility();
}

// Tooltip initialization
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Form validation functions
function initializeFormValidations() {
    // Trade form validation
    const addTradeForm = document.querySelector('form[action*="add_trade"]');
    if (addTradeForm) {
        setupTradeFormValidation(addTradeForm);
    }
    
    // CSV upload form validation
    const uploadForm = document.querySelector('form[action*="upload_portfolio"]');
    if (uploadForm) {
        setupUploadFormValidation(uploadForm);
    }
}

function setupTradeFormValidation(form) {
    const symbolInput = form.querySelector('#symbol');
    const quantityInput = form.querySelector('#quantity');
    const priceInput = form.querySelector('#buy_price');
    const dateInput = form.querySelector('#purchase_date');
    
    // Real-time validation
    if (symbolInput) {
        symbolInput.addEventListener('input', function() {
            validateSymbol(this);
        });
    }
    
    if (quantityInput) {
        quantityInput.addEventListener('input', function() {
            validateQuantity(this);
            updateInvestedAmount();
        });
    }
    
    if (priceInput) {
        priceInput.addEventListener('input', function() {
            validatePrice(this);
            updateInvestedAmount();
        });
    }
    
    if (dateInput) {
        // Set max date to today
        const today = new Date().toISOString().split('T')[0];
        dateInput.max = today;
        if (!dateInput.value) {
            dateInput.value = today;
        }
    }
    
    // Form submission validation
    form.addEventListener('submit', function(e) {
        if (!validateTradeForm(form)) {
            e.preventDefault();
        }
    });
}

function validateSymbol(input) {
    const value = input.value.trim().toUpperCase();
    const isValid = value.length >= 1 && /^[A-Z0-9._-]+$/.test(value);
    
    updateInputValidation(input, isValid, 'Please enter a valid symbol (letters, numbers, ., _, - only)');
    return isValid;
}

function validateQuantity(input) {
    const value = parseFloat(input.value);
    const isValid = !isNaN(value) && value > 0;
    
    updateInputValidation(input, isValid, 'Quantity must be greater than 0');
    return isValid;
}

function validatePrice(input) {
    const value = parseFloat(input.value);
    const isValid = !isNaN(value) && value > 0;
    
    updateInputValidation(input, isValid, 'Price must be greater than 0');
    return isValid;
}

function updateInputValidation(input, isValid, errorMessage) {
    const feedbackElement = input.parentNode.querySelector('.invalid-feedback') || 
                           input.nextElementSibling?.classList.contains('invalid-feedback') ? 
                           input.nextElementSibling : null;
    
    if (isValid) {
        input.classList.remove('is-invalid');
        input.classList.add('is-valid');
        if (feedbackElement) {
            feedbackElement.style.display = 'none';
        }
    } else {
        input.classList.remove('is-valid');
        input.classList.add('is-invalid');
        
        if (!feedbackElement) {
            const feedback = document.createElement('div');
            feedback.className = 'invalid-feedback';
            feedback.textContent = errorMessage;
            input.parentNode.appendChild(feedback);
        } else {
            feedbackElement.textContent = errorMessage;
            feedbackElement.style.display = 'block';
        }
    }
}

function updateInvestedAmount() {
    const quantityInput = document.getElementById('quantity');
    const priceInput = document.getElementById('buy_price');
    
    if (quantityInput && priceInput) {
        const quantity = parseFloat(quantityInput.value) || 0;
        const price = parseFloat(priceInput.value) || 0;
        const invested = quantity * price;
        
        // Update any invested amount display
        const investedDisplay = document.getElementById('invested-amount');
        if (investedDisplay) {
            investedDisplay.textContent = `₹${invested.toFixed(2)}`;
        }
    }
}

function validateTradeForm(form) {
    const symbol = form.querySelector('#symbol');
    const quantity = form.querySelector('#quantity');
    const price = form.querySelector('#buy_price');
    const assetType = form.querySelector('#asset_type');
    const date = form.querySelector('#purchase_date');
    
    let isValid = true;
    
    if (!validateSymbol(symbol)) isValid = false;
    if (!validateQuantity(quantity)) isValid = false;
    if (!validatePrice(price)) isValid = false;
    
    if (!assetType.value) {
        updateInputValidation(assetType, false, 'Please select an asset type');
        isValid = false;
    }
    
    if (!date.value) {
        updateInputValidation(date, false, 'Please select a purchase date');
        isValid = false;
    }
    
    return isValid;
}

function setupUploadFormValidation(form) {
    const fileInput = form.querySelector('#file');
    const uploadBtn = form.querySelector('#uploadBtn');
    
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            validateFileUpload(this, uploadBtn);
        });
    }
}

function validateFileUpload(input, uploadBtn) {
    const file = input.files[0];
    const preview = document.getElementById('uploadPreview');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    
    if (!file) {
        if (uploadBtn) uploadBtn.disabled = true;
        if (preview) preview.style.display = 'none';
        return false;
    }
    
    // Validate file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showAlert('Please select a CSV file.', 'error');
        input.value = '';
        if (uploadBtn) uploadBtn.disabled = true;
        if (preview) preview.style.display = 'none';
        return false;
    }
    
    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showAlert('File size must be less than 16MB.', 'error');
        input.value = '';
        if (uploadBtn) uploadBtn.disabled = true;
        if (preview) preview.style.display = 'none';
        return false;
    }
    
    // Show file preview
    if (fileName) fileName.textContent = file.name;
    if (fileSize) fileSize.textContent = `Size: ${(file.size / 1024).toFixed(1)} KB`;
    if (preview) preview.style.display = 'block';
    if (uploadBtn) uploadBtn.disabled = false;
    
    return true;
}

// Interactive features
function initializeInteractiveFeatures() {
    // Smooth scrolling for anchor links
    initializeSmoothScrolling();
    
    // Card hover effects
    initializeCardEffects();
    
    // Number formatting
    initializeNumberFormatting();
    
    // Search and filter functionality
    initializeSearchFilter();
    
    // Theme switching (if needed)
    initializeThemeSwitch();
}

function initializeSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

function initializeCardEffects() {
    const cards = document.querySelectorAll('.card');
    
    cards.forEach(card => {
        // Add hover effects
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
            this.style.transition = 'transform 0.3s ease';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
    
    // Special effects for action cards
    const actionCards = document.querySelectorAll('.action-card');
    actionCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 10px 25px rgba(0, 0, 0, 0.15)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '';
        });
    });
}

function initializeNumberFormatting() {
    // Format currency numbers
    const currencyElements = document.querySelectorAll('[data-currency]');
    currencyElements.forEach(element => {
        const value = parseFloat(element.textContent.replace(/[₹,]/g, ''));
        if (!isNaN(value)) {
            element.textContent = formatCurrency(value);
        }
    });
    
    // Format percentage numbers
    const percentageElements = document.querySelectorAll('[data-percentage]');
    percentageElements.forEach(element => {
        const value = parseFloat(element.textContent.replace(/%/g, ''));
        if (!isNaN(value)) {
            element.textContent = formatPercentage(value);
        }
    });
}

function formatCurrency(amount, currency = '₹') {
    return currency + new Intl.NumberFormat('en-IN', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(amount);
}

function formatPercentage(value) {
    return (value >= 0 ? '+' : '') + value.toFixed(2) + '%';
}

function initializeSearchFilter() {
    const searchInputs = document.querySelectorAll('[data-search-target]');
    
    searchInputs.forEach(input => {
        input.addEventListener('input', function() {
            const target = this.getAttribute('data-search-target');
            const searchTerm = this.value.toLowerCase();
            const items = document.querySelectorAll(target);
            
            items.forEach(item => {
                const text = item.textContent.toLowerCase();
                if (text.includes(searchTerm)) {
                    item.style.display = '';
                } else {
                    item.style.display = 'none';
                }
            });
        });
    });
}

function initializeThemeSwitch() {
    const themeSwitch = document.getElementById('theme-switch');
    if (themeSwitch) {
        themeSwitch.addEventListener('change', function() {
            if (this.checked) {
                document.body.classList.add('dark-theme');
                localStorage.setItem('theme', 'dark');
            } else {
                document.body.classList.remove('dark-theme');
                localStorage.setItem('theme', 'light');
            }
        });
        
        // Load saved theme
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            themeSwitch.checked = true;
            document.body.classList.add('dark-theme');
        }
    }
}

// Market data refresh
function initializeMarketDataRefresh() {
    // Auto-refresh market data every 5 minutes on home page
    if (window.location.pathname === '/' || window.location.pathname.includes('home')) {
        setInterval(refreshMarketData, 300000); // 5 minutes
    }
    
    // Auto-refresh portfolio report every 10 minutes
    if (window.location.pathname.includes('report')) {
        setInterval(refreshPortfolioData, 600000); // 10 minutes
    }
}

function refreshMarketData() {
    // Only refresh if page is visible
    if (!document.hidden) {
        const marketCards = document.querySelectorAll('.market-card');
        if (marketCards.length > 0) {
            // Add loading state
            marketCards.forEach(card => {
                card.classList.add('loading');
            });
            
            // Refresh the page to get new data
            setTimeout(() => {
                window.location.reload();
            }, 1000);
        }
    }
}

function refreshPortfolioData() {
    if (!document.hidden && window.location.pathname.includes('report')) {
        showAlert('Refreshing portfolio data...', 'info');
        setTimeout(() => {
            window.location.reload();
        }, 2000);
    }
}

// Accessibility features
function initializeAccessibility() {
    // Keyboard navigation for cards
    const clickableCards = document.querySelectorAll('.action-card, [data-clickable]');
    
    clickableCards.forEach(card => {
        card.setAttribute('tabindex', '0');
        card.setAttribute('role', 'button');
        
        card.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                const link = this.querySelector('a');
                if (link) {
                    link.click();
                } else {
                    this.click();
                }
            }
        });
    });
    
    // Skip navigation link
    addSkipNavigation();
    
    // Focus management for modals and dropdowns
    initializeFocusManagement();
}

function addSkipNavigation() {
    const skipNav = document.createElement('a');
    skipNav.href = '#main-content';
    skipNav.className = 'skip-nav';
    skipNav.textContent = 'Skip to main content';
    skipNav.style.cssText = `
        position: absolute;
        top: -40px;
        left: 6px;
        background: #000;
        color: #fff;
        padding: 8px;
        text-decoration: none;
        border-radius: 4px;
        z-index: 1000;
    `;
    
    skipNav.addEventListener('focus', function() {
        this.style.top = '6px';
    });
    
    skipNav.addEventListener('blur', function() {
        this.style.top = '-40px';
    });
    
    document.body.insertBefore(skipNav, document.body.firstChild);
    
    // Add main content landmark
    const main = document.querySelector('main');
    if (main && !main.id) {
        main.id = 'main-content';
    }
}

function initializeFocusManagement() {
    // Trap focus in modals
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        modal.addEventListener('shown.bs.modal', function() {
            const focusableElements = this.querySelectorAll(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );
            if (focusableElements.length > 0) {
                focusableElements[0].focus();
            }
        });
    });
}

// Utility functions
function showAlert(message, type = 'info', duration = 5000) {
    const alertContainer = document.querySelector('.container') || document.body;
    const alert = document.createElement('div');
    alert.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    alertContainer.insertBefore(alert, alertContainer.firstChild);
    
    // Auto-dismiss after duration
    setTimeout(() => {
        if (alert.parentNode) {
            alert.remove();
        }
    }, duration);
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

// Performance optimization
function optimizeImages() {
    const images = document.querySelectorAll('img[data-src]');
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.removeAttribute('data-src');
                imageObserver.unobserve(img);
            }
        });
    });
    
    images.forEach(img => imageObserver.observe(img));
}

// Error handling
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
    // Could send error to analytics or logging service
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    // Could send error to analytics or logging service
});

// Export functions for use in other scripts
window.PortfolioAnalyzer = {
    formatCurrency,
    formatPercentage,
    showAlert,
    validateSymbol,
    validateQuantity,
    validatePrice,
    debounce,
    throttle
};

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}
