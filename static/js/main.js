// =============================================================================
// Main JavaScript — Common Utilities
// =============================================================================
// Auto-dismiss flash messages, confirmation dialogs, and helpers.
// =============================================================================

/**
 * Read the per-session CSRF token rendered by base.html.
 *
 * Every state-changing fetch() call must send this header, otherwise Flask-WTF
 * rejects the request with HTTP 400. The token is session-scoped, so it is
 * rotated automatically by Flask-WTF as needed.
 */
function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute('content') : '';
}

/**
 * Build the standard JSON POST headers, including the CSRF token.
 */
function jsonPostHeaders(extra) {
    return Object.assign({
        'Content-Type': 'application/json',
        'X-CSRFToken': getCsrfToken()
    }, extra || {});
}

document.addEventListener('DOMContentLoaded', function() {
    // Auto-dismiss flash messages after 5 seconds
    const flashMessages = document.querySelectorAll('.flash-message');
    flashMessages.forEach(function(msg) {
        setTimeout(function() {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(msg);
            if (bsAlert) {
                bsAlert.close();
            }
        }, 5000);
    });
});
