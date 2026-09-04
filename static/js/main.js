// =============================================================================
// Main JavaScript — Common Utilities
// =============================================================================
// Auto-dismiss flash messages, confirmation dialogs, and helpers.
// =============================================================================

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
