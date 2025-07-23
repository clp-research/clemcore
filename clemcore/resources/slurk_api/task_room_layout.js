// Helper to make the javascript debuggable in web tools
function injectDebuggableScript(code, name) {
  const script = document.createElement("script");
  script.type = "text/javascript";
  script.textContent = `${code}\n//# sourceURL=${name}`;
  document.documentElement.appendChild(script);
}

window.show_toast = function(text) {
    const toast = document.createElement("div");
    toast.textContent = text;
    toast.style.position = "fixed";
    toast.style.top = "30px";
    toast.style.left = "50%";
    toast.style.transform = "translateX(-50%)";
    toast.style.background = "rgba(0, 0, 0, 0.8)";
    toast.style.color = "white";
    toast.style.padding = "12px 20px";
    toast.style.borderRadius = "8px";
    toast.style.fontSize = "16px";
    toast.style.zIndex = "9999";
    toast.style.transition = "opacity 0.5s ease";
    toast.style.opacity = "1";
    document.body.appendChild(toast);

    // Fade out and remove
    setTimeout(() => {
        toast.style.opacity = "0";
        setTimeout(() => {
            toast.remove();
        }, 500); // Wait for fade-out to complete
    }, 3000); // Show for 3 seconds
}

// Clear history when episode is done
injectDebuggableScript(`
    socket.on("command", (data) => {
        if (data.command === "done") {
            $("#chat-area").empty();
            window.show_toast("Episode done! You have been connected to a new player to play the next episode.")
        }
    });
`, name="on-done.js");


// Hide the sidebar
$('#sidebar').hide();

// Make sure the body and html allow full page scrolling
$('html, body').css({
    'overflow': 'auto',
    'height': 'auto',
    'margin': '0',
    'padding': '0',
    'scroll-behavior': 'smooth'
});

// Center the content properly
content = $('#content')
content.css({
    'width': '800px',
    'max-width': '90%',
    'margin': '0 auto',           // This should center it
    'padding-top': '80px',
    'padding-bottom': '120px',
    'box-sizing': 'border-box',
    // Make sure no positioning properties are interfering
    'position': 'static',         // Explicitly set to static
    'left': 'auto',
    'right': 'auto',
    'transform': 'none'
});

// Also check if there's a container wrapper that might be interfering
// If your content is inside another container, make sure it's also centered:
content.parent().css({
    'width': '100%',
    'text-align': 'center'  // This can help center the child
});

// Then reset text alignment for the content itself
content.css({
    'text-align': 'left'  // Reset text alignment
});

// Keep input fixed and centered
$('#text').css({
    "position": "fixed",
    "bottom": "20px",
    "left": "50%",
    "transform": "translateX(-50%)",
    "width": "800px",
    "max-width": "90%",
    "padding": "10px",
    "font-size": "16px",
    "box-sizing": "border-box",
    "z-index": "9999"
});

function scrollToBottom() {
    window.scrollTo({
        top: document.body.scrollHeight,
        behavior: 'smooth'
    });
}

// Make sure that newlines etc. are displayed properly
const observer = new MutationObserver(mutations => {
    mutations.forEach(mutation => {
        mutation.addedNodes.forEach(node => {
            // Check if it's a message element, or contains one
            if (node.nodeType === 1) {
                const messages = node.matches('.message')
                    ? [node]
                    : node.querySelectorAll?.('.message') || [];

                messages.forEach(msg => {
                    msg.style.whiteSpace = 'pre-wrap';
                });
            }
        });
    });
});

observer.observe(document.querySelector('#chat-area'), {
    childList: true,
    subtree: true
});

// New observer just for scrolling
const scrollObserver = new MutationObserver(mutations => {
    mutations.forEach(mutation => {
        if (mutation.addedNodes.length > 0) {
            // Check if any added nodes are messages
            const hasNewMessages = Array.from(mutation.addedNodes).some(node => {
                if (node.nodeType === 1) {
                    return node.matches('.message') || node.querySelector('.message');
                }
                return false;
            });

            if (hasNewMessages) {
                setTimeout(() => {
                    window.scrollTo({
                        top: document.body.scrollHeight,
                        behavior: 'smooth'
                    });
                }, 100);
            }
        }
    });
});

scrollObserver.observe(document.querySelector('#chat-area'), {
    childList: true,
    subtree: true
});