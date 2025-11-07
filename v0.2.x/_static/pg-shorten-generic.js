document.addEventListener("DOMContentLoaded", () => {
    const genericsToRemove = [
        // "paguro.models.vfm.vfmodel.VFrameModel",
        "U",
        "M",
    ];

    const removeSpecificGenerics = (root) => {
        root.querySelectorAll(".sig").forEach(sig => {
            const nodes = Array.from(sig.childNodes);
            for (let i = 0; i < nodes.length; i++) {
                const node = nodes[i];
                // Find the opening bracket '['
                if (node.textContent && node.textContent.trim() === "[") {
                    // Collect nodes until the matching closing bracket
                    const segment = [];
                    let j = i;
                    while (j < nodes.length) {
                        segment.push(nodes[j]);
                        if (
                            nodes[j].nodeType === Node.ELEMENT_NODE &&
                            nodes[j].classList.contains("sig-paren") &&
                            nodes[j].textContent.trim() === "]"
                        ) {
                            break;
                        }
                        j++;
                    }

                    const segmentHTML = segment.map(n => n.outerHTML || n.textContent).join("");
                    // If this bracketed segment contains one of our target generics → remove it
                    if (genericsToRemove.some(g => segmentHTML.includes(g))) {
                        segment.forEach(n => n.remove());
                        // Remove trailing whitespace / newline nodes
                        while (
                            nodes[i] &&
                            nodes[i].nodeType === Node.TEXT_NODE &&
                            nodes[i].textContent.match(/^\s*$/)
                            ) {
                            nodes[i].remove();
                        }
                    }
                }
            }

            // Clean stray whitespace before '('
            sig.innerHTML = sig.innerHTML.replace(/(?:\s|&nbsp;)*\(/g, "(");
        });
    };

    // Run initially
    removeSpecificGenerics(document.body);

    // Watch for SPA-style updates
    const observer = new MutationObserver((mutations) => {
        observer.disconnect();
        for (const m of mutations) {
            for (const n of m.addedNodes) {
                if (n.nodeType === Node.ELEMENT_NODE) {
                    removeSpecificGenerics(n);
                }
            }
        }
        observer.observe(document.body, {childList: true, subtree: true});
    });
    observer.observe(document.body, {childList: true, subtree: true});
});


document.addEventListener("DOMContentLoaded", () => {
    // Find the signature for paguro.vcol
    document.querySelectorAll("dl.py.class.objdesc dt.sig.sig-object.highlight.py").forEach(dt => {
        const prefix = dt.querySelector(".sig-prename.descclassname .pre");
        const name = dt.querySelector(".sig-name.descname .pre");

        if (prefix && name &&
            prefix.textContent.trim() === "paguro." &&
            name.textContent.trim() === "vcol") {

            // Find a text node or <em class="property">class</em> at the end
            const em = dt.querySelector("em.property, .sig-inline");
            if (em && /class/.test(em.textContent.trim())) {
                em.remove();   // clean removal
            } else if (dt.textContent.trim().endswith("class")) {
                // fallback: if theme inserts bare text
                dt.innerHTML = dt.innerHTML.replace(/class\s*$/, "");
            }
        }
    });
});
// // -----------------------------------------------------------------
// // -----------------------------------------------------------------
// // -----------------------------------------------------------------

