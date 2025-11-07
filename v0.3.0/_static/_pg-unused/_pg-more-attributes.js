// (function() {
//     if (window.moreAttributesInitialized) return;
//     window.moreAttributesInitialized = true;
//
//     document.addEventListener("DOMContentLoaded", function() {
//         const header = document.getElementById('more-attributes');
//         if (!header) return;
//         if (header.querySelector('.toggle-button')) return;
//
//         // --- Create toggle button ---
//         const toggle = document.createElement('span');
//         toggle.className = 'toggle-button';
//         toggle.textContent = '[+]';
//         toggle.style.cursor = 'pointer';
//         toggle.style.fontFamily = 'monospace';
//         toggle.style.marginLeft = '8px';
//
//         // --- Header layout ---
//         header.style.display = 'flex';
//         header.style.justifyContent = 'space-between';
//         header.style.alignItems = 'center';
//         header.appendChild(toggle);
//
//         // --- Collapsible container ---
//         const collapsible = document.createElement('div');
//         collapsible.className = 'collapsible-content';
//         collapsible.style.overflow = 'hidden';
//         collapsible.style.transition = 'max-height 0.3s ease';
//         collapsible.style.maxHeight = '0';
//
//         // ✅ Move all nodes *after the header* until next H1/H2/H3
//         let current = header.nextElementSibling;
//         const elementsToMove = [];
//
//         while (current) {
//             if (/^H[1-3]$/.test(current.tagName)) break;
//             elementsToMove.push(current);
//             current = current.nextElementSibling;
//         }
//
//         elementsToMove.forEach(el => collapsible.appendChild(el));
//         header.parentNode.insertBefore(collapsible, header.nextSibling);
//
//         // --- Toggle behavior ---
//         function toggleCollapse() {
//             const isOpen = collapsible.style.maxHeight !== '0px' && collapsible.style.maxHeight !== '';
//             if (isOpen) {
//                 collapsible.style.maxHeight = '0';
//                 toggle.textContent = '[+]';
//             } else {
//                 collapsible.style.maxHeight = collapsible.scrollHeight + 'px';
//                 toggle.textContent = '[-]';
//             }
//         }
//
//         header.addEventListener('click', toggleCollapse);
//
//         // --- Handle URL hash (#more-attributes) ---
//         function adjustForHash() {
//             if (window.location.hash === '#more-attributes') {
//                 collapsible.style.maxHeight = collapsible.scrollHeight + 'px';
//                 toggle.textContent = '[-]';
//             } else {
//                 collapsible.style.maxHeight = '0';
//                 toggle.textContent = '[+]';
//             }
//         }
//
//         adjustForHash();
//         window.addEventListener('hashchange', adjustForHash);
//     });
// })();
//
// //
// // // document.addEventListener("DOMContentLoaded", function() {
// // //     // Select the specified h3 element
// // //     var header = document.getElementById('more-attributes');
// // //     var toggle = document.createElement('span');
// // //     toggle.style.float = 'left'; // align the toggle to the right
// // //     toggle.style.cursor = 'pointer'; // make it look clickable
// // //     header.appendChild(toggle);
// // //
// // //     var collapsibleContainer = document.createElement('div');
// // //     collapsibleContainer.className = 'collapsible-content';
// // //     var currentNode = header.nextElementSibling;
// // //
// // //     // Move all nodes to the new collapsible container until another h1, h2, or h3 is reached
// // //     while (currentNode) {
// // //         if (currentNode.tagName === 'H1' || currentNode.tagName === 'H2' || currentNode.tagName === 'H3') {
// // //             break;
// // //         } else {
// // //             var nextNode = currentNode.nextElementSibling;
// // //             collapsibleContainer.appendChild(currentNode);
// // //             currentNode = nextNode;
// // //         }
// // //     }
// // //
// // //     header.parentNode.insertBefore(collapsibleContainer, header.nextSibling);
// // //
// // //     // Function to toggle collapse
// // //     function toggleCollapse() {
// // //         if (collapsibleContainer.style.maxHeight && collapsibleContainer.style.maxHeight !== '0px') {
// // //             collapsibleContainer.style.maxHeight = '0';
// // //             toggle.innerHTML = '+';
// // //         } else {
// // //             collapsibleContainer.style.maxHeight = collapsibleContainer.scrollHeight + 'px';
// // //             toggle.innerHTML = '-';
// // //         }
// // //     }
// // //
// // //     header.addEventListener('click', toggleCollapse);
// // //
// // //     // Adjust initial toggle state and collapse state based on the URL hash
// // //     function adjustForHash() {
// // //         if (window.location.hash === '#more-attributes') {
// // //             collapsibleContainer.style.maxHeight = collapsibleContainer.scrollHeight + 'px';
// // //             toggle.innerHTML = '-';
// // //         } else {
// // //             collapsibleContainer.style.maxHeight = '0';
// // //             toggle.innerHTML = '+';
// // //         }
// // //     }
// // //
// // //     // Run the hash adjustment function on load
// // //     adjustForHash();
// // //
// // //     // Optionally, listen for hash changes if dynamic URL updates are expected on the same page
// // //     window.addEventListener('hashchange', adjustForHash);
// // // });
