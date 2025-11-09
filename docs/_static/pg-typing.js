// document.addEventListener("DOMContentLoaded", function () {
//     // Internal links (within your built docs)
//     const internalLinks = {
//         "IntoValidation": "paguro.typing.IntoValidation.html",
//         "IntoKeepColumns": "paguro.typing.IntoKeepColumns.html",
//         "OnSuccess": "paguro.typing.OnSuccess.html",
//         "OnFailure": "paguro.typing.OnFailure.html",
//         "OnFailureExtra": "paguro.typing.OnFailureExtra.html",
//         "ValidatorOrExpr": "paguro.typing.ValidatorOrExpr.html",
//         "IntoDataType": "paguro.typing.IntoDataType.html",
//     };
//
//     // External links (absolute URLs to external documentation)
//     const externalLinks = {
//         "DataFrame": "https://docs.pola.rs/py-polars/html/reference/dataframe/index.html",
//         "LazyFrame": "https://docs.pola.rs/api/python/stable/reference/lazyframe/index.html",
//         "Series": "https://docs.pola.rs/api/python/stable/reference/series/index.html",
//         "Selector": "https://docs.pola.rs/api/python/stable/reference/selectors.html",
//
//         "polars.DataFrame": "https://docs.pola.rs/py-polars/html/reference/dataframe/index.html",
//         "polars.LazyFrame": "https://docs.pola.rs/api/python/stable/reference/lazyframe/index.html",
//         "polars.Series": "https://docs.pola.rs/api/python/stable/reference/series/index.html",
//         "polars.selectors.Selector": "https://docs.pola.rs/api/python/stable/reference/selectors.html",
//     };
//
//     document.querySelectorAll(".sig-param .pre, .sig-return .pre").forEach(function (el) {
//         const txt = el.textContent.trim();
//
//         // Helper to wrap element with link
//         function wrapLink(href, className) {
//             const a = document.createElement("a");
//             a.href = href;
//             a.className = className;
//             a.textContent = txt;
//             // External links open in new tab
//             if (className.includes("external")) {
//                 a.target = "_blank";
//                 a.rel = "noopener noreferrer";
//             }
//             el.replaceWith(a);
//         }
//
//         if (Object.prototype.hasOwnProperty.call(internalLinks, txt)) {
//             wrapLink(internalLinks[txt], "reference internal");
//         } else if (Object.prototype.hasOwnProperty.call(externalLinks, txt)) {
//             wrapLink(externalLinks[txt], "reference external");
//         }
//     });
// });


document.addEventListener("DOMContentLoaded", function () {
    const internalLinks = {
        "ValidationMode": "paguro.typing.ValidationMode",
        "Validators": "paguro.typing.Validators",
        "ValidatorOrExpr": "paguro.typing.ValidatorOrExpr",
        "IntoValidators": "paguro.typing.IntoValidators",
        "IntoValidation": "paguro.typing.IntoValidation",
        "FieldsValidators": "paguro.typing.FieldsValidators",
        "IntoKeepColumns": "paguro.typing.IntoKeepColumns",
        "OnSuccess": "paguro.typing.OnSuccess",
        "OnFailure": "paguro.typing.OnFailure",
        "OnFailureExtra": "paguro.typing.OnFailureExtra",
        "FrameLike": "paguro.typing.FrameLike",
    };

    const externalLinks = {
        "DataFrame": "https://docs.pola.rs/py-polars/html/reference/dataframe/index.html",
        "LazyFrame": "https://docs.pola.rs/api/python/stable/reference/lazyframe/index.html",
        "Series": "https://docs.pola.rs/api/python/stable/reference/series/index.html",
        "Selector": "https://docs.pola.rs/api/python/stable/reference/selectors.html",

        "polars.DataFrame": "https://docs.pola.rs/py-polars/html/reference/dataframe/index.html",
        "polars.LazyFrame": "https://docs.pola.rs/api/python/stable/reference/lazyframe/index.html",
        "polars.Series": "https://docs.pola.rs/api/python/stable/reference/series/index.html",
        "polars.selectors.Selector": "https://docs.pola.rs/api/python/stable/reference/selectors.html",
    };

    const docsRoot = "/paguro/latest/api/"; // <-- Base path for internal docs

    document.querySelectorAll(".sig-param .pre, .sig-return .pre").forEach(function (el) {
        if (el.closest("a")) return;

        const txt = el.textContent.trim();
        let href = null;
        let classes = [];

        if (Object.prototype.hasOwnProperty.call(internalLinks, txt)) {
            href = docsRoot + internalLinks[txt] + "/";
            classes = ["reference", "internal"];
        } else if (Object.prototype.hasOwnProperty.call(externalLinks, txt)) {
            href = externalLinks[txt];
            classes = ["reference", "external"];
        }

        if (!href) return;

        const a = document.createElement("a");
        a.href = href;
        a.classList.add(...classes);

        if (classes.includes("external")) {
            a.target = "_blank";
            a.rel = "noopener noreferrer";
        }

        el.replaceWith(a);
        a.appendChild(el);
    });
});
