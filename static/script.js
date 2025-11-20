function sleep(ms) {
    return new Promise(res => setTimeout(res, ms));
}

async function sendText() {
    const text = document.getElementById("inputText").value.trim();
    const loading = document.getElementById("loading");
    const outputCard = document.getElementById("output");
    const tokenList = document.getElementById("tokenList");

    if (!text) return;

    loading.classList.remove("hidden");
    outputCard.classList.add("hidden");
    tokenList.innerHTML = "";

    const words = text.split(/\s+/);

    const res = await fetch("/api/tokenize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ words })
    });

    const data = await res.json();

    loading.classList.add("hidden");
    outputCard.classList.remove("hidden");

    let delay = 0;

    for (let item of data) {
        let div = document.createElement("div");
        div.classList.add("token-box");
        div.style.animationDelay = delay + "ms";
        div.innerHTML = `<strong>${item.token}</strong> →${item.status === "Valid split" ? `<span>${item.first_token}</span> + <span>${item.second_token}</span>` : `<span>${item.token}</span>`}`;
        delay += 120;
        tokenList.appendChild(div);
    }
}
