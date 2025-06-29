window.addEventListener("load", () => {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  let drawing = false;
  let lastResult = null;

  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  canvas.addEventListener("mousedown", () => drawing = true);
  canvas.addEventListener("mouseup", () => {
    drawing = false;
    ctx.beginPath();
  });
  canvas.addEventListener("mousemove", e => {
    if (!drawing) return;
    ctx.lineWidth = 12;
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  });

  document.getElementById("clearBtn").addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById("result").textContent = "（未認識）";
    lastResult = null;
  });

  document.getElementById("predictBtn").addEventListener("click", async () => {
    const dataUrl = canvas.toDataURL("image/png");
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl })
    });

    if (!response.ok) {
      alert("推論に失敗しました");
      return;
    }

    const result = await response.json();
    document.getElementById("result").textContent = result.prediction;
    lastResult = result;
  });

  document.getElementById("overviewBtn").addEventListener("click", () => {
    if (!lastResult || !lastResult.nodes || !lastResult.edges) {
      alert("まず推論を実行してください。");
      return;
    }

    const win = window.open("/overview", "_blank");
    win.addEventListener("load", () => {
      win.postMessage(
        { nodes: lastResult.nodes, edges: lastResult.edges },
        "*"
      );
    });
  });
});