# -*- coding: utf-8 -*-
"""ElCaminoMisticoIA"""

import json, os, re, textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
import pdfplumber
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from llama_cpp import Llama
from difflib import SequenceMatcher
import requests

BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
PDF_DIR = BASE_DIR / "pdfs"
HISTORY_DIR = BASE_DIR / "historial"
HISTORY_DIR.mkdir(exist_ok=True)

def descargar_modelo_si_no_existe():
    if MODEL_PATH.exists():
        print("[INFO] Modelo ya existe. No se descarga.")
        return

    print("[INFO] Descargando modelo desde Google Drive…")
    url = "https://drive.google.com/uc?export=download&id=1kwCxE9g_TAyS-UEsm_kSaZA3Fdw1z66w"
    response = requests.get(url, stream=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("[INFO] Modelo descargado ✔")

# 👇 Esta es la única línea que necesitás agregar
descargar_modelo_si_no_existe()

LLM = Llama(model_path=str(MODEL_PATH), n_ctx=4096, n_gpu_layers=-1, n_threads=os.cpu_count() or 8)
print("[INFO] Modelo cargado ✔")

INDEX_HTML="""<!DOCTYPE html>
<html lang='es'>
<head><meta charset='UTF-8'><meta name='viewport' content='width=device-width,initial-scale=1.0'>
<title>El Camino Místico</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
:root{--bg:#14021c;--panel:#271742;--primary:#c300ff;--text:#ffffff}
*{box-sizing:border-box;font-family:'Poppins',sans-serif}
body{margin:0;background:var(--bg);color:var(--text);display:flex;flex-direction:column;align-items:center;padding:1.5rem}
h1{margin-top:0;font-size:2.75rem;text-align:center;line-height:1.2}
header span.small{display:block;text-align:center;margin-bottom:.75rem;font-weight:600;font-size:1.3rem}
textarea{width:100%;max-width:1000px;height:220px;resize:vertical;padding:1.25rem;border-radius:12px;background:var(--panel);color:var(--text);border:2px solid transparent;font-size:1rem}
textarea:focus{outline:none;border-color:var(--primary)}
.buttons{margin-top:1.5rem;display:flex;gap:1.5rem;flex-wrap:wrap;justify-content:center}
button{cursor:pointer;border:none;padding:.9rem 2.5rem;border-radius:12px;background:var(--primary);color:var(--text);font-size:1.1rem;font-weight:600;transition:transform .15s ease}
button:hover{transform:scale(1.05)}
#consultasBtn{display:flex;align-items:center;gap:.5rem}
#indicator{margin:1.25rem 0;font-style:italic;opacity:0;transition:opacity .3s ease}
#answerBox{width:100%;max-width:1000px;min-height:220px;background:var(--panel);border-radius:12px;padding:1.25rem;white-space:pre-wrap;overflow-y:auto}
@media(max-width:600px){h1{font-size:2.1rem}textarea{height:180px}}
</style></head><body>
<header><span class='small'>Martín Arízaga</span><h1>✨ El Camino Místico ✨</h1>
<p style='text-align:center;max-width:760px;margin:0 auto 1.5rem;'>Consulta cualquier verdad desde los libros ocultos que resguarda esta IA.</p></header>
<textarea id='question' placeholder='Escribe tu pregunta aquí…'></textarea>
<div class='buttons'><button id='askBtn'>Preguntar</button><button id='consultasBtn'><span>📄</span>Ver consultas</button></div>
<div id='indicator'>⏳ Buscando respuesta…</div>
<div id='answerBox'></div>
<script>
const q=document.getElementById('question'), ask=document.getElementById('askBtn'), hist=document.getElementById('consultasBtn'), ind=document.getElementById('indicator'), ans=document.getElementById('answerBox');
function speak(t){if(!window.speechSynthesis)return;const u=new SpeechSynthesisUtterance(t);u.lang='es-ES';window.speechSynthesis.cancel();window.speechSynthesis.speak(u);}
ask.addEventListener('click',async()=>{
 const question=q.value.trim();
 if(!question)return;
 ind.style.opacity=1; ans.textContent='';
 try{
  const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question})});
  if(!r.ok)throw new Error('Error en la consulta');
  const d=await r.json(); ans.innerHTML=d.answer; speak(ans.textContent);
 }catch(e){ans.textContent=e.message;}
 ind.style.opacity=0;
});
  hist.addEventListener("click", async () => {
    try {
      const r = await fetch("/historial");
      const list = await r.json();
      const lines = list.map(r =>
  "🕑 " + new Date(r.ts).toLocaleString() + "\\n" +
  "❓ Pregunta:\\n" + r.q + "\\n" +
  "💬 Respuesta:\\n" + (r.a?.replace(/<[^>]+>/g, "") || "")
).join("\\n\\n────────────────────────\\n\\n");
      alert(lines || "Sin consultas recientes");
    } catch (e) {
      alert("Error leyendo historial");
    }
  });
</script></body></html>"""  # <- cierre triple quote correcto

BASE_DIR = Path(__file__).parent.resolve()
PDF_DIR = BASE_DIR / "pdfs"
HISTORY_DIR = BASE_DIR / "historial"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "mistral-7b-instruct-v0.1.Q4_K_M.gguf"

HISTORY_DIR.mkdir(exist_ok=True)

print("[INFO] Cargando modelo…")
LLM = Llama(model_path=str(MODEL_PATH), n_ctx=4096, n_gpu_layers=-1, n_threads=os.cpu_count() or 8)
print("[INFO] Modelo cargado ✔")

print("[INFO] Indexando encabezados PDF…")
PDF_CACHE: List[Tuple[str, str]] = []  # (archivo, encabezado)
encabezado_patron = re.compile(r"^(CAP[IÍ]TULO|M[ÓO]DULO|SECCI[ÓO]N|TEMA|\d+)[\s\.:\-]", re.IGNORECASE)

def extract_all_blocks(pdf_path: Path) -> List[Tuple[str, str]]:
    bloques = []
    with pdfplumber.open(pdf_path) as pdf:
        all_lines = []
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            all_lines.extend(lines)

        i = 0
        while i < len(all_lines):
            line = all_lines[i]
            if encabezado_patron.search(line) or re.match(r"^[A-ZÁÉÍÓÚÑ\d\s,:;\-]{10,}$", line):
                title = line.strip()
                start = i
                i += 1
                while i < len(all_lines):
                    if encabezado_patron.search(all_lines[i]):
                        break
                    i += 1
                bloques.append((title, "\n".join(all_lines[start + 1:i]).strip()))
            else:
                i += 1
    return bloques

for pdf_path in PDF_DIR.glob("*.pdf"):
    try:
        bloques = extract_all_blocks(pdf_path)
        for title, _ in bloques:
            PDF_CACHE.append((pdf_path.name, title))
    except Exception as e:
        print(f"[WARN] Error leyendo {pdf_path.name}: {e}")

print(f"[INFO] Indexados {len(PDF_CACHE)} encabezados")

def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_block_from_title(pdf_name: str, title: str) -> str:
    pdf_path = PDF_DIR / pdf_name
    bloques = extract_all_blocks(pdf_path)
    for t, para in bloques:
        if title.strip().lower() == t.strip().lower():
            return para
    return ""

def match_exact(query: str) -> Tuple[str, str, str]:
    query_clean = query.lower().strip()
    for pdf, title in PDF_CACHE:
        if query_clean == title.lower():
            para = find_block_from_title(pdf, title)
            return (pdf, title, para)
    return None

def search_pdfs(query: str, max_hits: int = 6) -> Tuple[str, List[Tuple[str, str]]]:
    scores = [((pdf, title), similar(query, title)) for (pdf, title) in PDF_CACHE]

    if not scores:
        return "nula", []

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    best_score = sorted_scores[0][1]

    if best_score >= 0.90:
        nivel = "exacta"
        hits = [tpl for tpl, score in sorted_scores if score >= 0.90]
    elif best_score >= 0.60:
        nivel = "parcial"
        hits = [tpl for tpl, score in sorted_scores if score >= 0.60]
    else:
        return "nula", []

    return nivel, hits[:max_hits]

def save_query(q, a):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    path = HISTORY_DIR / f"{today}.json"
    data = json.loads(path.read_text()) if path.exists() else []
    data.append({"ts": datetime.utcnow().isoformat(), "q": q, "a": a})
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def clean_history():
    limit_date = (datetime.utcnow() - timedelta(days=2)).date()
    for file in HISTORY_DIR.glob("*.json"):
        try:
            file_date = datetime.strptime(file.stem, "%Y-%m-%d").date()
            if file_date < limit_date:
                file.unlink()
        except: pass

def answer_with_llm(prompt):
    system = "Eres una IA espiritual que responde extensamente en español."
    full = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST]"
    out = LLM(full, max_tokens=512, temperature=0.7)
    return out['choices'][0]['text'].strip()

app = FastAPI(title="El Camino Místico IA")

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(INDEX_HTML)

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "").strip()
    if not question:
        raise HTTPException(400, "Pregunta vacía")

    exact = match_exact(question)
    if exact:
        pdf, title, para = exact
        content = para or title
        snippet = textwrap.shorten(content, width=2000, placeholder="…")
        answer = f"<b>{pdf}</b>: {title}\n\n{snippet}"
    else:
        nivel, hits = search_pdfs(question, max_hits=6)

        if nivel == "nula":
            print("[INFO] No se encontraron coincidencias en los libros PDF")
            try:
                ext = answer_with_llm(question)
                answer = (
                    "Entre los libros sagrados no existe un tema relacionado con tu pregunta, "
                    "pero tal vez te pueda ayudar esto:\n\n" + ext
                )
            except Exception as e:
                print("[ERROR] Fallo al generar respuesta externa:", e)
                answer = "No se encontró respuesta en los libros, y ocurrió un error al generar una respuesta alternativa."

        elif nivel == "exacta" and len(hits) == 1:
            pdf, title = hits[0]
            para = find_block_from_title(pdf, title)
            snippet = textwrap.shorten(para or title, width=2000, placeholder="…")
            answer = f"<b>{pdf}</b>: {title}\n\n{snippet}"

        elif nivel == "parcial":
            options = [f"<b>{pdf}</b>: {title}" for pdf, title in hits]
            joined = "\n".join(options)
            intro = "Esto fue lo que se encontró en los Libros Sagrados de Martin, ¿cuál quieres conocer?. Cópialo y pégalo como pregunta."
            answer = f"{intro}\n\n{joined}"

        save_query(question, answer)
        clean_history()
    return JSONResponse({"answer": answer})

@app.get("/historial")
async def history():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    rec = []
    for d in (yesterday, today):
        p = HISTORY_DIR / f"{d}.json"
        if p.exists():
            rec.extend(json.loads(p.read_text()))
    rec.sort(key=lambda r: r['ts'], reverse=True)
    return JSONResponse(rec)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
