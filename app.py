import os
import io
import torch
import pandas as pd
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO_ID = "distilbert-base-uncased"
MAX_LEN = 256
LABELS = ["FAKE", "REAL"]

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
model = AutoModelForSequenceClassification.from_pretrained(REPO_ID).to(device)
model.eval()

def predict_one(text):
    if not text or not text.strip():
        return {}, "⚠️ Please enter some text."
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN).to(device)
    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()
    conf = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    top_label = LABELS[int(probs.argmax())]
    return conf, f"**Result:** {top_label}"

def predict_batch(csv_file):
    if csv_file is None:
        return pd.DataFrame(), None
    df = pd.read_csv(csv_file.name)
    if "text" not in df.columns:
        raise gr.Error("CSV must contain a 'text' column.")
    texts = df["text"].astype(str).tolist()
    all_preds, all_fake, all_real = [], [], []
    B = 32
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN).to(device)
        with torch.no_grad():
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        for row in probs:
            all_fake.append(float(row[0]))
            all_real.append(float(row[1]))
            all_preds.append(LABELS[int(row.argmax())])
    out = pd.DataFrame({
        "text": texts,
        "prediction": all_preds,
        "prob_fake": all_fake,
        "prob_real": all_real
    })
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8")
    return out, csv_bytes

EXAMPLES = [
    "NASA announces successful Artemis I mission, marking a new era for lunar exploration.",
    "Scientists confirm aliens built the Egyptian pyramids, official documents reveal.",
    "Indian government unveils budget focusing on renewable energy and infrastructure development.",
    "Government to replace all teachers with AI robots by 2026.",
    "UN reports significant progress in international climate agreements after COP28 summit."
]

with gr.Blocks(theme=gr.themes.Soft(), css="""
#header {text-align:center}
.prob-note {opacity:0.8; font-size:14px; color:#555;}
""") as demo:
    gr.Markdown("""
<div id="header">
  <h1>📰 Fake News Detector</h1>
  <p>DistilBERT-based classifier (Hugging Face Transformers). Paste a headline/article to get a prediction with probabilities.</p>
</div>
    """)
    with gr.Tab("Single Article"):
        txt = gr.Textbox(lines=6, placeholder="Paste a news headline or article...")
        go = gr.Button("Analyze")
        pred = gr.Label(num_top_classes=2, label="Prediction & Probabilities")
        explain = gr.Markdown("Results will appear above.", elem_id="note")
        go.click(fn=predict_one, inputs=txt, outputs=[pred, explain])
        gr.Examples(examples=EXAMPLES, inputs=txt, label="Try examples")
    with gr.Tab("Batch (CSV)"):
        gr.Markdown("Upload a **CSV** with a column named **`text`**. You'll get a table and a downloadable results file.")
        csv_in = gr.File(file_types=[".csv"], label="Upload CSV")
        run_batch = gr.Button("Run Batch Prediction")
        table = gr.Dataframe(interactive=False, label="Results")
        dl = gr.File(label="Download predictions.csv")
        def batch_wrapper(f):
            df, csv_bytes = predict_batch(f)
            if csv_bytes is None:
                return pd.DataFrame(), None
            out_path = "predictions.csv"
            with open(out_path, "wb") as w:
                w.write(csv_bytes)
            return df, out_path
        run_batch.click(batch_wrapper, inputs=csv_in, outputs=[table, dl])
    gr.Markdown("""
**Model:** `distilbert-base-uncased`   
**Labels:** `FAKE` (0), `REAL` (1)   
> The bars show class probabilities. Higher = more confidence.
    """)

if __name__ == "__main__":
    # Correct launch for Vercel/production:
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
