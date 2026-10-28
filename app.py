import os
import pickle
import numpy as np
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# ----------------------------------------------------------------------------
# Load model once (cold start) -- file must sit next to app.py
# ----------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "random_forest.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ----------------------------------------------------------------------------
# Dropdown options (user sees names, model receives the encoded number).
# The POSITION in each list = the label-encoded value used in training
# (LabelEncoder sorts alphabetically). Replace these names with the exact
# categories from your own dataset, in alphabetical order.
# ----------------------------------------------------------------------------
CATEGORIES = {
    "company_name": [
        "Amazon", "Apple", "Cisco", "Dell", "Google", "HP", "IBM", "Intel",
        "Meta", "Microsoft", "Netflix", "Nvidia", "Oracle", "SAP",
        "Salesforce", "Shopify", "Spotify", "Tesla", "Uber", "Zoom",
    ],
    "industry": [
        "Automotive", "E-commerce", "Finance", "Healthcare",
        "Manufacturing", "Retail", "Technology",
    ],
    "country": ["Australia", "Canada", "Germany", "India", "UK", "USA"],
    "company_size": ["Large", "Medium", "Small", "Startup"],
    "reason_for_layoffs": [
        "AI Automation", "Cost Cutting", "Market Slowdown",
        "Over-hiring", "Restructuring",
    ],
    "hiring_trend": ["Decreasing", "Freeze", "Increasing", "Stable"],
    "top_hiring_role": [
        "AI Engineer", "Cybersecurity Analyst", "Data Scientist",
        "DevOps Engineer", "Product Manager", "Sales Executive",
        "Software Engineer", "UX Designer",
    ],
}

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
YEARS = [2024, 2025, 2026]

# Exact feature order the model was trained with
FEATURES = list(model.feature_names_in_)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market Outlook Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#070b17; --card:rgba(255,255,255,.05); --border:rgba(255,255,255,.10);
    --text:#eef1fb; --muted:#9aa3bf; --accent:#7c5cff; --accent2:#22d3ee;
    --bull:#22c55e; --rec:#ef4444; --stable:#f59e0b;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:'Inter',system-ui,sans-serif;color:var(--text);min-height:100vh;
    background:
      radial-gradient(900px 500px at 10% -10%,rgba(124,92,255,.35),transparent 60%),
      radial-gradient(800px 500px at 100% 0%,rgba(34,211,238,.22),transparent 60%),
      var(--bg);
    padding:40px 18px 70px;
  }
  .wrap{max-width:1080px;margin:0 auto}
  header{text-align:center;margin-bottom:34px}
  .badge{display:inline-block;padding:6px 14px;border-radius:999px;font-size:12px;font-weight:600;
    letter-spacing:.08em;text-transform:uppercase;color:var(--accent2);
    background:rgba(34,211,238,.10);border:1px solid rgba(34,211,238,.25)}
  h1{font-size:clamp(30px,5vw,48px);font-weight:800;margin:16px 0 10px;letter-spacing:-.02em;
    background:linear-gradient(90deg,#fff,#b9a8ff 50%,#67e8f9);-webkit-background-clip:text;background-clip:text;color:transparent}
  header p{color:var(--muted);max-width:620px;margin:0 auto;line-height:1.6}

  .grid{display:grid;grid-template-columns:1fr;gap:20px}
  @media(min-width:900px){.grid{grid-template-columns:1.6fr 1fr;align-items:start}}

  .card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:24px;
    backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px)}
  .section{margin-bottom:26px}
  .section:last-of-type{margin-bottom:8px}
  .section h3{font-size:13px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#c8bfff;
    margin-bottom:14px;display:flex;align-items:center;gap:8px}
  .section h3 span{width:26px;height:26px;border-radius:8px;display:grid;place-items:center;font-size:13px;
    background:linear-gradient(135deg,var(--accent),var(--accent2))}
  .fields{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px}
  label{display:block;font-size:12.5px;color:var(--muted);margin-bottom:6px;font-weight:500}
  input,select{
    width:100%;padding:12px 14px;border-radius:12px;font:inherit;font-size:14.5px;color:var(--text);
    background:rgba(8,12,26,.7);border:1px solid var(--border);outline:none;transition:.2s;
  }
  select option{background:#0c1226;color:var(--text)}
  input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(124,92,255,.25)}

  button{
    width:100%;margin-top:18px;padding:15px;border:0;border-radius:14px;cursor:pointer;
    font:inherit;font-weight:700;font-size:15.5px;color:#fff;letter-spacing:.02em;
    background:linear-gradient(135deg,var(--accent),#4f8cff 55%,var(--accent2));
    box-shadow:0 10px 30px -10px rgba(124,92,255,.8);transition:.2s;
  }
  button:hover{transform:translateY(-2px);filter:brightness(1.08)}
  button:disabled{opacity:.6;cursor:wait;transform:none}

  .result{position:sticky;top:20px}
  .result h2{font-size:15px;color:var(--muted);font-weight:600;margin-bottom:16px}
  .placeholder{text-align:center;padding:36px 10px;color:var(--muted);line-height:1.6;font-size:14px}
  .placeholder .ico{font-size:42px;margin-bottom:10px}
  .verdict{text-align:center;padding:22px 12px;border-radius:16px;margin-bottom:20px;border:1px solid var(--border)}
  .verdict small{display:block;color:var(--muted);font-size:12px;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px}
  .verdict strong{font-size:30px;font-weight:800;letter-spacing:-.01em}
  .verdict .conf{margin-top:6px;color:var(--muted);font-size:13px}
  .bull{background:rgba(34,197,94,.12);border-color:rgba(34,197,94,.4)} .bull strong{color:var(--bull)}
  .rec{background:rgba(239,68,68,.12);border-color:rgba(239,68,68,.4)} .rec strong{color:var(--rec)}
  .stable{background:rgba(245,158,11,.12);border-color:rgba(245,158,11,.4)} .stable strong{color:var(--stable)}

  .bar{margin-bottom:14px}
  .bar .row{display:flex;justify-content:space-between;font-size:13.5px;margin-bottom:6px;font-weight:500}
  .track{height:10px;border-radius:99px;background:rgba(255,255,255,.08);overflow:hidden}
  .fill{height:100%;width:0;border-radius:99px;transition:width .9s cubic-bezier(.2,.8,.2,1)}
  .err{color:#fca5a5;background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);padding:12px;border-radius:12px;font-size:13.5px}
  .spin{display:inline-block;width:16px;height:16px;border:2px solid rgba(255,255,255,.4);border-top-color:#fff;
    border-radius:50%;animation:s .7s linear infinite;vertical-align:-3px;margin-right:8px}
  @keyframes s{to{transform:rotate(360deg)}}
  footer{text-align:center;color:var(--muted);font-size:12.5px;margin-top:34px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="badge">Random Forest · ML Model</div>
    <h1>Market Outlook Predictor</h1>
    <p>Enter company, workforce and financial indicators to predict whether the outlook is a Bull Market, Recession or Stable.</p>
  </header>

  <div class="grid">
    <form id="form" class="card" autocomplete="off">

      <div class="section">
        <h3><span>🏢</span> Company</h3>
        <div class="fields">
          <div><label>Company</label><select name="company_name">{{ opts('company_name') }}</select></div>
          <div><label>Industry</label><select name="industry">{{ opts('industry') }}</select></div>
          <div><label>Country</label><select name="country">{{ opts('country') }}</select></div>
          <div><label>Company size</label><select name="company_size">{{ opts('company_size') }}</select></div>
          <div><label>Month</label>
            <select name="month">{% for m in months %}<option value="{{ loop.index }}">{{ m }}</option>{% endfor %}</select></div>
          <div><label>Year</label>
            <select name="year">{% for y in years %}<option value="{{ y }}" {% if y==2025 %}selected{% endif %}>{{ y }}</option>{% endfor %}</select></div>
        </div>
      </div>

      <div class="section">
        <h3><span>📉</span> Layoffs</h3>
        <div class="fields">
          <div><label>Layoffs count</label><input type="number" name="layoffs_count" value="1500" step="any" required></div>
          <div><label>Layoff percentage (%)</label><input type="number" name="layoff_percentage" value="8" step="any" required></div>
          <div><label>Reason for layoffs</label><select name="reason_for_layoffs">{{ opts('reason_for_layoffs') }}</select></div>
        </div>
      </div>

      <div class="section">
        <h3><span>🤖</span> AI Impact</h3>
        <div class="fields">
          <div><label>AI automation impact</label><input type="number" name="ai_automation_impact" value="8" step="any" required></div>
          <div><label>AI replacement risk (1–10)</label><input type="number" name="ai_replacement_risk" value="5" step="any" required></div>
          <div><label>AI adoption level (1–10)</label><input type="number" name="ai_adoption_level" value="5" step="any" required></div>
        </div>
      </div>

      <div class="section">
        <h3><span>👥</span> Hiring</h3>
        <div class="fields">
          <div><label>Open roles</label><input type="number" name="open_roles" value="3000" step="any" required></div>
          <div><label>Hiring trend</label><select name="hiring_trend">{{ opts('hiring_trend') }}</select></div>
          <div><label>Top hiring role</label><select name="top_hiring_role">{{ opts('top_hiring_role') }}</select></div>
          <div><label>Remote jobs (%)</label><input type="number" name="remote_jobs_percentage" value="50" step="any" required></div>
        </div>
      </div>

      <div class="section">
        <h3><span>💰</span> Financials & Sentiment</h3>
        <div class="fields">
          <div><label>Stock growth (%)</label><input type="number" name="stock_growth_percent" value="10" step="any" required></div>
          <div><label>Revenue growth (%)</label><input type="number" name="revenue_growth_percent" value="8" step="any" required></div>
          <div><label>Salary budget change (%)</label><input type="number" name="salary_budget_change" value="3" step="any" required></div>
          <div><label>Employee sentiment (1–10)</label><input type="number" name="employee_sentiment" value="6.5" step="any" required></div>
          <div><label>Job security score (1–10)</label><input type="number" name="job_security_score" value="6" step="any" required></div>
        </div>
      </div>

      <button type="submit" id="btn">Predict Outlook</button>
    </form>

    <aside class="card result" id="result">
      <h2>Prediction result</h2>
      <div class="placeholder"><div class="ico">📊</div>Fill in the form and hit <b>Predict Outlook</b> to see the model's forecast.</div>
    </aside>
  </div>

  <footer>Built with Flask · scikit-learn · Deployed on Vercel</footer>
</div>

<script>
const COLORS = {"Bull Market":"var(--bull)","Recession":"var(--rec)","Stable":"var(--stable)"};
const CLS    = {"Bull Market":"bull","Recession":"rec","Stable":"stable"};
const ICONS  = {"Bull Market":"📈","Recession":"📉","Stable":"⚖️"};

document.getElementById('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('btn');
  const box = document.getElementById('result');
  btn.disabled = true; btn.innerHTML = '<span class="spin"></span>Predicting...';

  const data = Object.fromEntries(new FormData(e.target).entries());
  try {
    const res = await fetch('/predict', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)
    });
    const out = await res.json();
    if(!res.ok) throw new Error(out.error || 'Prediction failed');

    const top = Object.entries(out.probabilities).sort((a,b)=>b[1]-a[1]);
    box.innerHTML = `
      <h2>Prediction result</h2>
      <div class="verdict ${CLS[out.prediction]||''}">
        <small>Predicted outlook</small>
        <strong>${ICONS[out.prediction]||''} ${out.prediction}</strong>
        <div class="conf">Confidence: ${(top[0][1]*100).toFixed(1)}%</div>
      </div>
      ${top.map(([k,v]) => `
        <div class="bar">
          <div class="row"><span>${k}</span><span>${(v*100).toFixed(1)}%</span></div>
          <div class="track"><div class="fill" data-w="${(v*100).toFixed(1)}" style="background:${COLORS[k]||'var(--accent)'}"></div></div>
        </div>`).join('')}`;
    requestAnimationFrame(()=>requestAnimationFrame(()=>{
      box.querySelectorAll('.fill').forEach(f => f.style.width = f.dataset.w + '%');
    }));
    if (window.innerWidth < 900) box.scrollIntoView({behavior:'smooth'});
  } catch(err){
    box.innerHTML = `<h2>Prediction result</h2><div class="err">⚠️ ${err.message}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = 'Predict Outlook';
  }
});
</script>
</body>
</html>
"""


def opts(field):
    """Render <option> tags; value = label-encoded index, text = readable name."""
    from markupsafe import Markup, escape
    return Markup("".join(
        f'<option value="{i}">{escape(label)}</option>'
        for i, label in enumerate(CATEGORIES[field])
    ))


@app.route("/")
def index():
    return render_template_string(HTML, opts=opts, months=MONTHS, years=YEARS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        row = [float(payload[name]) for name in FEATURES]
        X = np.array([row], dtype=float)

        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]

        return jsonify({
            "prediction": str(pred),
            "probabilities": {str(c): float(p) for c, p in zip(model.classes_, probs)},
        })
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# Local run: python app.py
if __name__ == "__main__":
    app.run(debug=True)