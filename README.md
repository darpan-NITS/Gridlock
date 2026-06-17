#  **Gridlock: EventFlow Copilot**

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-Dark%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Hackathon-Prototype-7C3AED?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Prototype%20Phase-10B981?style=for-the-badge" />
</p>

<p align="center">
  <b>An AI-assisted traffic operations copilot that forecasts event-driven congestion and recommends manpower, barricades, and diversion plans.</b>
</p>

<p align="center">
  <i>Built for Gridlock Hackathon 2.0 • Solo-built prototype • Focused on practical city-ops impact</i>
</p>

---

##  What this project does

**Gridlock** turns raw incident data into an operator-friendly decision dashboard for traffic control rooms.

It helps answer the questions that matter in real life:

* **How risky is this event or incident?**
* **Which corridor is most vulnerable?**
* **How many officers and barricades should be deployed?**
* **What happens if crowd size or road closure conditions change?**
* **Why is the system making this recommendation?**

Instead of just showing charts, the project behaves like a **mini traffic command center**.

---

##  Problem Statement

### **Event-Driven Congestion (Planned & Unplanned)**

Political rallies, festivals, sports events, construction activities, and sudden gatherings create localized traffic breakdowns.

### Why it is hard today

* Event impact is not quantified in advance.
* Resource deployment is experience-driven.
* No post-event learning system.

### Product direction

**How can historical and real-time data be used to forecast event-related traffic impact and recommend optimal manpower, barricading, and diversion plans?**

---

##  Core Features

###  Forecast the impact

* Predict congestion severity
* Estimate expected delay
* Estimate recovery time
* Display confidence / severity indicators

###  Corridor risk intelligence

* Rank corridors by vulnerability
* Identify high-risk zones from historical patterns
* Highlight repeated trouble corridors and hotspots

###  What-if simulator

* Test changes in crowd scale, duration, road closure, time of day, and event type
* Compare submitted plan vs simulated scenario
* See how the recommended response changes

###  Explainable recommendations

* Show why a corridor or event is considered high risk
* Surface historical patterns behind the recommendation
* Keep the logic readable and operational

###  Response timeline

* Incident detected
* Officer dispatch
* Barricade setup
* Diversion active
* Expected clearance

###  Resource optimization

* Minimum safe plan
* Recommended plan
* Aggressive plan
* Officers and barricades per scenario
* Expected recovery improvement

###  AI Traffic Commander *(optional / if time permits)*

* Ask questions like:

  * Why is this corridor high risk?
  * What happens if crowd increases by 30%?
  * Why did the system recommend 5 officers?
* Rule-based or LLM-assisted depending on build time

---

##  Dataset Overview

The prototype is built using a Bengaluru incident / event operations dataset with **8,173 rows** and **46 columns**.

### Most useful fields

* `start_datetime`, `end_datetime`, `created_date`, `modified_datetime`
* `latitude`, `longitude`
* `event_type`, `event_cause`, `status`, `priority`
* `requires_road_closure`
* `corridor`, `junction`, `police_station`
* `description`, `veh_type`, `reason_breakdown`
* resolution-related fields like `closed_datetime`, `resolved_datetime`

### What the data tells us

* Most events are **unplanned**.
* Common causes include **vehicle breakdown, construction, water logging, accident, tree fall, public events, processions, protests, and VIP movement**.
* The dataset is sparse in some operational columns, so the app is designed to be **robust to missingness**.

---

##  System Architecture

```text
Dataset → Cleaning & Feature Engineering → Risk Scoring Engine
       → Corridor Vulnerability Model → What-if Simulator
       → Explainability Layer → Response Planner
       → Streamlit Command Center Dashboard
```

### Logic layers

* **Deterministic rules** for fast, stable recommendations
* **Lightweight statistical scoring** for corridor risk
* **Scenario simulation** for changing inputs
* **Optional ML/LLM layer** for richer explanation and Q&A

---

##  Why this project is worth building

This is not a generic dashboard.
It is a **decision-support product**.

That matters because judges tend to reward projects that:

* solve a concrete operational problem,
* show clear reasoning,
* and look usable by real people.

This prototype focuses on:

* **impact** over hype
* **clarity** over complexity
* **demo value** over research theater

---

##  Tech Stack

* **Frontend / App:** Streamlit
* **Maps:** Leaflet / Folium / Map-based heatmap layer
* **Data:** Pandas, NumPy
* **Visualization:** Plotly / Matplotlib / Streamlit charts
* **ML / scoring:** Scikit-learn or rule-based scoring
* **Optional AI layer:** LLM-assisted Q&A for the command-center panel

---

##  Screenshots

> Add your best dashboard screenshots here.

<p align="center">
  <img src="assets/screenshot-home.png" width="900" alt="Dashboard home" />
</p>

<p align="center">
  <img src="assets/screenshot-simulator.png" width="900" alt="What-if simulator" />
</p>

<p align="center">
  <img src="assets/screenshot-risk.png" width="900" alt="Risk intelligence view" />
</p>

---

##  How it works

### 1) Incident intake

The user selects or enters:

* event cause
* event type
* junction
* corridor
* hour of day
* road closure flag

### 2) Forecast generation

The app computes or estimates:

* severity score
* delay minutes
* recovery estimate
* vulnerability score

### 3) Response planning

The system recommends:

* officer allocation
* barricade count
* diversion strategy
* action priority

### 4) Simulation

The what-if panel changes the assumptions and updates the response plan.

### 5) Explainability

The app shows why the event is high risk using historical and contextual signals.

---

##  Feature Highlights

### Live Risk Intelligence Map

A visual hotspot view showing incident density and corridor-level pressure.

### Historical Corridor Intelligence

Ranks roads by average resolution time, closure rate, and repeated disruption patterns.

### Scenario Comparison

Shows submitted plan vs simulated scenario side by side.

### Control Room Response Card

Summarizes the action plan in a format that feels useful to an operator.

### Post-event learning *(future scope)*

Compare predicted vs actual impact to improve future response planning.

---

## Metrics / Impact Targets

These are prototype targets, not production claims.

* Reduce manual decision time for event response planning
* Improve prioritization of high-risk corridors
* Make recommendations explainable and operational
* Help users compare multiple response plans quickly

---

##  Project Structure

```text
.
├── app.py
├── data/
│   └── astram_event_data.csv
├── assets/
│   ├── screenshot-home.png
│   ├── screenshot-simulator.png
│   └── screenshot-risk.png
├── utils/
│   ├── data_processing.py
│   ├── scoring.py
│   ├── simulator.py
│   └── explainability.py
├── requirements.txt
└── README.md
```

---

## Run Locally

### 1) Clone the repo

```bash
git clone https://github.com/your-username/gridlock-eventflow-copilot.git
cd gridlock-eventflow-copilot
```

### 2) Create a virtual environment

```bash
python -m venv .venv
```

### 3) Activate it

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 4) Install dependencies

```bash
pip install -r requirements.txt
```

### 5) Run the app

```bash
streamlit run app.py
```

---

##  Requirements

Example dependencies:

```txt
streamlit
pandas
numpy
plotly
folium
streamlit-folium
scikit-learn
matplotlib
```

---

## Hackathon Design Choices

### What I prioritized

* Fast demo load time
* Clear visual hierarchy
* Real-world operations framing
* Simple but believable intelligence
* Solo-builder feasibility

### What I intentionally avoided

* Over-engineered deep learning pipelines
* Fake precision
* Too many charts with no story
* Features that are impressive but useless in a control-room setting

---

## Roadmap

### Done / in progress

* Dashboard shell
* Live heatmap
* Incident feed
* Historical corridor analytics
* Resource planning section
* What-if simulator

### Next

* Forecast panel
* Corridor vulnerability ranking
* Explainability cards
* Response timeline
* Better scenario comparison
* Optional AI Q&A layer

---

## Acknowledgements

* **Flipkart Gridlock Hackathon 2.0** for the problem statement
* Open-source Python ecosystem
* Streamlit for making the prototype fast to build

---

## Final Pitch

**Gridlock: EventFlow Copilot** helps traffic teams forecast disruption, rank vulnerable corridors, and deploy the right response before congestion spirals out of control.

---

## License

This project is submitted for hackathon demonstration purposes. Add your preferred license if you plan to open-source it.
