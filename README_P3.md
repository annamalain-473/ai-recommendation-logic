# 🎯 AI Recommendation Logic — Tech Stack Recommender

**DecodeLabs AI Internship | Batch 2026 | Project 3**

---

## 📌 Description

A content-based AI recommendation engine that maps a user's skills to the most relevant tech career paths using **TF-IDF vectorization** and **Cosine Similarity**. Built as part of the **DecodeLabs AI Internship** to demonstrate the full recommendation pipeline — from raw user input to ranked, personalized output.

The engine implements the **4-Step Ranking Pipeline**:
1. **Ingestion** → Collect user skill inputs (min. 3)
2. **Scoring** → Cosine Similarity between user vector & job role vectors
3. **Sorting** → Descending order by match score
4. **Filtering** → Top-N results to prevent choice overload

---

## ✨ Features

- 🧑 Interactive user skill input (minimum 3 skills)
- 🔢 TF-IDF vectorization (penalizes generic words, rewards specific skills)
- 📐 Cosine Similarity scoring (magnitude-invariant, industry standard)
- 🏆 Top 5 ranked career path recommendations with match %
- 🚨 Cold Start detection (warns if no matching skills found)
- 📋 Full ranking table of all 15 job roles
- 📊 2-panel matplotlib visualization saved as `recommendation_results.png`
- 15 real-world job roles in the dataset

---

## 🛠️ Tech Stack

| Library      | Purpose                                  |
|--------------|------------------------------------------|
| Python 3.x   | Core language                            |
| scikit-learn | TF-IDF vectorizer + Cosine Similarity    |
| pandas       | Data handling and display                |
| numpy        | Array operations                         |
| matplotlib   | Visualization                            |

---

## ▶️ How to Run

### Prerequisites

```bash
pip install scikit-learn pandas numpy matplotlib
```

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ai-recommendation-system.git

# 2. Navigate to the folder
cd ai-recommendation-system

# 3. Run the recommender
python recommender.py
```

### Sample Interaction

```
==============================================================
  DecodeLabs AI Internship | Project 3: Tech Stack Recommender
==============================================================

Enter your skills one by one (minimum 3). Type 'done' when finished.

  Skill 1: python
  ✅ Added: python
  Skill 2: machine learning
  ✅ Added: machine learning
  Skill 3: tensorflow
  ✅ Added: tensorflow
  Skill 4: done

🧑 Your skill profile: ['python', 'machine learning', 'tensorflow']

TOP RECOMMENDED CAREER PATHS:
  🥇 Machine Learning Engineer    [██████████████████████░░░░░░░]  74.3%
  🥈 Data Scientist               [████████████████████░░░░░░░░░]  68.1%
  🥉 AI Research Scientist        [███████████████████░░░░░░░░░░]  62.5%
  4️⃣  NLP Engineer                 [████████████░░░░░░░░░░░░░░░░░]  41.2%
  5️⃣  Computer Vision Engineer     [███████████░░░░░░░░░░░░░░░░░░]  38.7%
```

---

## 🏗️ Recommendation Engine Architecture

```
INPUT (User State)
  └── User enters 3+ skills (e.g., ["python", "cloud", "docker"])
  └── Skills joined into a profile text string
        │
PROCESS (Similarity Logic)
  └── TF-IDF Vectorizer → converts text to weighted numeric vectors
  └── User vector vs 15 job role vectors
  └── Cosine Similarity → scores between 0.0 and 1.0
        │
OUTPUT (Top-N List)
  └── Sort by score (descending)
  └── Display Top 5 + Full ranking table
  └── Visualization → recommendation_results.png
```

---

## 📂 File Structure

```
ai-recommendation-system/
│
├── recommender.py              # Main recommendation engine
├── recommendation_results.png  # Auto-generated visualization (after run)
└── README.md                   # Project documentation
```

---

## 🧠 Key Concepts Demonstrated

- **Content-Based Filtering** — matches user attributes to item attributes (no user history needed)
- **TF-IDF Weighting** — rewards specific skills, penalizes generic terms
- **Cosine Similarity** — direction-based comparison, magnitude-invariant
- **4-Step Ranking Pipeline** — Ingestion → Scoring → Sorting → Filtering
- **Cold Start Problem** — detected and flagged when no skills match the vocabulary
- **Vector Space Model** — text transformed into numerical arrays for math operations

---

## 👨‍💻 Author

**ABUVAN**  
ECE Student | AI Intern @ DecodeLabs (Batch 2026)

---

*Built with ❤️ at DecodeLabs — "You are ready to build the Digital Matchmaker."*
