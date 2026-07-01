# ============================================================
#  DecodeLabs AI Internship | Batch 2026
#  Project 3: AI Recommendation Logic
#  Tech Stack Recommender — TF-IDF + Cosine Similarity
#  Author   : ABUVAN
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise      import cosine_similarity

# ============================================================
# DATASET — Job Roles & Their Required Skills
# (This acts as our "items" in the recommendation engine)
# ============================================================
job_roles = [
    {
        "role"  : "Data Scientist",
        "skills": "python machine learning deep learning statistics sql data analysis numpy pandas scikit-learn tensorflow"
    },
    {
        "role"  : "Machine Learning Engineer",
        "skills": "python machine learning tensorflow pytorch deep learning algorithms model deployment docker kubernetes"
    },
    {
        "role"  : "Data Analyst",
        "skills": "sql excel python data analysis visualization tableau power bi statistics reporting business intelligence"
    },
    {
        "role"  : "AI Research Scientist",
        "skills": "python deep learning pytorch research neural networks nlp computer vision mathematics optimization"
    },
    {
        "role"  : "Backend Developer",
        "skills": "python java nodejs sql rest api docker databases postgresql mongodb microservices git"
    },
    {
        "role"  : "Frontend Developer",
        "skills": "javascript html css react nodejs typescript ui ux design git responsive web"
    },
    {
        "role"  : "DevOps Engineer",
        "skills": "docker kubernetes aws linux ci cd git jenkins automation cloud infrastructure terraform ansible"
    },
    {
        "role"  : "Cloud Architect",
        "skills": "aws azure gcp cloud infrastructure docker kubernetes terraform networking security automation devops"
    },
    {
        "role"  : "Cybersecurity Analyst",
        "skills": "networking security linux ethical hacking penetration testing firewalls siem python encryption vulnerabilities"
    },
    {
        "role"  : "Full Stack Developer",
        "skills": "javascript python react nodejs html css sql mongodb docker git rest api microservices"
    },
    {
        "role"  : "NLP Engineer",
        "skills": "python nlp natural language processing transformers bert gpt huggingface deep learning tensorflow pytorch text"
    },
    {
        "role"  : "Computer Vision Engineer",
        "skills": "python opencv deep learning convolutional neural networks image processing tensorflow pytorch yolo object detection"
    },
    {
        "role"  : "Business Intelligence Developer",
        "skills": "sql power bi tableau data warehousing etl excel reporting analytics business intelligence visualization"
    },
    {
        "role"  : "Embedded Systems Engineer",
        "skills": "c c++ microcontrollers arduino raspberry pi rtos embedded linux hardware programming iot sensors"
    },
    {
        "role"  : "IoT Developer",
        "skills": "python c embedded systems arduino raspberry pi mqtt sensors cloud aws iot networking protocols"
    },
]

# ============================================================
# PHASE 1 — INPUT: Collect User Skills (min 3 required)
# ============================================================
print("=" * 62)
print("  DecodeLabs AI Internship | Project 3: Tech Stack Recommender")
print("=" * 62)
print("\n🎯 This engine maps your skills to the best-matching job roles")
print("   using TF-IDF vectorization + Cosine Similarity.\n")

print("📌 Available skill keywords (examples):")
print("   python, java, sql, machine learning, deep learning, docker,")
print("   aws, react, javascript, nlp, tensorflow, cloud, kubernetes,")
print("   data analysis, networking, security, embedded, iot, git\n")

# --- Collect at least 3 skills from the user ----------------
user_skills_list = []
print("Enter your skills one by one (minimum 3). Type 'done' when finished.\n")

while True:
    skill = input(f"  Skill {len(user_skills_list)+1}: ").strip().lower()
    if skill == 'done':
        if len(user_skills_list) < 3:
            print("  ⚠️  Please enter at least 3 skills!")
            continue
        break
    if skill:
        user_skills_list.append(skill)
        print(f"  ✅ Added: {skill}")

user_profile_text = " ".join(user_skills_list)
print(f"\n🧑 Your skill profile: {user_skills_list}")

# ============================================================
# PHASE 2 — PROCESS: TF-IDF Vectorization + Cosine Similarity
# ============================================================
print("\n⚙️  Processing similarity engine...")

# Step 1: Build the corpus (all job role skill texts + user profile)
corpus      = [role["skills"] for role in job_roles]
role_names  = [role["role"]   for role in job_roles]

# Step 2: TF-IDF Vectorization
#   - Converts text into weighted numerical vectors
#   - Penalizes common generic words, rewards specific terms
vectorizer   = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)          # job role vectors
user_vector  = vectorizer.transform([user_profile_text]) # user profile vector

print(f"   Vocabulary size  : {len(vectorizer.vocabulary_)} unique skill terms")
print(f"   Job roles indexed: {len(job_roles)}")

# Step 3: Cosine Similarity — 4-Step Ranking Pipeline
#   Step 1 Ingestion  → done above
#   Step 2 Scoring    → cosine similarity scores
#   Step 3 Sorting    → descending order
#   Step 4 Filtering  → Top N results

similarity_scores = cosine_similarity(user_vector, tfidf_matrix)[0]

# Build results dataframe
results_df = pd.DataFrame({
    "Job Role"        : role_names,
    "Match Score"     : similarity_scores,
    "Match %"         : (similarity_scores * 100).round(2)
}).sort_values("Match Score", ascending=False).reset_index(drop=True)

# ============================================================
# PHASE 3 — OUTPUT: Top-N Recommendations
# ============================================================
TOP_N = 5

print("\n" + "=" * 62)
print("  TOP RECOMMENDED CAREER PATHS FOR YOUR SKILL PROFILE")
print("=" * 62)
print(f"\n  Input Skills : {user_skills_list}\n")

top_results = results_df.head(TOP_N)

for i, row in top_results.iterrows():
    bar_len = int(row["Match Score"] * 30)
    bar     = "█" * bar_len + "░" * (30 - bar_len)
    medal   = ["🥇", "🥈", "🥉", "4️⃣ ", "5️⃣ "][i]
    print(f"  {medal} {row['Job Role']:<30}  [{bar}]  {row['Match %']:.1f}%")

print("\n" + "=" * 62)

# Cold start detection
if similarity_scores.max() == 0:
    print("\n⚠️  COLD START DETECTED: No matching skills found in our database.")
    print("   Try using standard skill keywords like: python, sql, docker, etc.")
else:
    best_match = top_results.iloc[0]
    print(f"\n✅ Best Match → {best_match['Job Role']} ({best_match['Match %']:.1f}% alignment)")

# Full ranked table
print("\n📋 Full Ranking Table:")
print(results_df[["Job Role", "Match %"]].to_string(index=True))

# ============================================================
# VISUALIZATION — Bar Chart of Top N Matches
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    f"DecodeLabs Project 3 — Tech Stack Recommender\nSkills: {user_skills_list}",
    fontsize=13, fontweight='bold'
)

colors_top = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']

# --- Plot 1: Horizontal bar chart (Top N) -------------------
bars = axes[0].barh(
    top_results["Job Role"][::-1],
    top_results["Match %"][::-1],
    color=colors_top[::-1],
    edgecolor='white',
    height=0.6
)
axes[0].set_xlabel("Match Score (%)")
axes[0].set_title(f"Top {TOP_N} Job Role Matches")
axes[0].set_xlim(0, 100)
axes[0].axvline(x=50, color='gray', linestyle='--', alpha=0.4, label='50% threshold')
for bar, val in zip(bars, top_results["Match %"][::-1]):
    axes[0].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)
axes[0].legend(fontsize=9)

# --- Plot 2: All roles similarity scores --------------------
all_colors = ['#2ecc71' if i < TOP_N else '#bdc3c7'
              for i in range(len(results_df))]
axes[1].bar(
    range(len(results_df)),
    results_df["Match %"],
    color=all_colors,
    edgecolor='white'
)
axes[1].set_xticks(range(len(results_df)))
axes[1].set_xticklabels(results_df["Job Role"], rotation=45, ha='right', fontsize=8)
axes[1].set_ylabel("Match Score (%)")
axes[1].set_title("Similarity Scores — All Job Roles")
axes[1].set_ylim(0, 100)
axes[1].grid(axis='y', alpha=0.3)
green_patch = mpatches.Patch(color='#2ecc71', label=f'Top {TOP_N} matches')
gray_patch  = mpatches.Patch(color='#bdc3c7', label='Other roles')
axes[1].legend(handles=[green_patch, gray_patch], fontsize=9)

plt.tight_layout()
plt.savefig("recommendation_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n📈 Visualization saved → recommendation_results.png")
print("=" * 62)
