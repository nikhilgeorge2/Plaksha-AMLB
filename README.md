# Applied Machine Learning for Business
**Plaksha University | Spring 2026**
**Instructor:** Nikhil George

## Course Materials

This repository contains homework assignments and datasets for the course.

### Homework 1: Dimensionality Reduction & Customer Lifetime Value
- **Released:** January 24, 2026
- **Download:** See [Releases](https://github.com/nikhilgeorge2/Plaksha-AMLB/releases)

### Homework 2: Selective Labels, Calibration, and Simulation
- **Released:** March 2, 2026
- **Folder:** [HW2/](HW2/)
- `HW2_Notebook.ipynb` — assignment notebook (complete all work here)
- `cardiac_patients.csv` — hospital dataset (2000 patients, features + referral decisions)
- `cardiac_truth.csv` — released at Step 5; load when instructed

### Homework 3: Networks, Rankings, and Academic Prestige
- **Released:** March 15, 2026
- **Folder:** [HW3/](HW3/)
- `HW3_Notebook.ipynb` — assignment notebook (complete all work here)
- `cs_hiring_edges.csv` — CS faculty hiring network (edges)
- `cs_hiring_nodes.csv` — CS institutions with US News rankings
- `biz_hiring_edges.csv` — Business faculty hiring network (edges)
- `biz_hiring_nodes.csv` — Business institutions with US News rankings

### Homework 4: Wikipedia Navigation Agent
- **Released:** April 2026
- **Folder:** [HW4/](HW4/)
- Build an AI agent that navigates Wikipedia by following hyperlinks from a start article to a target article. The agent uses Gemini 2.5 Flash to decide which link to follow at each step, under a strict 2-minute time budget for a set of pairs.
- `agent.py` — starter template to fork and improve
- `wiki_tool.py` — provided Wikipedia link fetcher (do not modify)
- `score.py` — local scorer to test your agent before submission
- `practice_pairs.csv` — 10 pairs for local development
- `test_pairs_round2.csv` — 20-pair test pool (released post-grading)
