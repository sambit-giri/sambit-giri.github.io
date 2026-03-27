# sambit-giri.github.io

Personal academic website of **Dr. Sambit K. Giri**, Researcher in Astrophysics & Data Science at Stockholm University.

Live at: [sambit-giri.github.io](https://sambit-giri.github.io)

## Structure

Static single-page site — no build step required. Open `index.html` directly in a browser.

```
index.html          # Layout and rendering logic (Tailwind CSS via CDN, Lucide icons)
content/            # All site content as plain JS data files
  papers.js         # Publication list
  research_pillars.js # Research themes and sub-topics
  tools.js          # Featured open-source software
  collaborations.js # Major collaborations (SKA, LOFAR, Euclid, Roman, SEarCH)
  teaching_projects.js # Educational projects
  invited_talks.js  # Conference talks
  recent_updates.js # News feed
  recent_work.js    # Featured recent papers
images/             # Figures and photos
data/               # Simulation images and other assets
```

## Updating content

All content lives in `content/*.js` — edit the relevant file and reload the page. No compilation needed.
