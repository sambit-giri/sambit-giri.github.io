// recent_updates.js — news and updates shown in the Recent Updates section
//
// Each entry:
//   date    : ISO format "YYYY-MM" — used for sorting (newest first)
//   display : human-readable date string shown in the UI
//   type    : label shown as a badge (e.g. "Paper", "Position", "Event", "Teaching", "Supervision", "Upcoming")
//   text    : HTML string for the update text (links allowed)
//   color   : Tailwind text color class for the badge
//   bg      : Tailwind bg color class for the badge (use /10 opacity)

const recentUpdates = [
  // {
  //   date:    "2026-10",
  //   display: "Oct 2026",
  //   type:    "Teaching",
  //   text:    "Scheduled to teach <strong>Cosmology</strong> (7.5 ECTS) at Stockholm University.",
  //   color:   "text-violet-400",
  //   bg:      "bg-violet-400/10"
  // },
  {
    date:    "2026-07",
    display: "Jun–Jul 2026",
    type:    "Talk",
    text:    "Invited talk at the <strong>European Astronomical Society Annual Meeting</strong>, Lausanne, Switzerland — <em>Witnessing the Epoch of Reionization with the Square Kilometre Array</em>.",
    color:   "text-pink-400",
    bg:      "bg-pink-400/10"
  },
  {
    date:    "2026-06",
    display: "Jun 2026",
    type:    "Book",
    text:    "Contributing author to <a href='https://arxiv.org/html/2606.20366/' target='_blank' class='text-cyan-400 hover:text-cyan-300 hover:underline transition-colors'><strong>Advancing Astrophysics with the SKA II</strong></a> — SKAO science book.",
    color:   "text-cyan-400",
    bg:      "bg-cyan-400/10"
  },
  {
    date:    "2026-06",
    display: "Jun 2026",
    type:    "Paper",
    text:    "Submitted to MNRAS: <a href='https://arxiv.org/abs/2606.14682' target='_blank' class='text-cyan-400 hover:text-cyan-300 hover:underline transition-colors'>Impact of 21-cm foreground mitigation strategies on reionization power spectrum constraints</a>.",
    color:   "text-cyan-400",
    bg:      "bg-cyan-400/10"
  },
  {
    date:    "2026-06",
    display: "Jan–Jun 2026",
    type:    "Supervision",
    text:    "Master's student <strong>Rikard Lesley</strong> successfully defended their thesis at Stockholm University — <em>Construction of a 21 cm Reionization Power Spectrum Emulators Utilising Transfer Learning</em>.",
    color:   "text-teal-400",
    bg:      "bg-teal-400/10"
  },
  {
    date:    "2026-04",
    display: "Apr 2026",
    type:    "Paper",
    text:    "Published in OJAp: <a href='https://arxiv.org/abs/2601.18784' target='_blank' class='text-cyan-400 hover:text-cyan-300 hover:underline transition-colors'>Baryonification III: An accurate analytical model for the DM PDF of FRBs</a>.",
    color:   "text-cyan-400",
    bg:      "bg-cyan-400/10"
  },
  {
    date:    "2026-04",
    display: "Apr 2026",
    type:    "Grant",
    text:    "Granted <strong>Stockholm University Special Educational Funding</strong> (PI) for the project <em>Development of a Privacy-Preserving AI-based Assessment Assistant for Higher Education</em>.",
    color:   "text-amber-400",
    bg:      "bg-amber-400/10"
  },
  {
    date:    "2026-03",
    display: "Mar 2026",
    type:    "Supervision",
    text:    "Supervising bachelor's student <strong>Suha Alam</strong> at Stockholm University — <em>Cosmic Expansion vs. Cosmic Dawn: A Bayesian Test of Dynamical Dark Energy</em>.",
    color:   "text-teal-400",
    bg:      "bg-teal-400/10"
  },
  {
    date:    "2026-03",
    display: "Mar 2026",
    type:    "Talk",
    text:    "Presented <strong>Probing Cosmic Reionization's Non-Gaussianity with 21-cm Fourier Phases</strong> at <a href='https://indico.skatelescope.org/event/1273/' target='_blank' class='text-cyan-400 hover:text-cyan-300 hover:underline transition-colors'>Cosmology in the Alps 2026</a>, Les Diablerets, Switzerland.",
    color:   "text-pink-400",
    bg:      "bg-pink-400/10"
  },
  {
    date:    "2026-02",
    display: "Feb 2026",
    type:    "Event",
    text:    "Organising committee member for the <a href='https://indico.chalmers.se/event/371/overview' target='_blank' class='text-cyan-400 hover:text-cyan-300 hover:underline transition-colors'>3rd National Sweden SKA Science Days</a>.",
    color:   "text-amber-400",
    bg:      "bg-amber-400/10"
  },
  {
    date:    "2025-12",
    display: "Dec 2025",
    type:    "Paper",
    text:    "Published in JCAP: <a href='https://arxiv.org/abs/2507.07892' target='_blank' class='text-cyan-400 hover:text-cyan-300 hover:underline transition-colors'>Baryonification: an alternative to hydrodynamical simulations for cosmological studies</a>.",
    color:   "text-cyan-400",
    bg:      "bg-cyan-400/10"
  },
  {
    date:    "2025-12",
    display: "Dec 2025",
    type:    "Paper",
    text:    "Published in MNRAS: <a href='https://arxiv.org/abs/2505.06350' target='_blank' class='text-cyan-400 hover:text-cyan-300 hover:underline transition-colors'>Mapping neutral islands during end stages of reionization</a>.",
    color:   "text-cyan-400",
    bg:      "bg-cyan-400/10"
  },
  {
    date:    "2025-11",
    display: "Nov 2025",
    type:    "Paper",
    text:    "Submitted to MNRAS: <a href='https://arxiv.org/abs/2511.11568' target='_blank' class='text-cyan-400 hover:text-cyan-300 hover:underline transition-colors'>Implicit inference of the reionization history with higher-order statistics</a>.",
    color:   "text-cyan-400",
    bg:      "bg-cyan-400/10"
  },
  {
    date:    "2025-11",
    display: "Nov 2025",
    type:    "Paper",
    text:    "Published in JCAP: <a href='https://arxiv.org/abs/2507.07991' target='_blank' class='text-cyan-400 hover:text-cyan-300 hover:underline transition-colors'>Baryonification II: constraining feedback</a>.",
    color:   "text-cyan-400",
    bg:      "bg-cyan-400/10"
  },
  {
    date:    "2025-09",
    display: "Sep 2025",
    type:    "Supervision",
    text:    "Started co-supervising PhD student Viktor Köhlin Lövfors at Stockholm University.",
    color:   "text-teal-400",
    bg:      "bg-teal-400/10"
  },
  {
    date:    "2025-09",
    display: "Sep 2025",
    type:    "Position",
    text:    "Started as Independent Researcher (PI) at Stockholm University, funded by Olle Engkvists Stiftelse.",
    color:   "text-emerald-400",
    bg:      "bg-emerald-400/10"
  },
];
