// recent_work.js — figures shown in the hero banner
//
// Each entry:
//   path    : relative path to image (PNG/JPG) inside the images/ folder
//   heading : short caption describing the figure
//   paper   : author(s) + year + journal, shown as subtitle
//   href    : link to paper (arXiv or journal)
//   tags    : 1–2 topic tags (uses same TAG_STYLES colours as publications)
//   bg      : background colour behind the image (default: '#020617' dark)
//             use '#ffffff' for figures with a white background

const recentWorkFigures = [
  {
    path:    "images/Cerardi2026_FoM.png",
    heading: "Figure of merit comparing constraining power of summary statistics for upcoming SKA-Low data",
    paper:   "Cerardi, Giri et al. 2025, submitted to MNRAS",
    href:    "https://arxiv.org/abs/2511.11568",
    tags:    ['21cm', 'EoR', 'SBI'],
    bg:      '#ffffff'
  },
  {
    path:    "images/Schneider2025_baryonified_maps.png",
    heading: "Baryonification: transforming N-body dark matter fields to emulate hydrodynamical simulations at a fraction of the cost",
    paper:   "Schneider et al. 2025, JCAP, 2025(12), 043.",
    href:    "https://arxiv.org/abs/2507.07892",
    tags:    ['Feedback', 'Weak Lensing', 'LSS'],
    bg:      '#ffffff'
  },
  {
    path:    "images/Kovac2025_Sk_constraints.png",
    heading: "Constraints on matter power spectrum suppression due to baryonic feedback",
    paper:   "Kovač et al. 2025, JCAP, 2025(11), 046.",
    href:    "https://arxiv.org/abs/2511.11568",
    tags:    ['Feedback', 'Weak Lensing', 'LSS'],
    bg:      '#ffffff'
  },
];
