// collaborations.js — research collaborations shown in the About section
//
// Each entry:
//   name   : short acronym/name
//   full   : full name of the collaboration
//   role   : your role within the collaboration
//   desc   : description paragraph
//   color  : Tailwind text color class
//   border : Tailwind border color class
//   bg     : Tailwind bg color class
//   href   : URL to the collaboration's website

const collaborations = [
    { name: "SKA", full: "Square Kilometre Array", role: "Key Member — EoR Science Working Group", desc: "My work focuses on producing high-fidelity cosmological simulations of the Epoch of Reionization and developing advanced data analysis methods. By building robust statistical frameworks, I aim to connect the complex, contaminated radio data expected from the SKA directly to theoretical models.", color: "text-indigo-400", border: "border-indigo-500/30", bg: "bg-indigo-500/10", href: "https://www.skao.int/" },
    { name: "LOFAR", full: "Low Frequency Array", role: "Core Member — EoR Key Science Project", desc: "As a precursor to the SKA, LOFAR provides the current best data to test our theoretical models and Bayesian inference frameworks. By analyzing LOFAR's observations today, I am actively refining and scaling these methods using machine learning to handle the massive, highly informative datasets the SKA will soon produce.", color: "text-emerald-400", border: "border-emerald-500/30", bg: "bg-emerald-500/10", href: "https://www.lofar.eu/" },
    { name: "Euclid", full: "ESA Euclid Space Mission", role: "Member — Cosmological Simulations Working Group", desc: "I am actively involved in understanding baryonic feedback effects, which are highly degenerate with the imprints of dark matter models on the weak lensing signal. To address this, we developed a fast baryonification framework, for which I created BCemu—a JAX-based neural emulator designed to rapidly model these feedback impacts on the matter power spectrum.", color: "text-amber-400", border: "border-amber-500/30", bg: "bg-amber-500/10", href: "https://www.euclid-ec.org/" },
    { name: "Roman", full: "Nancy Grace Roman Space Telescope", role: "Member — Cosmic Dawn Group", desc: "I develop theoretical models and observation strategies to probe reionization physics and test dark matter models with Roman deep-field surveys at high redshift. Crucially, I am exploring how cross-correlation and synergy studies between Roman and the SKA will provide a much more comprehensive understanding of the cosmic dawn.", color: "text-pink-400", border: "border-pink-500/30", bg: "bg-pink-500/10", href: "https://roman.gsfc.nasa.gov/" },
    { name: "SEarCH", full: "", role: "Founder & Lead", desc: "A cross-disciplinary collaboration linking astrophysicists and data scientists across Europe. We aim to develop improved Simulation-Based Inference (SBI) methods for cosmological data analysis, specifically designed to handle the massive datasets that will be produced by the SKAO.", color: "text-cyan-400", border: "border-cyan-500/30", bg: "bg-cyan-500/10", href: "#" }
];
