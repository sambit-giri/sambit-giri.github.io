// tools.js — software tools shown in the Research section
//
// Each entry:
//   title : tool name
//   desc  : description paragraph
//   tags  : array of tag strings
//   icon  : Lucide icon name
//   color : Tailwind text color class
//   link  : URL to the tool (GitHub or project page)

const tools = [
    { title: "BEoRN", desc: "Bubbles during the Epoch of Reionization Numerical Simulator (BEoRN) is a fast, flexible radiative-transfer framework that reproduces realistic reionization morphology suitable for statistical interpretation of measurement of EoR and cosmic dawn, such as the 21-cm signal.", tags: ["Simulations", "Python", "SBI", "Cosmic Dawn"], icon: "globe", color: "text-indigo-400", link: "https://github.com/sambit-giri/BEORN" },
    { title: "pyC2Ray", desc: "A highly accurate 3D ray-tracing radiative transfer code. While tools like BEoRN optimize for speed using 1D spherical symmetry, pyC2Ray performs full 3D radiative transfer. It leverages GPU acceleration to achieve speedups of several orders of magnitude compared to traditional CPU implementations.", tags: ["3D Radiative Transfer", "GPU", "C2Ray", "Cosmic Dawn"], icon: "cpu", color: "text-rose-400", link: "https://github.com/cosmic-reionization/pyC2Ray" },
    { title: "Tools21cm", desc: "A comprehensive Python package for 21-cm data analysis and observational pipelines. It enables end-to-end workflows: adding instrumental effects (noise, antenna gain errors) and foregrounds, followed by robust signal extraction. It features advanced summary statistics, including power spectra, image-based structure identification (ionized bubbles), and topological metrics like bubble size distributions, Betti numbers, and Euler characteristics. Widely adopted by the global 21-cm community and SKAO scientists.", tags: ["Python", "21cm", "Topology", "Open Source"], icon: "telescope", color: "text-cyan-400", link: "https://github.com/sambit-giri/tools21cm" },
    { title: "BCemu", desc: "A machine-learning emulator predicting matter power spectrum suppression caused by baryonic feedback. Energetic processes from galaxies, such as active galactic nuclei (AGN) jets and supernovae, redistribute gas on massive scales—a crucial effect that can severely bias weak-lensing measurements if not accurately modeled. BCemu provides fast, robust predictions for these effects and is being extended into JAX-based differentiable frameworks for next-generation surveys.", tags: ["Machine Learning", "JAX", "Weak Lensing", "N-body"], icon: "database", color: "text-emerald-400", link: "https://github.com/sambit-giri/BCemu" }
];
