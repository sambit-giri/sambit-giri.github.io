// research_pillars.js — two research themes shown in the Research section
//
// Each entry:
//   id         : unique identifier
//   label      : short label shown as a tab
//   title      : full section heading
//   icon       : Lucide icon name
//   theme      : { text, border, shadow, bg } — Tailwind color classes
//   desc       : brief overview paragraph (shown above sub-topics)
//   subTopics  : array of { heading, desc, imagePath, paper: { title, authors, href } }

const researchPillars = [
    {
        id: "structures",
        label: "01 // COSMIC STRUCTURES",
        title: "Astrophysical Processes & Cosmic Structures",
        icon: "globe",
        theme: { text: "text-cyan-400", border: "border-cyan-500/50", shadow: "shadow-cyan-900/20", bg: "bg-cyan-500/10" },
        desc: "Two interconnected astrophysical processes reshape the large-scale structure of the universe: the reionization of the intergalactic medium in the early universe, and the redistribution of matter by baryonic feedback at late times.",
        subTopics: [
            {
                heading: "Epoch of Reionization & 21-cm Signal",
                desc: "The first stars and galaxies emitted energetic photons that heated and ionised the intergalactic medium, driving a phase transition that transformed the universe from neutral to fully ionised. I simulate this process using radiative-transfer codes such as BEoRN (Bubbles during the Epoch of Reionization Numerical Simulator), generating realistic 21-cm signal maps to interpret current LOFAR observations and develop analysis pipelines for the SKA.",
                imagePath: "images/Schaeffer2023_lightcone_default.png",
                paper: { title: "BEoRN: a fast and flexible framework to simulate the epoch of reionization and the cosmic dawn", authors: "Schaeffer, T., Giri, S. K., Schneider, A. (2023) • MNRAS", href: "https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.2942S/abstract" }
            },
            {
                heading: "Baryonic Feedback & Large-Scale Structure",
                desc: "Energetic feedback from AGN and supernovae redistributes gas across large scales, suppressing the matter power spectrum and biasing weak lensing surveys. We developed a baryonification framework that displaces dark-matter-only simulation particles to reproduce the observed distribution of gas and stars in a fraction of the time required by full hydrodynamical simulations, making baryonic corrections tractable at survey scale.",
                imagePath: "images/Schneider2025_baryonified_gas_stars.png",
                paper: { title: "Baryonification: An alternative to hydrodynamical simulations for cosmological studies", authors: "Schneider, A., Kovač, M., Bucko, J., Nicola, A., Reischke, R., Giri, S. K., et al. (2025) • JCAP", href: "https://arxiv.org/abs/2507.07892" }
            }
        ]
    },
    {
        id: "inference",
        label: "02 // BAYESIAN INFERENCE",
        title: "Data-Driven Cosmology & Inference",
        icon: "database",
        theme: { text: "text-indigo-400", border: "border-indigo-500/50", shadow: "shadow-indigo-900/20", bg: "bg-indigo-500/10" },
        desc: "Connecting high-fidelity simulations to real observations requires both fast emulators that make parameter space exploration tractable, and inference frameworks that extract the maximum information from complex, high-dimensional data.",
        subTopics: [
            {
                heading: "Machine Learning Emulators",
                desc: "I build machine learning emulators that replace expensive simulators in inference pipelines. BCemu reproduces the suppression of the matter power spectrum predicted by multiple independent hydrodynamical simulation suites, and is actively used in analyses for Euclid and other weak lensing surveys. I have also developed 21-cm power spectrum emulators that enabled the first direct Bayesian constraints on reionization parameters from LOFAR observations.",
                imagePath: "images/Giri2021_bcemu.png",
                paper: { title: "Emulation of baryonic effects on the matter power spectrum and constraints from galaxy cluster data", authors: "Giri, S. K., Schneider, A. (2021) • JCAP", href: "https://arxiv.org/abs/2108.08863" }
            },
            {
                heading: "Simulation-Based Inference",
                desc: "Traditional likelihood-based inference becomes intractable for the high-dimensional, non-Gaussian data from next-generation surveys. I build Simulation-Based Inference pipelines that learn the posterior directly from simulations, bypassing explicit likelihood computation entirely. This makes it feasible to constrain physics using higher-order statistics and topological descriptors — summary statistics that carry far more information than the standard power spectrum.",
                imagePath: "images/Cerardi2026_FoM.png",
                paper: { title: "Implicit inference of the reionization history with higher-order statistics of the 21-cm signal", authors: "Cerardi, N., Giri, S. K., et al. (2025) • MNRAS", href: "https://arxiv.org/abs/2511.11568" }
            }
        ]
    }
];
