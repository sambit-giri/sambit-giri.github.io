// papers.js — Edit this file to update the publication list on the website.
//
// Each entry has four fields:
//   cite     : Full citation string. "Giri, S. K." is auto-bolded in the display.
//   href     : URL to the paper (arXiv, ADS, or journal). Use "#" if unknown.
//   selected : true  → shown in the default "Selected" tab (your key contributions)
//              false → shown only in the "All" tab
//   tags     : Up to 3 topic tags (max 3 are shown). Available tags:
//              '21cm', 'EoR', 'Cosmic Dawn', 'Dark Matter', 'LSS', 'Feedback',
//              'Weak Lensing', 'ML/AI', 'SBI', 'Pattern Recognition',
//              'Topology', 'Cosmology', 'FRBs', 'Software',
//              'First Stars', 'Early Galaxies'
//
// ADD NEW PAPERS AT THE TOP of the relevant year block.

const papers = [

  // ── 2026 ──────────────────────────────────────────────────────────────────
  {
    cite: "Torkamani, M., Reischke, R., Kovač, M., Nicola, A., Bucko, J., Refregier, A., Giri, S. K., Schneider, A., Hagstotz, S., 2026. Baryonification III: An accurate analytical model for the dispersion measure probability density function of fast radio bursts. Submitted to OJAp.",
    href: "https://arxiv.org/abs/2601.18784",
    selected: false,
    tags: ['FRBs', 'Feedback', 'Cosmology']
  },
  {
    cite: "Schwandt, T. P., Georgiev, I., Giri, S. K., Mellema, G., Iliev, I. T., 2026. Impact of anisotropic photon emission from sources during the epoch of reionisation. Accepted by MNRAS.",
    href: "https://arxiv.org/abs/2505.02716",
    selected: true,
    tags: ['21cm', 'EoR', 'Early Galaxies']
  },

  // ── 2025 ──────────────────────────────────────────────────────────────────
  {
    cite: "Cerardi, N., Giri, S. K., Bianco, M., Piras, D., de Salis, E., De Santis, M., Selcuk-Simsek, M., Denzel, P., et al., 2025. SEarCH: Constraining the reionisation history with higher-order statistics of the 21-cm signal. Submitted to MNRAS.",
    href: "https://arxiv.org/abs/2511.11568",
    selected: true,
    tags: ['21cm', 'EoR', 'SBI']
  },
  {
    cite: "Kovač, M., Nicola, A., Bucko, J., Schneider, A., Reischke, R., Giri, S. K., Teyssier, R., Schaller, M., Schaye, J., 2025. Baryonification II: Constraining feedback with X-ray and kinematic Sunyaev-Zel'dovich observations. JCAP, 2025(11), 046.",
    href: "https://arxiv.org/abs/2507.07991",
    selected: true,
    tags: ['Feedback', 'Weak Lensing', 'LSS']
  },
  {
    cite: "Schneider, A., Kovač, M., Bucko, J., Nicola, A., Reischke, R., Giri, S. K., et al., 2025. Baryonification: An alternative to hydrodynamical simulations for cosmological studies. JCAP, 2025(12), 043.",
    href: "https://arxiv.org/abs/2507.07892",
    selected: true,
    tags: ['Feedback', 'LSS', 'Cosmology']
  },
  {
    cite: "Giri, S. K., Kakiichi, K., Bianco, M., Meerburg, P. D., 2025. Mapping neutral islands during end stages of reionization with photometric intergalactic medium tomography. MNRAS, 544(4), 3076–3093.",
    href: "https://arxiv.org/abs/2505.06350",
    selected: true,
    tags: ['21cm', 'EoR', 'Pattern Recognition']
  },
  {
    cite: "Ceccotti, E., Offringa, A. R., Mertens, F. G., Koopmans, L. V. E., Munshi, S., Chege, J. K., Acharya, A., Giri, S. K., et al., 2025. First upper limits on the 21-cm signal power spectrum of neutral hydrogen at z=9.16 from the LOFAR 3C196 field. MNRAS, 544(1), 1255–1283.",
    href: "https://arxiv.org/abs/2504.18534",
    selected: false,
    tags: ['21cm', 'EoR']
  },
  {
    cite: "Bonaldi, A., Hartley, P., Braun, R., Purser, S., Acharya, A., Giri, S. K., et al., 2025. Square Kilometre Array Science Data Challenge 3a: foreground removal for an EoR experiment. MNRAS, 543(2), 1092–1119.",
    href: "https://arxiv.org/abs/2503.11740",
    selected: false,
    tags: ['21cm', 'EoR', 'ML/AI']
  },
  {
    cite: "Giri, S. K., 2025. Astronomy Calc: A python toolkit for teaching Astronomical Calculations and Data Analysis methods. JOSE, 8(87), 261.",
    href: "https://arxiv.org/abs/2501.05491",
    selected: true,
    tags: ['Software', 'Cosmology']
  },
  {
    cite: "Ghara, R., Zaroubi, S., Ciardi, B., Mellema, G., Giri, S. K., Mertens, F. G., et al., 2025. Constraints on the state of the IGM at z~8–10 using redshifted 21-cm observations with LOFAR. A&A, 699, A109.",
    href: "https://arxiv.org/abs/2505.00373",
    selected: true,
    tags: ['21cm', 'EoR']
  },
  {
    cite: "Mertens, F. G., Mevius, M., Koopmans, L. V. E., Offringa, A. R., Zaroubi, S., Acharya, A., Giri, S. K., et al., 2025. Deeper multi-redshift upper limits on the Epoch of Reionization 21-cm signal power spectrum from LOFAR between z=8.3 and z=10.1. A&A, 698, A186.",
    href: "https://arxiv.org/abs/2503.05576",
    selected: false,
    tags: ['21cm', 'EoR', 'Cosmology']
  },
  {
    cite: "Georgiev, I., Mellema, G., Giri, S. K., 2025. The forest at EndEoR: The effect of Lyman Limit Systems on the End of Reionisation. MNRAS, 536(4), 3689–3706.",
    href: "https://ui.adsabs.harvard.edu/abs/2025MNRAS.536.3689G/abstract",
    selected: true,
    tags: ['21cm', 'EoR', 'Cosmology']
  },
  {
    cite: "Nebrin, O., Smith, A., Lorinc, K., Hörnquist, J., Larson, A., Mellema, G., Giri, S. K., 2025. Lyman-α feedback prevails at Cosmic Dawn: Implications for the first galaxies, stars, and star clusters. MNRAS, 537(2), 1646–1687.",
    href: "https://arxiv.org/abs/2409.19288",
    selected: false,
    tags: ['Cosmic Dawn', 'First Stars', 'Early Galaxies']
  },
  {
    cite: "Gao, L.-Y., Koopmans, L. V. E., Mertens, F. G., Munshi, S., Li, Y., Brackenhoff, S. A., Giri, S. K., et al., 2025. Extracting the Epoch of Reionization Signal with 3D U-Net Neural Networks Using a Data-driven Systematic Effect Model. ApJ, 988(1), 84.",
    href: "https://arxiv.org/abs/2412.16853",
    selected: false,
    tags: ['21cm', 'EoR', 'ML/AI']
  },
  {
    cite: "Acharya, A., Ma, Q., Giri, S. K., Ciardi, B., Ghara, R., Mellema, G., Zaroubi, S., Hothi, I., Iliev, I. T., Koopmans, L. V. E., Bianco, M., 2025. Exploring the effect of different cosmologies on the Epoch of Reionization 21-cm signal with POLAR. MNRAS, 543(2), 1058–1078.",
    href: "https://ui.adsabs.harvard.edu/abs/2025MNRAS.543.1058A/abstract",
    selected: true,
    tags: ['21cm', 'EoR', 'Cosmology']
  },
  {
    cite: "Bianco, M., Giri, S. K., Sharma, R., Chen, T., Krishna, S. P., Finlay, C., Nistane, V., Denzel, P., De Santis, M., Ghorbel, H., 2025. Deep learning approach for identification of HII regions during reionization in 21-cm observations III: image recovery. MNRAS, 541(1), 234–250.",
    href: "https://ui.adsabs.harvard.edu/abs/2025MNRAS.541..234B/abstract",
    selected: true,
    tags: ['21cm', 'EoR', 'ML/AI']
  },
  {
    cite: "Choudhury, M., Ghara, R., Zaroubi, S., Ciardi, B., Koopmans, L. V. E., Mellema, G., Shaw, A. K., Acharya, A., Iliev, I. T., Ma, Q.-B., Giri, S. K., 2025. Inferring IGM parameters from the redshifted 21-cm Power Spectrum using Artificial Neural Networks. JCAP, 2025(06), 32.",
    href: "https://arxiv.org/abs/2407.03523",
    selected: false,
    tags: ['21cm', 'EoR', 'ML/AI']
  },
  {
    cite: "de Salis, E., De Santis, M., Piras, D., Giri, S. K., Bianco, M., et al., 2025. Exploring the Early Universe with Deep Learning. Progress in Artificial Intelligence, EPIA 2025. Lecture Notes in Computer Science, vol 16121, Springer.",
    href: "https://arxiv.org/abs/2509.22018",
    selected: true,
    tags: ['Cosmic Dawn', 'ML/AI', 'SBI']
  },
  {
    cite: "Euclid Collaboration, Lesgourgues, J., Schwagereit, J., Bucko, J., Parimbelli, G., Giri, S. K., et al., 2025. Euclid preparation. LVI. Sensitivity to non-standard particle dark matter model. A&A, 693, A249.",
    href: "https://arxiv.org/abs/2406.18274",
    selected: true,
    tags: ['Dark Matter', 'Weak Lensing', 'Cosmology']
  },
  {
    cite: "Euclid Collaboration, ..., Giri, S. K., et al., 2025. Euclid. I. Overview of the Euclid mission. A&A, 697, A1.",
    href: "https://arxiv.org/abs/2405.13491",
    selected: false,
    tags: ['LSS', 'Cosmology', 'Weak Lensing']
  },

  // ── 2024 ──────────────────────────────────────────────────────────────────
  {
    cite: "Schaeffer, T., Giri, S. K., Schneider, A., 2024. Testing common approximations to predict the 21cm signal at the Epoch of Reionization and Cosmic Dawn. PRD, 110, 023543.",
    href: "https://ui.adsabs.harvard.edu/abs/2024PhRvD.110b3543S/abstract",
    selected: true,
    tags: ['21cm', 'EoR', 'Cosmic Dawn']
  },
  {
    cite: "Giri, S. K., Bianco, M., Schaeffer, T., Iliev, I. T., Mellema, G., Schneider, A., 2024. The 21-cm signal during the end stages of reionization. MNRAS, 533(2), 2364–2378.",
    href: "https://ui.adsabs.harvard.edu/abs/2024MNRAS.533.2364G/abstract",
    selected: true,
    tags: ['21cm', 'EoR', 'Topology']
  },
  {
    cite: "Dayal, P., Giri, S. K., 2024. Warm dark matter constraints from the JWST. MNRAS, 528(2), 2784–2789.",
    href: "https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.2784D/abstract",
    selected: true,
    tags: ['Dark Matter', 'Early Galaxies', 'Cosmology']
  },
  {
    cite: "Ghara, R., Shaw, A. K., Zaroubi, S., Ciardi, B., Mellema, G., Koopmans, L. V. E., Acharya, A., Choudhury, M., Giri, S. K., Iliev, I. T., Ma, Q., Mertens, F. G., 2024. Probing the intergalactic medium during the Epoch of Reionization using 21-cm signal power spectra. A&A, 687, A252.",
    href: "https://arxiv.org/abs/2404.11686",
    selected: false,
    tags: ['21cm', 'EoR']
  },
  {
    cite: "He, Y., Giri, S. K., Sharma, R., Mtchedlidze, S., Georgiev, I., 2024. Inverse Gertsenshtein effect as a probe of high-frequency gravitational waves. JCAP, 2024(05), 051.",
    href: "https://ui.adsabs.harvard.edu/abs/2024JCAP...05..051H/abstract",
    selected: true,
    tags: ['Cosmology', 'Dark Matter']
  },
  {
    cite: "Hirling, P., Bianco, M., Giri, S. K., Iliev, I. T., Mellema, G., Kneib, J.-P., 2024. pyC2Ray: A flexible and GPU-accelerated Radiative Transfer Framework for Simulating the Cosmic Epoch of Reionization. A&C, 48, 100861.",
    href: "https://ui.adsabs.harvard.edu/abs/2024A%26C....4800861H/abstract",
    selected: true,
    tags: ['Software', 'EoR', 'Cosmic Dawn']
  },
  {
    cite: "Peters, F. H., Schneider, A., Bucko, J., Giri, S. K., Parimbelli, G., 2024. Constraining Hot Dark Matter Sub-Species with Weak Lensing and the Cosmic Microwave Background Radiation. A&A, 687, A161.",
    href: "https://arxiv.org/abs/2309.03865",
    selected: false,
    tags: ['Dark Matter', 'Weak Lensing', 'Cosmology']
  },
  {
    cite: "Bucko, J., Giri, S. K., Peters, F. H., Schneider, A., 2024. Probing the two-body decaying dark matter scenario with weak lensing and the cosmic microwave background. A&A, 683, A152.",
    href: "https://ui.adsabs.harvard.edu/abs/2024A%26A...683A.152B/abstract",
    selected: true,
    tags: ['Dark Matter', 'Weak Lensing', 'Cosmology']
  },
  {
    cite: "Bianco, M., Giri, S. K., Prelogović, D., Chen, T., Mertens, F. G., Tolley, E., Mesinger, A., Kneib, J.-P., 2024. Deep learning approach for identification of HII regions during reionization in 21-cm observations II: foreground contamination. MNRAS, 528(3), 5212–5230.",
    href: "https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.5212B/abstract",
    selected: true,
    tags: ['21cm', 'EoR', 'ML/AI']
  },

  // ── 2023 ──────────────────────────────────────────────────────────────────
  {
    cite: "Schaeffer, T., Giri, S. K., Schneider, A., 2023. BEORN: a fast and flexible framework to simulate the epoch of reionization and the cosmic dawn. MNRAS, 526(2), 2942–2959.",
    href: "https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.2942S/abstract",
    selected: true,
    tags: ['Software', 'EoR', 'Cosmic Dawn']
  },
  {
    cite: "Acharya, A., Mertens, F., Ciardi, B., Ghara, R., Koopmans, L. V. E., Giri, S. K., Hothi, I., Ma, Q.-B., Mellema, G., Munshi, S., 2023. 21-cm Signal from the Epoch of Reionization: A Machine Learning upgrade to Foreground Removal with Gaussian Process Regression. MNRAS, 527(3), 7835–7846.",
    href: "https://arxiv.org/abs/2311.16633",
    selected: false,
    tags: ['21cm', 'EoR', 'ML/AI']
  },
  {
    cite: "Nebrin, O., Giri, S. K., Mellema, G., 2023. Starbursts in low-mass haloes at Cosmic Dawn. I. The critical halo mass for star formation. MNRAS, 524(2), 2290–2311.",
    href: "https://ui.adsabs.harvard.edu/abs/2023MNRAS.524.2290N/abstract",
    selected: true,
    tags: ['Cosmic Dawn', 'First Stars', 'EoR']
  },
  {
    cite: "Schneider, A., Schaeffer, T., Giri, S. K., 2023. Cosmological forecast of the 21-cm power spectrum using the halo model of reionization. PRD, 108, 043030.",
    href: "https://ui.adsabs.harvard.edu/abs/2023PhRvD.108d3030S/abstract",
    selected: true,
    tags: ['21cm', 'EoR', 'SBI']
  },
  {
    cite: "Bucko, J., Giri, S. K., Schneider, A., 2023. Constraining dark matter decays with cosmic microwave background and weak lensing shear observations. A&A, 672, A157.",
    href: "https://ui.adsabs.harvard.edu/abs/2023A%26A...672A.157B/abstract",
    selected: true,
    tags: ['Dark Matter', 'Weak Lensing', 'Cosmology']
  },
  {
    cite: "Giri, S. K., Schneider, A., Maion, F., Angulo, R. E., 2023. Suppressing variance in 21-cm signal simulations during reionization. A&A, 669, A6.",
    href: "https://ui.adsabs.harvard.edu/abs/2023A%26A...669A...6G/abstract",
    selected: true,
    tags: ['21cm', 'EoR', 'Cosmic Dawn']
  },
  {
    cite: "Gan, H., Mertens, F. G., Koopmans, L. V. E., Offringa, A. R., Mevius, M., Pandey, V. N., Giri, S. K., et al., 2023. Assessing the impact of two independent direction-dependent calibration algorithms on the LOFAR 21-cm signal power spectrum. A&A, 669, A20.",
    href: "https://arxiv.org/abs/2209.07854",
    selected: false,
    tags: ['21cm', 'EoR', 'Cosmic Dawn']
  },
  {
    cite: "Giri, S. K., Schneider, A., 2023. BCemu: Emulator for baryonic feedback effects on the matter power spectrum. ASCL, ascl:2308.010.",
    href: "https://ui.adsabs.harvard.edu/abs/2023ascl.soft08010G/abstract",
    selected: true,
    tags: ['Software', 'Feedback', 'Cosmology']
  },
  {
    cite: "Giri, S. K., Nebrin, O., Mellema, G., 2023. Imprints of early Universe galaxy formation on the 21-cm signal at cosmic dawn. The URSI GASS 2023 Proceedings.",
    href: "https://www.ursi.org/proceedings/procGA23/papers/3549.pdf",
    selected: true,
    tags: ['Cosmic Dawn', 'Early Galaxies', '21cm']
  },
  {
    cite: "Hirling, P., Bianco, M., Giri, S. K., Iliev, I. T., Mellema, G., Kneib, J.-P., 2023. pyC2Ray: Python interface to C2Ray with GPU acceleration. ASCL, ascl:2312.025.",
    href: "https://ui.adsabs.harvard.edu/abs/2023ascl.soft12025H/abstract",
    selected: false,
    tags: ['Software', 'EoR', 'Cosmic Dawn']
  },
  {
    cite: "Mellema, G., Iliev, I. T., Giri, S. K., Bianco, M., 2023. C2-Ray3Dm: 3D version of C2-Ray for multiple sources, hydrogen only. ASCL, ascl:2312.023.",
    href: "https://ui.adsabs.harvard.edu/abs/2023ascl.soft12023M/abstract",
    selected: false,
    tags: ['Software', 'EoR', 'Cosmic Dawn']
  },
  {
    cite: "Bianco, M., Iliev, I. T., Ahn, K., Giri, S. K., Mao, Y., Park, H., Shapiro, P. R., 2023. Subgrid-Clumping: Clumping factor for large low-resolution N-body simulations. ASCL, ascl:2306.050.",
    href: "https://ui.adsabs.harvard.edu/abs/2023ascl.soft06050B/abstract",
    selected: false,
    tags: ['Software', 'EoR', 'Cosmic Dawn']
  },

  // ── 2022 ──────────────────────────────────────────────────────────────────
  {
    cite: "Gan, H., Koopmans, L. V. E., Mertens, F. G., Mevius, M., Offringa, A. R., Ciardi, B., Giri, S. K., et al., 2022. Statistical analysis of the causes of excess variance in the 21 cm signal power spectra obtained with LOFAR. A&A, 663, A9.",
    href: "https://arxiv.org/abs/2203.02345",
    selected: false,
    tags: ['21cm', 'EoR', 'Cosmic Dawn']
  },
  {
    cite: "Gehlot, B. K., Koopmans, L. V. E., Offringa, A. R., Gan, H., Ghara, R., Giri, S. K., et al., 2022. Degree-Scale Galactic Radio Emission at 122 MHz around the North Celestial Pole with LOFAR-AARTFAAC. A&A, 662, A97.",
    href: "https://arxiv.org/abs/2112.00721",
    selected: false,
    tags: ['21cm', 'Cosmic Dawn']
  },
  {
    cite: "Mevius, M., Mertens, F., Koopmans, L. V. E., Offringa, A. R., Yatawatta, S., Giri, S. K., et al., 2022. A numerical study of 21-cm signal suppression and noise increase in direction-dependent calibration. MNRAS, 509(3), 3693–3702.",
    href: "https://arxiv.org/abs/2111.02537",
    selected: false,
    tags: ['21cm', 'EoR', 'Cosmic Dawn']
  },
  {
    cite: "Giri, S. K., Schneider, A., 2022. Imprints of fermionic and bosonic mixed dark matter on the 21-cm signal at cosmic dawn. PRD, 105, 083011.",
    href: "https://ui.adsabs.harvard.edu/abs/2022PhRvD.105h3011G/abstract",
    selected: true,
    tags: ['Dark Matter', '21cm', 'Cosmic Dawn']
  },
  {
    cite: "Georgiev, I., Mellema, G., Giri, S. K., Mondal, R., 2022. The large-scale 21-cm power spectrum from reionization. MNRAS, 513(4), 5109–5124.",
    href: "https://ui.adsabs.harvard.edu/abs/2022MNRAS.513.5109G/abstract",
    selected: true,
    tags: ['21cm', 'EoR', 'Cosmology']
  },
  {
    cite: "Schneider, A., Giri, S. K., Amodeo, S., Refregier, A., 2022. Constraining baryonic feedback and cosmology with weak-lensing, X-ray, and kinematic Sunyaev-Zel'dovich observations. MNRAS, 514(3), 3802–3814.",
    href: "https://ui.adsabs.harvard.edu/abs/2022MNRAS.514.3802S/abstract",
    selected: true,
    tags: ['Feedback', 'Weak Lensing', 'Cosmology']
  },

  // ── 2021 ──────────────────────────────────────────────────────────────────
  {
    cite: "Hothi, I., Chapman, E., Pritchard, J. R., Mertens, F. G., Koopmans, L. V. E., Ciardi, B., Giri, S. K., et al., 2021. Comparing Foreground Removal Techniques for Recovery of the LOFAR-EOR 21cm Power Spectrum. MNRAS, 500(2), 2264–2277.",
    href: "https://arxiv.org/abs/2011.01284",
    selected: false,
    tags: ['21cm', 'EoR', 'Cosmic Dawn']
  },
  {
    cite: "Hubert, J., Schneider, A., Potter, D., Stadel, J., Giri, S. K., 2021. Decaying dark matter: simulations and weak-lensing forecast. JCAP, 2021(10), 040.",
    href: "https://arxiv.org/abs/2104.07675",
    selected: false,
    tags: ['Dark Matter', 'Weak Lensing', 'Cosmology']
  },
  {
    cite: "Giri, S. K., Schneider, A., 2021. Emulation of baryonic effects on the matter power spectrum and constraints from galaxy cluster data. JCAP, 2021(12), 046.",
    href: "https://ui.adsabs.harvard.edu/abs/2021JCAP...12..046G/abstract",
    selected: true,
    tags: ['Feedback', 'Cosmology', 'ML/AI']
  },
  {
    cite: "Parimbelli, G., Scelfo, G., Giri, S. K., Schneider, A., Archidiacono, M., Camera, S., Viel, M., 2021. Mixed dark matter: matter power spectrum and halo mass function. JCAP, 2021(12), 044.",
    href: "https://ui.adsabs.harvard.edu/abs/2021JCAP...12..044P/abstract",
    selected: true,
    tags: ['Dark Matter', 'LSS', 'Cosmology']
  },
  {
    cite: "Ghara, R., Giri, S. K., Ciardi, B., Mellema, G., Zaroubi, S., 2021. Constraining the state of the intergalactic medium during the Epoch of Reionization using MWA 21-cm signal observations. MNRAS, 503(3), 4551–4562.",
    href: "https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.4551G/abstract",
    selected: true,
    tags: ['21cm', 'EoR', 'Cosmic Dawn']
  },
  {
    cite: "Bianco, M., Iliev, I. T., Ahn, K., Giri, S. K., Mao, Y., Park, H., Shapiro, P. R., 2021. The impact of inhomogeneous subgrid clumping on cosmic reionization II: modelling stochasticity. MNRAS, 504(2), 2443–2460.",
    href: "https://arxiv.org/abs/2101.01712",
    selected: false,
    tags: ['EoR', '21cm', 'Cosmology']
  },
  {
    cite: "Bianco, M., Giri, S. K., Iliev, I. T., Mellema, G., 2021. Deep learning approach for identification of HII regions during reionization in 21-cm observations. MNRAS, 505(3), 3982–3997.",
    href: "https://ui.adsabs.harvard.edu/abs/2021MNRAS.505.3982B/abstract",
    selected: true,
    tags: ['21cm', 'EoR', 'ML/AI']
  },
  {
    cite: "Giri, S. K., Mellema, G., 2021. Measuring the topology of reionization with Betti numbers. MNRAS, 505(2), 1863–1877.",
    href: "https://ui.adsabs.harvard.edu/abs/2021MNRAS.505.1863G/abstract",
    selected: true,
    tags: ['21cm', 'EoR', 'Topology']
  },
  {
    cite: "Schneider, A., Giri, S. K., Mirocha, J., 2021. A halo model approach for the 21-cm power spectrum at cosmic dawn. PRD, 103(8), 083025.",
    href: "https://ui.adsabs.harvard.edu/abs/2021PhRvD.103h3025S/abstract",
    selected: true,
    tags: ['21cm', 'Cosmic Dawn', 'EoR']
  },
  {
    cite: "Ross, H. E., Giri, S. K., Mellema, G., Dixon, K. L., Ghara, R., Iliev, I. T., 2021. Redshift-space distortions in simulations of the 21-cm signal from the cosmic dawn. MNRAS, 506(3), 3717–3733.",
    href: "https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.3717R/abstract",
    selected: true,
    tags: ['21cm', 'Cosmic Dawn', 'EoR']
  },
  {
    cite: "Greig, B., Mesinger, A., Koopmans, L. V. E., Ciardi, B., Mellema, G., Zaroubi, S., Giri, S. K., et al., 2021. Interpreting LOFAR 21-cm signal upper limits at z ~ 9.1 in the context of high-z galaxy and reionisation observations. MNRAS, 501(1), 1–13.",
    href: "https://arxiv.org/abs/2006.03203",
    selected: false,
    tags: ['21cm', 'EoR', 'Cosmic Dawn']
  },

  // ── 2020 ──────────────────────────────────────────────────────────────────
  {
    cite: "Mondal, R., Fialkov, A., Fling, C., Iliev, I. T., Barkana, R., Ciardi, B., Mellema, G., Zaroubi, S., Giri, S. K., et al., 2020. Tight Constraints on the Excess Radio Background at z=9.1 from LOFAR. MNRAS, 498(3), 4178–4191.",
    href: "https://arxiv.org/abs/2004.00678",
    selected: false,
    tags: ['21cm', 'EoR', 'Cosmology']
  },
  {
    cite: "Zackrisson, E., Majumdar, S., Mondal, R., Binggeli, C., Sahlén, M., Choudhury, T. R., Ciardi, B., Dayal, P., Giri, S. K., et al., 2020. Bubble mapping with the Square Kilometer Array – I. Detecting galaxies with Euclid, JWST, WFIRST and ELT within ionized bubbles at z>6. MNRAS, 493(1), 855–870.",
    href: "https://arxiv.org/abs/1905.00437",
    selected: false,
    tags: ['EoR', 'Early Galaxies', '21cm']
  },
  {
    cite: "Giri, S. K., Mellema, G., Jensen, H., 2020. Tools21cm: A python package to analyse the large-scale 21-cm signal from the Epoch of Reionization and Cosmic Dawn. JOSS, 5(52), 2363.",
    href: "https://ui.adsabs.harvard.edu/abs/2020JOSS....5.2363G/abstract",
    selected: true,
    tags: ['Software', '21cm', 'EoR']
  },
  {
    cite: "Ghara, R., Giri, S. K., Mellema, G., Ciardi, B., Zaroubi, S., Iliev, I. T., Koopmans, L. V. E., et al., 2020. Constraining the intergalactic medium at z ≈ 9.1 using the LOFAR Epoch of Reionization observations. MNRAS, 493(4), 4728–4747.",
    href: "https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.4728G/abstract",
    selected: true,
    tags: ['21cm', 'EoR', 'Cosmic Dawn']
  },
  {
    cite: "Mertens, F. G., Mevius, M., Koopmans, L. V. E., Offringa, A. R., Mellema, G., Zaroubi, S., Giri, S. K., et al., 2020. Improved upper limits on the 21-cm signal power spectrum of neutral hydrogen at z ≈ 9.1 from LOFAR. MNRAS, 493(2), 1662–1685.",
    href: "https://arxiv.org/abs/2002.07196",
    selected: false,
    tags: ['21cm', 'EoR', 'Cosmic Dawn']
  },
  {
    cite: "Giri, S. K., Zackrisson, E., Binggeli, C., Pelckmans, K., Cubo, R., 2020. Identifying reionization-epoch galaxies with extreme levels of Lyman continuum leakage in James Webb Space Telescope surveys. MNRAS, 491(4), 5277–5286.",
    href: "https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.5277G/abstract",
    selected: true,
    tags: ['EoR', 'Early Galaxies', 'ML/AI']
  },

  // ── 2019 ──────────────────────────────────────────────────────────────────
  {
    cite: "Giri, S. K., D'Aloisio, A., Mellema, G., Komatsu, E., Ghara, R., Majumdar, S., 2019. Position-dependent power spectra of the 21-cm signal from the epoch of reionization. JCAP, 2019(02), 058.",
    href: "https://ui.adsabs.harvard.edu/abs/2019JCAP...02..058G/abstract",
    selected: true,
    tags: ['21cm', 'EoR', 'Cosmology']
  },
  {
    cite: "Giri, S. K., Mellema, G., Aldheimer, T., Dixon, K. L., Iliev, I. T., 2019. Neutral island statistics during reionization from 21-cm tomography. MNRAS, 489(2), 1590–1605.",
    href: "https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.1590G/abstract",
    selected: true,
    tags: ['21cm', 'EoR', 'Pattern Recognition']
  },
  {
    cite: "Watkinson, C. A., Giri, S. K., Ross, H. E., Dixon, K. L., Iliev, I. T., Mellema, G., Pritchard, J. R., 2019. The 21cm bispectrum as a probe of non-Gaussianities due to X-ray heating. MNRAS, 482(2), 2653–2669.",
    href: "https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.2653W/abstract",
    selected: true,
    tags: ['21cm', 'Cosmic Dawn', 'Early Galaxies']
  },

  // ── 2018 ──────────────────────────────────────────────────────────────────
  {
    cite: "Giri, S. K., Mellema, G., Ghara, R., 2018. Optimal identification of HII regions during reionization in 21-cm observations. MNRAS, 479(4), 5596–5611.",
    href: "https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.5596G/abstract",
    selected: true,
    tags: ['21cm', 'EoR', 'Pattern Recognition']
  },
  {
    cite: "Ghara, R., Mellema, G., Giri, S. K., Choudhury, T. R., Datta, K. K., Majumdar, S., 2018. Prediction of the 21-cm signal from reionization: comparison between 3D and 1D radiative transfer schemes. MNRAS, 476(2), 1741–1755.",
    href: "https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.1741G/abstract",
    selected: true,
    tags: ['EoR', '21cm', 'Early Galaxies']
  },
  {
    cite: "Giri, S. K., Mellema, G., Dixon, K. L., Iliev, I. T., 2018. Bubble size statistics during reionization from 21-cm tomography. MNRAS, 473(3), 2949–2964.",
    href: "https://ui.adsabs.harvard.edu/abs/2018MNRAS.473.2949G/abstract",
    selected: true,
    tags: ['21cm', 'EoR', 'Early Galaxies']
  },

  // ── 2017 / Proceedings ────────────────────────────────────────────────────
  {
    cite: "Giri, S. K., Zackrisson, E., Binggeli, C., Pelckmans, K., Cubo, R., Mellema, G., 2017. Constraining Lyman continuum escape using Machine Learning. Proceedings of the IAU, 12(S333), 254–258.",
    href: "https://ui.adsabs.harvard.edu/abs/2018IAUS..333..254G/abstract",
    selected: true,
    tags: ['Early Galaxies', 'ML/AI', 'EoR']
  },
  {
    cite: "Mellema, G., Giri, S. K., Ghara, R., 2017. Analysis of 21-cm tomographic data. Proceedings of the IAU, 12(S333), 26–29.",
    href: "https://ui.adsabs.harvard.edu/abs/2018IAUS..333...26M/abstract",
    selected: true,
    tags: ['21cm', 'EoR', 'Pattern Recognition']
  },
  {
    cite: "Ghara, R., Choudhury, T. R., Datta, K. K., Mellema, G., Choudhuri, S., Majumdar, S., Giri, S. K., 2017. Prospects of detection of the first sources with SKA using matched filters. Proceedings of the IAU, 12(S333), 122–125.",
    href: "https://ui.adsabs.harvard.edu/abs/2018IAUS..333..122G/abstract",
    selected: false,
    tags: ['EoR', 'Cosmic Dawn', 'Pattern Recognition']
  }

];
