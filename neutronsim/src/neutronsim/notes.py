"""Short conceptual notes (qualitative; contain no cross-section numbers)."""

TOPICS = {
    "probability": (
        "Probabilities, not predictions. For a given isotope AND neutron energy, the evaluated data give "
        "relative likelihoods of different interaction types. A simulation draws outcomes from those "
        "likelihoods; it does not tell you what a particular real neutron will do. Digits printed are the "
        "numerical precision of stored data, not a statement of physical certainty."),
    "energy": (
        "Energy matters. The same isotope can be dominated by capture at one energy, scattering at another, "
        "and show sharp resonances in between. Isotope identity alone does not determine the outcome; "
        "this program therefore requires an incident-neutron energy."),
    "channels": (
        "Channels are distinct processes: elastic scattering (n,n); inelastic scattering (n,n'); radiative "
        "capture (n,gamma); fission (n,f) for some heavy nuclei; charged-particle emission such as (n,alpha) "
        "or (n,p); and others. Which exist, and how likely they are, depends on the nucleus and the energy."),
    "capture": (
        "Capture: the nucleus absorbs the neutron and de-excites, usually by emitting gamma rays. The product "
        "may be stable or radioactive; that follow-on decay is a separate process."),
    "scattering": (
        "Scattering: the neutron is deflected. Elastic leaves the nucleus in its ground state; inelastic leaves "
        "it excited. Neither absorbs the neutron."),
    "fission": (
        "Fission: a few heavy nuclei can split after absorbing a neutron. This program reports only the "
        "probability of the fission channel relative to other channels. It does not model neutron multiplicity, "
        "fission products, energy release, chain reactions or criticality, by design."),
    "decay": (
        "Radioactive decay is a spontaneous process, not a neutron-induced channel. No decay data are loaded, "
        "so decay is: unsupported by available data."),
    "box": (
        "Box test: neutrons are released inside a block made of one isotope and followed collision by collision until they are "
        "absorbed, leak out, or the clock stops. Density sets how often collisions happen; the box shows how many neutrons are "
        "left, where the energy went (heat) and how that changes over time. It runs for non-fissile isotopes only: with a fission "
        "channel the same box would be a chain-reaction / criticality calculation, which is excluded by design. Heat assumes gammas "
        "and charged particles stop inside the box."),
    "fusion": (
        "Neutron-induced reactions on nuclei are generally not fusion. Fusion (light nuclei combining) is a "
        "different topic with different data and is NOT modelled here."),
}
