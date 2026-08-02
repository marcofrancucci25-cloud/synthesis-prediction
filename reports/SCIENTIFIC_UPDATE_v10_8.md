# Scientific update v10.8.0 — curated MOF literature matching

## Added capability

The prediction workflow now checks the entered metal and canonical linker identity against a local, curated literature registry. When an exact pair is present, the interface displays the documented framework candidate, the article title and a DOI link.

## Error-control policy

- No framework name is inferred from a web-search snippet.
- Ligands are matched only through explicit aliases; chemically substituted bipyrazoles cannot fall through to the unsubstituted BPZ record.
- Metal identity must match exactly.
- Oxidation-state disagreement is retained and displayed to the user rather than silently ignored.
- DOI syntax is validated at application import, and every clickable URL is constructed directly as `https://doi.org/<validated DOI>`.
- Unknown pairs return no claim; there is no fuzzy fallback.

## Scientific interpretation

A metal–linker pair is not a unique structural identifier. Solvent, precursor, modulator, stoichiometry, temperature and kinetic history can lead to different phases or interpenetration states. Therefore, the interface describes registry results as **literature candidates**. Product identity still requires PXRD comparison with an appropriate simulated/reference pattern or SCXRD.

## Initial coverage

The curated release registry covers 16 exact literature records across UiO-66, MOF-5/IRMOF-1, MIL-101(Cr), MIL-53(Al), HKUST-1, ZIF-8, ZIF-67, Mg/Ni-MOF-74 and selected unsubstituted or functionalized bipyrazolate frameworks. Registry expansion requires the same DOI and pair-verification review.
