# Interface update v10.7.2

This release changes only the public interface and supporting chemical labels. The predictive model, training data and probability calculations are unchanged.

- The “Scientific scope of this optimization” diagnostic expander is no longer rendered.
- The “Similar experimental records” diagnostic expander is no longer rendered.
- “Reset optimizer fields” restores objective, limits, search depth, constraints and generated results without clearing the synthesis input form.
- Metal options are displayed as `symbol — HSAB class`.
- HSAB character is evaluated using the selected oxidation state for variable-valence ions. For example, Cu(I) is shown as soft while Cu(II) is shown as borderline; Fe(III) is hard while Fe(II) is borderline.
- The label remains indicative because coordination environment can shift practical Lewis acidity and softness.
