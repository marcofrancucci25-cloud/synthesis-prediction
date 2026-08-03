from pathlib import Path


def test_literature_search_has_reset_button_next_to_search():
    source = Path('app.py').read_text(encoding='utf-8')
    assert 'def _reset_literature_inputs():' in source
    assert '"Reset search", on_click=_reset_literature_inputs' in source
    assert 'search_col, reset_col = st.columns(2)' in source


def test_literature_results_are_ordered_title_doi_abstract():
    source = Path('app.py').read_text(encoding='utf-8')
    lit_section = source.split('elif page=="Literature search":')[1]
    lit_section = lit_section.split('\nelse:')[0]
    title_pos = lit_section.find('### {i}.')
    doi_pos = lit_section.find('DOI: {html.escape')
    abstract_pos = lit_section.find("font-size:0.92rem")
    assert -1 < title_pos < doi_pos < abstract_pos, (
        "Expected order in source: title markdown, then DOI caption, then the abstract block"
    )


def test_literature_result_text_is_html_escaped():
    source = Path('app.py').read_text(encoding='utf-8')
    lit_section = source.split('elif page=="Literature search":')[1]
    lit_section = lit_section.split('\nelse:')[0]
    assert 'html.escape(str(article.get("title")' in lit_section
    assert 'html.escape(str(article["summary"]))' in lit_section
