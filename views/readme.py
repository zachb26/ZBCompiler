# -*- coding: utf-8 -*-
import streamlit as st

import constants as const


def render_readme_view():
    st.subheader("ReadMe / Usage")
    st.caption("Edit the README_USAGE_TEXT constant near the top of streamlit_app.py to customize this section.")
    if const.README_USAGE_TEXT.strip():
        st.markdown(const.README_USAGE_TEXT)
    else:
        st.text_area(
            "ReadMe / Usage Placeholder",
            value="",
            height=240,
            placeholder="Add your ReadMe / Usage copy in the README_USAGE_TEXT constant in streamlit_app.py.",
            disabled=True,
            label_visibility="collapsed",
        )
