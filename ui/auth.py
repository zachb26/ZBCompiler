# -*- coding: utf-8 -*-
import os

import streamlit as st


def get_secret_value(secret_name):
    try:
        value = st.secrets[secret_name]
    except Exception:
        value = os.environ.get(secret_name)
    text = str(value or "").strip()
    return text or None


def render_password_gate(session_key, secret_name, heading, description, button_label):
    if st.session_state.get(session_key):
        return True

    st.markdown(f"#### {heading}")
    st.caption(description)
    expected_password = get_secret_value(secret_name)
    if not expected_password:
        st.error(
            f"Configuration missing. Add `{secret_name}` in Streamlit Secrets before using this section."
        )
        return False

    password_key = f"{session_key}_password"
    error_key = f"{session_key}_error"
    entered_password = st.text_input("Password", type="password", key=password_key)
    if st.button(button_label, key=f"{session_key}_submit", width="stretch"):
        if entered_password == expected_password:
            st.session_state[session_key] = True
            st.session_state.pop(error_key, None)
            st.session_state[password_key] = ""
            st.rerun()
        else:
            st.session_state[error_key] = "Incorrect password."

    if st.session_state.get(error_key):
        st.error(st.session_state[error_key])
    return False
