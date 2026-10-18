"""Admin console for user management, role delegation, system health, and model run audit."""

import logging

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

from ..db.models import Dataset, ModelRun, User
from ..db.session import SessionLocal


def render_admin_page() -> None:
    """Admin console: user management, role assignments, system status, and model run history."""
    st.header("Admin Console")

    current_user = st.session_state.get("user")
    if not current_user or current_user.get("role") != "Admin":
        st.error("Access Restricted: Administrator role required to view this console.")
        return

    if SessionLocal is None or User is None:
        st.warning("Database session is not available. Please ensure database configuration is active.")
        return

    st.markdown(f"Signed in as Administrator: **{current_user.get('name', 'Admin')}** (`{current_user.get('email', '')}`)")

    tab_users, tab_runs, tab_system = st.tabs(["User Management", "Recent Model Runs", "System Diagnostics"])

    with tab_users:
        st.subheader("Registered Users")
        try:
            with SessionLocal() as db:
                users = db.query(User).all()
                if not users:
                    st.info("No registered users found.")
                else:
                    user_data = [
                        {
                            "User ID": u.id,
                            "Email": u.email,
                            "Full Name": u.full_name or "N/A",
                            "Role": u.role or "Viewer",
                            "Phone": getattr(u, "phone", None) or "N/A",
                            "Score": getattr(u, "score", 0) or 0,
                            "Joined At (UTC)": u.created_at.strftime("%Y-%m-%d %H:%M:%S") if u.created_at else "N/A",
                        }
                        for u in users
                    ]
                    users_df = pd.DataFrame(user_data)
                    st.dataframe(users_df, use_container_width=True)

                    st.markdown("### Manage User Role")
                    user_options = {f"{u.id} — {u.email} ({u.role})": u.id for u in users}
                    selected_user_label = st.selectbox("Select user to modify role", list(user_options.keys()), key="admin_sel_user")
                    selected_user_id = user_options[selected_user_label]

                    new_role = st.selectbox("Assign new role", ["Viewer", "Analyst", "Admin"], key="admin_new_role")
                    if st.button("Update User Role", key="admin_update_role_btn"):
                        u_target = db.query(User).get(selected_user_id)
                        if u_target:
                            old_role = u_target.role
                            u_target.role = new_role
                            db.commit()
                            st.success(f"Updated role for {u_target.email}: {old_role} → {new_role}")
                            logger.info("Admin updated user %s role to %s", u_target.email, new_role)
                        else:
                            st.error("User not found in database.")
        except Exception as e:
            st.error(f"Failed to fetch user list: {e}")
            logger.exception("Admin user query failed")

    with tab_runs:
        st.subheader("Recent Model Training Runs")
        if ModelRun is None:
            st.info("ModelRun entity not configured.")
        else:
            try:
                with SessionLocal() as db:
                    runs = db.query(ModelRun).order_by(ModelRun.created_at.desc()).limit(30).all()
                    if not runs:
                        st.info("No model runs recorded.")
                    else:
                        runs_data = [
                            {
                                "ID": r.id,
                                "Model Name": r.model_name,
                                "Version": r.version,
                                "Artifact Path": r.file_path,
                                "Created At (UTC)": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "N/A",
                            }
                            for r in runs
                        ]
                        st.dataframe(pd.DataFrame(runs_data), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to fetch model runs: {e}")
                logger.exception("Admin model runs query failed")

    with tab_system:
        st.subheader("Platform Diagnostics")
        try:
            with SessionLocal() as db:
                user_count = db.query(User).count() if User is not None else 0
                run_count = db.query(ModelRun).count() if ModelRun is not None else 0
                ds_count = db.query(Dataset).count() if Dataset is not None else 0

                c1, c2, c3 = st.columns(3)
                c1.metric("Registered Users", user_count)
                c2.metric("Total Model Runs", run_count)
                c3.metric("Saved Datasets", ds_count)
        except Exception as e:
            st.warning(f"Could not load platform metrics: {e}")
