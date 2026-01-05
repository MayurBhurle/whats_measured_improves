# app.py
# Streamlit Daily Tracker: journal -> auto-categorize -> analytics + study log + planner + finance
#
# Run:
#   pip install streamlit pandas plotly sqlalchemy python-dateutil pytz
#   streamlit run app.py
#
# Notes:
# - Auto-category is keyword-based (fast + reliable). You can tweak KEYWORDS anytime.
# - Stores data in SQLite (tracker.db) in the same folder.
# - Day selector starts from Jan 1, 2026 (Asia/Kolkata).

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Tuple, Optional

import pandas as pd
import plotly.express as px
import pytz
import streamlit as st

TZ = pytz.timezone("Asia/Kolkata")
MIN_DATE = date(2026, 1, 1)
DB_PATH = "tracker.db"

# ---------- Categorization (edit these freely) ----------
CATEGORIES = [
    "Study",
    "Work",
    "Reading",
    "Coding",
    "Competitive Exam Prep",
    "Exercise",
    "Social Media",
    "Finance/Admin",
    "Break/Personal",
    "Other",
]

KEYWORDS: Dict[str, List[str]] = {
    "Study": [
        "study", "studied", "revise", "revision", "notes", "lecture", "class",
        "homework", "assignment", "learning", "topic", "chapter"
    ],
    "Work": ["work", "meeting", "office", "client", "task", "jira", "email", "call", "review"],
    "Reading": ["read", "reading", "book", "article", "paper", "blog", "novel"],
    "Coding": ["code", "coding", "program", "debug", "bug", "git", "repo", "pr", "commit", "leetcode", "dsa"],
    "Competitive Exam Prep": ["exam", "prep", "mock", "test", "practice", "quant", "aptitude", "reasoning", "gk"],
    "Exercise": ["gym", "workout", "run", "running", "walk", "walking", "yoga", "stretch", "exercise", "cycling"],
    "Social Media": ["instagram", "ig", "youtube", "yt", "twitter", "x ", "reddit", "scroll", "reels", "shorts", "social"],
    "Finance/Admin": ["bank", "payment", "upi", "bill", "expense", "tax", "admin", "docs", "form", "recharge"],
    "Break/Personal": ["break", "rest", "nap", "lunch", "dinner", "breakfast", "snack", "family", "friends", "chat"],
}

STOPWORDS = set(
    "a an the and or to of in on for with at from into up down over under again further then once "
    "here there when where why how all any both each few more most other some such no nor not only "
    "own same so than too very can will just should now i me my we our you your".split()
)


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def score_category(text: str) -> Tuple[str, float, Dict[str, float]]:
    """
    Returns (best_category, confidence_0_to_1, per_category_scores)
    Simple keyword scoring: count keyword hits with small boosts for exact words.
    """
    t = " " + normalize_text(text) + " "
    scores = {c: 0.0 for c in CATEGORIES}

    # Token set for exact word match bonus
    tokens = [w for w in re.findall(r"[a-z0-9]+", t) if w not in STOPWORDS]
    token_set = set(tokens)

    for cat, kws in KEYWORDS.items():
        s = 0.0
        for kw in kws:
            kw_n = normalize_text(kw)
            # phrase hit
            if kw_n in t:
                s += 1.0
            # exact word bonus
            kw_token = kw_n.strip()
            if kw_token in token_set:
                s += 0.5
        scores[cat] = s

    # Default fallbacks
    best = max(scores.items(), key=lambda kv: kv[1])[0]
    best_score = scores[best]

    # If nothing matched, call it Other with low confidence
    if best_score <= 0:
        return "Other", 0.15, scores

    # Confidence heuristic: best / (best + runner_up + 1)
    sorted_scores = sorted(scores.values(), reverse=True)
    runner = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    conf = best_score / (best_score + runner + 1.0)
    conf = max(0.2, min(0.95, conf))
    return best, conf, scores


# ---------- DB helpers ----------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_date TEXT NOT NULL,
            start_ts TEXT NOT NULL,
            end_ts TEXT NOT NULL,
            text TEXT NOT NULL,
            predicted_category TEXT NOT NULL,
            final_category TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_totals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_date TEXT NOT NULL,
            category TEXT NOT NULL,
            minutes INTEGER NOT NULL,
            UNIQUE(entry_date, category)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS study_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_date TEXT NOT NULL,
            subject TEXT NOT NULL,
            topics TEXT NOT NULL,
            minutes INTEGER NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS planner (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_date TEXT NOT NULL,
            plan_text TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 2,
            done INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS finance_txns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_date TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            merchant TEXT,
            note TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


def db_execute(query: str, params: Tuple = ()) -> None:
    conn = get_conn()
    conn.execute(query, params)
    conn.commit()
    conn.close()


def db_query_df(query: str, params: Tuple = ()) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def iso(dt: datetime) -> str:
    return dt.isoformat()


def parse_iso(s: str) -> datetime:
    # stored without timezone; interpret as Asia/Kolkata
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = TZ.localize(dt)
    return dt


def minutes_between(start_dt: datetime, end_dt: datetime) -> int:
    delta = end_dt - start_dt
    mins = int(max(0, round(delta.total_seconds() / 60)))
    return mins


# ---------- Aggregation ----------
def recompute_daily_totals(entry_date: date):
    d = entry_date.isoformat()
    df = db_query_df(
        """
        SELECT final_category, start_ts, end_ts
        FROM journal_entries
        WHERE entry_date = ?
        """,
        (d,),
    )
    # Clear existing totals for the day
    db_execute("DELETE FROM daily_totals WHERE entry_date = ?", (d,))

    if df.empty:
        return

    totals: Dict[str, int] = {c: 0 for c in CATEGORIES}
    for _, row in df.iterrows():
        s = parse_iso(row["start_ts"])
        e = parse_iso(row["end_ts"])
        mins = minutes_between(s, e)
        cat = row["final_category"]
        if cat not in totals:
            cat = "Other"
        totals[cat] += mins

    # Insert
    conn = get_conn()
    cur = conn.cursor()
    for cat, mins in totals.items():
        if mins > 0:
            cur.execute(
                "INSERT OR REPLACE INTO daily_totals(entry_date, category, minutes) VALUES(?,?,?)",
                (d, cat, int(mins)),
            )
    conn.commit()
    conn.close()


# ---------- UI helpers ----------
def kst_now() -> datetime:
    return datetime.now(TZ)


def clamp_date(d: date) -> date:
    return max(MIN_DATE, d)


def human_minutes(mins: int) -> str:
    h = mins // 60
    m = mins % 60
    if h and m:
        return f"{h}h {m}m"
    if h:
        return f"{h}h"
    return f"{m}m"


# ---------- Streamlit ----------
st.set_page_config(page_title="Daily Activity Tracker", layout="wide")
init_db()

st.title("Daily Activity Tracker")
st.caption("Journal your day → auto-categorize → charts + study + planner + finance")

# Sidebar nav
page = st.sidebar.radio(
    "Navigate",
    ["Today (Journal)", "Review (Graphs)", "Study Log", "Planner", "Finance", "Settings"],
    index=0,
)

# Global date selection (min Jan 1 2026)
default_day = clamp_date(kst_now().date())
selected_day = st.sidebar.date_input(
    "Select day",
    value=default_day,
    min_value=MIN_DATE,
)

# ---------- TODAY ----------
if page == "Today (Journal)":
    st.subheader(f"Journal — {selected_day.isoformat()}")

    # Quick summary (today totals)
    totals_df = db_query_df(
        "SELECT category, minutes FROM daily_totals WHERE entry_date = ?",
        (selected_day.isoformat(),),
    )
    total_mins = int(totals_df["minutes"].sum()) if not totals_df.empty else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total tracked", human_minutes(total_mins))

    focus_set = {"Study", "Work", "Reading", "Coding", "Competitive Exam Prep", "Exercise"}
    focus_mins = int(totals_df[totals_df["category"].isin(focus_set)]["minutes"].sum()) if not totals_df.empty else 0
    leisure_mins = int(totals_df[~totals_df["category"].isin(focus_set)]["minutes"].sum()) if not totals_df.empty else 0
    c2.metric("Focus time", human_minutes(focus_mins))
    c3.metric("Leisure/Other", human_minutes(leisure_mins))

    st.divider()

    # Entry form
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown("### Add a journal entry (time slot + what you did)")
        now_dt = kst_now()
        default_start = datetime.combine(selected_day, time(now_dt.hour, now_dt.minute))
        default_start = TZ.localize(default_start)
        default_end = default_start + timedelta(minutes=30)

        with st.form("journal_form", clear_on_submit=True):
            colA, colB = st.columns(2)
            with colA:
                start_time = st.time_input("Start time", value=default_start.timetz().replace(tzinfo=None))
            with colB:
                end_time = st.time_input("End time", value=default_end.timetz().replace(tzinfo=None))

            text = st.text_area(
                "What did you do in this slot?",
                placeholder="e.g., Solved 5 DSA problems, watched a lecture on DP, did a 30-min run, scrolled Instagram...",
                height=120,
            )
            override = st.checkbox("Manually choose category (override auto)", value=False)

            predicted_cat, conf, _scores = score_category(text) if text.strip() else ("Other", 0.0, {})
            chosen_cat = predicted_cat

            if override:
                chosen_cat = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(predicted_cat))
            else:
                st.info(f"Auto-category: **{predicted_cat}** (confidence ~ {int(conf*100)}%)")

            submitted = st.form_submit_button("Save entry")

        if submitted:
            if not text.strip():
                st.error("Please write what you did for this time slot.")
            else:
                # Build timezone-aware datetimes
                sdt = TZ.localize(datetime.combine(selected_day, start_time))
                edt = TZ.localize(datetime.combine(selected_day, end_time))
                # Handle crossing midnight: if end < start, assume end is next day
                if edt <= sdt:
                    edt = edt + timedelta(days=1)

                mins = minutes_between(sdt, edt)
                if mins <= 0:
                    st.error("End time must be after start time (or it will be treated as next day).")
                else:
                    now_iso = iso(kst_now())
                    db_execute(
                        """
                        INSERT INTO journal_entries(
                            entry_date, start_ts, end_ts, text,
                            predicted_category, final_category, confidence, created_at
                        ) VALUES (?,?,?,?,?,?,?,?)
                        """,
                        (
                            selected_day.isoformat(),
                            iso(sdt),
                            iso(edt),
                            text.strip(),
                            predicted_cat,
                            chosen_cat,
                            float(conf),
                            now_iso,
                        ),
                    )
                    recompute_daily_totals(selected_day)
                    st.success(f"Saved ({human_minutes(mins)}) → {chosen_cat}")

        st.markdown("### Today’s entries")
        entries_df = db_query_df(
            """
            SELECT id, start_ts, end_ts, final_category, confidence, text
            FROM journal_entries
            WHERE entry_date = ?
            ORDER BY start_ts ASC
            """,
            (selected_day.isoformat(),),
        )

        if entries_df.empty:
            st.caption("No entries yet.")
        else:
            # Prepare display
            def _to_local_time(s: str) -> str:
                dt = parse_iso(s)
                return dt.strftime("%H:%M")

            entries_df["start"] = entries_df["start_ts"].apply(_to_local_time)
            entries_df["end"] = entries_df["end_ts"].apply(_to_local_time)
            entries_df["mins"] = entries_df.apply(
                lambda r: minutes_between(parse_iso(r["start_ts"]), parse_iso(r["end_ts"])), axis=1
            )
            show_df = entries_df[["id", "start", "end", "mins", "final_category", "text"]].rename(
                columns={"final_category": "category"}
            )
            st.dataframe(show_df, use_container_width=True, hide_index=True)

            st.markdown("#### Edit / delete an entry")
            ids = show_df["id"].tolist()
            chosen_id = st.selectbox("Select entry ID", ids)
            row = entries_df[entries_df["id"] == chosen_id].iloc[0]

            edit_col1, edit_col2 = st.columns(2)
            with edit_col1:
                new_cat = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(row["final_category"]))
            with edit_col2:
                do_delete = st.checkbox("Delete this entry", value=False)

            new_text = st.text_area("Text", value=row["text"], height=90)

            if st.button("Apply change"):
                if do_delete:
                    db_execute("DELETE FROM journal_entries WHERE id = ?", (int(chosen_id),))
                    recompute_daily_totals(selected_day)
                    st.success("Deleted.")
                    st.rerun()
                else:
                    # Re-score if text changed (optional)
                    pred2, conf2, _ = score_category(new_text) if new_text.strip() else ("Other", 0.15, {})
                    db_execute(
                        """
                        UPDATE journal_entries
                        SET text = ?, predicted_category = ?, final_category = ?, confidence = ?
                        WHERE id = ?
                        """,
                        (new_text.strip(), pred2, new_cat, float(conf2), int(chosen_id)),
                    )
                    recompute_daily_totals(selected_day)
                    st.success("Updated.")
                    st.rerun()

    with right:
        st.markdown("### Day timeline (sessions)")
        # timeline chart
        entries_df2 = db_query_df(
            """
            SELECT start_ts, end_ts, final_category, text
            FROM journal_entries
            WHERE entry_date = ?
            ORDER BY start_ts ASC
            """,
            (selected_day.isoformat(),),
        )
        if entries_df2.empty:
            st.caption("Add entries to see a timeline.")
        else:
            chart_df = entries_df2.copy()
            chart_df["Start"] = chart_df["start_ts"].apply(parse_iso)
            chart_df["Finish"] = chart_df["end_ts"].apply(parse_iso)
            chart_df["Category"] = chart_df["final_category"]
            chart_df["Label"] = chart_df["text"].str.slice(0, 50)
            fig = px.timeline(
                chart_df,
                x_start="Start",
                x_end="Finish",
                y="Category",
                hover_data={"Label": True, "Start": True, "Finish": True},
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Quick totals (by category)")
        totals_df2 = db_query_df(
            "SELECT category, minutes FROM daily_totals WHERE entry_date = ? ORDER BY minutes DESC",
            (selected_day.isoformat(),),
        )
        if totals_df2.empty:
            st.caption("No totals yet.")
        else:
            totals_df2["time"] = totals_df2["minutes"].apply(human_minutes)
            st.dataframe(totals_df2[["category", "time"]], use_container_width=True, hide_index=True)

        if st.button("Finalize / recompute totals for the day"):
            recompute_daily_totals(selected_day)
            st.success("Recomputed daily totals.")


# ---------- REVIEW / GRAPHS ----------
elif page == "Review (Graphs)":
    st.subheader("Review (Graphs)")

    # Range selector
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        start_d = st.date_input("From", value=clamp_date(selected_day - timedelta(days=7)), min_value=MIN_DATE, key="rev_from")
    with col2:
        end_d = st.date_input("To", value=selected_day, min_value=MIN_DATE, key="rev_to")
    with col3:
        chosen_acts = st.multiselect("Activities", CATEGORIES, default=["Study", "Work", "Coding", "Competitive Exam Prep"])

    if end_d < start_d:
        st.error("End date must be on/after start date.")
    else:
        totals = db_query_df(
            """
            SELECT entry_date, category, minutes
            FROM daily_totals
            WHERE entry_date BETWEEN ? AND ?
            """,
            (start_d.isoformat(), end_d.isoformat()),
        )
        if totals.empty:
            st.caption("No data in this range yet. Add journal entries first.")
        else:
            totals["entry_date"] = pd.to_datetime(totals["entry_date"])
            totals["hours"] = totals["minutes"] / 60.0

            # 1) Stacked bar: daily time by activity
            st.markdown("### Daily total by activity (stacked)")
            stacked = totals.copy()
            fig1 = px.bar(
                stacked,
                x="entry_date",
                y="hours",
                color="category",
                title="Hours per day by activity",
            )
            st.plotly_chart(fig1, use_container_width=True)

            # 2) Line: selected activities trend (X=time, Y=time spent)
            st.markdown("### Trend (X = date, Y = time spent)")
            trend = totals[totals["category"].isin(chosen_acts)].copy()
            if trend.empty:
                st.caption("Select at least one activity that has data in the chosen range.")
            else:
                fig2 = px.line(
                    trend,
                    x="entry_date",
                    y="hours",
                    color="category",
                    markers=True,
                    title="Hours per day (selected activities)",
                )
                st.plotly_chart(fig2, use_container_width=True)

            # 3) Per-activity explorer (graph for each)
            st.markdown("### Activity Explorer")
            activity = st.selectbox("Pick one activity", CATEGORIES, index=CATEGORIES.index("Study") if "Study" in CATEGORIES else 0)
            single = totals[totals["category"] == activity].copy()
            if single.empty:
                st.caption("No data for this activity in the selected range.")
            else:
                fig3 = px.line(single, x="entry_date", y="hours", markers=True, title=f"{activity} — daily hours")
                st.plotly_chart(fig3, use_container_width=True)

                # Rolling average (7 days)
                single_sorted = single.sort_values("entry_date")
                single_sorted["roll7"] = single_sorted["hours"].rolling(7, min_periods=1).mean()
                fig4 = px.line(single_sorted, x="entry_date", y="roll7", markers=True, title=f"{activity} — 7-day average")
                st.plotly_chart(fig4, use_container_width=True)

            # 4) Timeline for a selected day
            st.markdown("### Intra-day timeline (X = clock time)")
            day_for_tl = st.date_input("Choose a day to view timeline", value=selected_day, min_value=MIN_DATE, key="tl_day")
            entries = db_query_df(
                """
                SELECT start_ts, end_ts, final_category, text
                FROM journal_entries
                WHERE entry_date = ?
                ORDER BY start_ts ASC
                """,
                (day_for_tl.isoformat(),),
            )
            if entries.empty:
                st.caption("No entries for this day.")
            else:
                entries["Start"] = entries["start_ts"].apply(parse_iso)
                entries["Finish"] = entries["end_ts"].apply(parse_iso)
                entries["Category"] = entries["final_category"]
                entries["Label"] = entries["text"].str.slice(0, 60)
                fig5 = px.timeline(
                    entries,
                    x_start="Start",
                    x_end="Finish",
                    y="Category",
                    hover_data={"Label": True, "Start": True, "Finish": True},
                    title=f"Timeline — {day_for_tl.isoformat()}",
                )
                fig5.update_yaxes(autorange="reversed")
                st.plotly_chart(fig5, use_container_width=True)


# ---------- STUDY LOG ----------
elif page == "Study Log":
    st.subheader(f"Study Log — {selected_day.isoformat()}")

    with st.form("study_form", clear_on_submit=True):
        subject = st.text_input("Subject", placeholder="e.g., Math, DSA, English, Physics")
        topics = st.text_input("Topics covered", placeholder="e.g., DP basics, Integration, RC practice")
        minutes = st.number_input("Minutes", min_value=0, max_value=24 * 60, value=60, step=5)
        notes = st.text_area("Notes (optional)", height=90)
        ok = st.form_submit_button("Add study record")

    if ok:
        if not subject.strip() or not topics.strip():
            st.error("Please fill Subject and Topics.")
        else:
            db_execute(
                """
                INSERT INTO study_log(entry_date, subject, topics, minutes, notes, created_at)
                VALUES(?,?,?,?,?,?)
                """,
                (selected_day.isoformat(), subject.strip(), topics.strip(), int(minutes), notes.strip(), iso(kst_now())),
            )
            st.success("Saved study record.")

    df = db_query_df(
        """
        SELECT id, subject, topics, minutes, notes, created_at
        FROM study_log
        WHERE entry_date = ?
        ORDER BY created_at DESC
        """,
        (selected_day.isoformat(),),
    )
    if df.empty:
        st.caption("No study records yet.")
    else:
        df["time"] = df["minutes"].apply(human_minutes)
        st.dataframe(df[["id", "subject", "topics", "time", "notes"]], use_container_width=True, hide_index=True)


# ---------- PLANNER ----------
elif page == "Planner":
    st.subheader(f"Planner — {selected_day.isoformat()}")

    with st.form("plan_form", clear_on_submit=True):
        plan_text = st.text_input("Task / plan item", placeholder="e.g., Finish DP sheet, 30-min run, read 10 pages")
        priority = st.selectbox("Priority", [1, 2, 3], index=1, format_func=lambda x: {1: "High", 2: "Medium", 3: "Low"}[x])
        ok = st.form_submit_button("Add plan item")

    if ok:
        if not plan_text.strip():
            st.error("Please type a plan item.")
        else:
            db_execute(
                """
                INSERT INTO planner(entry_date, plan_text, priority, done, created_at)
                VALUES(?,?,?,?,?)
                """,
                (selected_day.isoformat(), plan_text.strip(), int(priority), 0, iso(kst_now())),
            )
            st.success("Added.")

    plans = db_query_df(
        """
        SELECT id, plan_text, priority, done
        FROM planner
        WHERE entry_date = ?
        ORDER BY done ASC, priority ASC, id DESC
        """,
        (selected_day.isoformat(),),
    )
    if plans.empty:
        st.caption("No plan items yet.")
    else:
        st.markdown("### Your tasks")
        for _, r in plans.iterrows():
            cols = st.columns([0.1, 0.75, 0.15])
            done_now = cols[0].checkbox("", value=bool(r["done"]), key=f"done_{r['id']}")
            pr = {1: "High", 2: "Medium", 3: "Low"}.get(int(r["priority"]), "Medium")
            cols[1].write(f"**[{pr}]** {r['plan_text']}")
            if cols[2].button("Delete", key=f"del_{r['id']}"):
                db_execute("DELETE FROM planner WHERE id = ?", (int(r["id"]),))
                st.rerun()
            # update done
            if int(r["done"]) != int(done_now):
                db_execute("UPDATE planner SET done = ? WHERE id = ?", (int(done_now), int(r["id"])))

# ---------- FINANCE ----------
elif page == "Finance":
    st.subheader(f"Finance — {selected_day.isoformat()}")

    fin_cats = ["Food", "Transport", "Shopping", "Bills", "Entertainment", "Health", "Education", "Other"]

    with st.form("fin_form", clear_on_submit=True):
        amount = st.number_input("Amount (₹)", min_value=0.0, value=100.0, step=10.0)
        fcat = st.selectbox("Category", fin_cats, index=0)
        merchant = st.text_input("Where / Merchant (optional)", placeholder="e.g., Swiggy, Metro, Amazon")
        note = st.text_input("Note (optional)", placeholder="e.g., lunch, cab, stationery")
        ok = st.form_submit_button("Add expense")

    if ok:
        if amount <= 0:
            st.error("Amount must be > 0.")
        else:
            db_execute(
                """
                INSERT INTO finance_txns(entry_date, amount, category, merchant, note, created_at)
                VALUES(?,?,?,?,?,?)
                """,
                (selected_day.isoformat(), float(amount), fcat, merchant.strip(), note.strip(), iso(kst_now())),
            )
            st.success("Added expense.")

    df = db_query_df(
        """
        SELECT id, amount, category, merchant, note, created_at
        FROM finance_txns
        WHERE entry_date = ?
        ORDER BY created_at DESC
        """,
        (selected_day.isoformat(),),
    )

    if df.empty:
        st.caption("No expenses yet.")
    else:
        total = float(df["amount"].sum())
        st.metric("Total spent today", f"₹{total:,.2f}")

        st.dataframe(df[["id", "amount", "category", "merchant", "note"]], use_container_width=True, hide_index=True)

        # Pie for today
        pie = df.groupby("category", as_index=False)["amount"].sum()
        fig = px.pie(pie, names="category", values="amount", title="Spend distribution (today)")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Finance over time")
    col1, col2 = st.columns(2)
    with col1:
        f_from = st.date_input("From", value=clamp_date(selected_day - timedelta(days=30)), min_value=MIN_DATE, key="fin_from")
    with col2:
        f_to = st.date_input("To", value=selected_day, min_value=MIN_DATE, key="fin_to")

    if f_to >= f_from:
        hist = db_query_df(
            """
            SELECT entry_date, amount, category
            FROM finance_txns
            WHERE entry_date BETWEEN ? AND ?
            """,
            (f_from.isoformat(), f_to.isoformat()),
        )
        if hist.empty:
            st.caption("No expenses in this range.")
        else:
            hist["entry_date"] = pd.to_datetime(hist["entry_date"])
            daily = hist.groupby("entry_date", as_index=False)["amount"].sum()
            fig2 = px.line(daily, x="entry_date", y="amount", markers=True, title="Daily total spend")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.error("End date must be on/after start date.")


# ---------- SETTINGS ----------
elif page == "Settings":
    st.subheader("Settings")

    st.markdown("### Categorization keywords")
    st.caption("Edit keywords in code under KEYWORDS to improve auto-classification.")

    st.markdown("### Export data")
    colA, colB = st.columns(2)

    with colA:
        if st.button("Export journal_entries CSV"):
            df = db_query_df("SELECT * FROM journal_entries ORDER BY entry_date, start_ts", ())
            st.download_button(
                "Download journal_entries.csv",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="journal_entries.csv",
                mime="text/csv",
            )

    with colB:
        if st.button("Export daily_totals CSV"):
            df = db_query_df("SELECT * FROM daily_totals ORDER BY entry_date, category", ())
            st.download_button(
                "Download daily_totals.csv",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="daily_totals.csv",
                mime="text/csv",
            )

    st.divider()
    st.markdown("### Danger zone")
    if st.checkbox("I understand, show delete options"):
        if st.button("DELETE ALL DATA (cannot undo)"):
            conn = get_conn()
            cur = conn.cursor()
            cur.execute("DELETE FROM journal_entries")
            cur.execute("DELETE FROM daily_totals")
            cur.execute("DELETE FROM study_log")
            cur.execute("DELETE FROM planner")
            cur.execute("DELETE FROM finance_txns")
            conn.commit()
            conn.close()
            st.success("All data deleted.")
