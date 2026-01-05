import io
import json
from datetime import date

import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload


APP_MIN_DATE = date(2026, 1, 1)

# -------- Google Drive helpers --------
SCOPES = ["https://www.googleapis.com/auth/drive.file"]  # file access for files your app creates/uses

def get_drive_service():
    """
    Auth using a Service Account JSON stored in st.secrets.
    """
    sa_json_str = st.secrets["google"]["service_account_json"]
    sa_info = json.loads(sa_json_str)

    creds = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=SCOPES,
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def find_file_in_folder(service, folder_id: str, filename: str):
    """
    Return fileId if filename exists in folder; else None.
    """
    # Escape single quotes in filename for Drive query.
    safe_name = filename.replace("'", "\\'")
    q = (
        f"'{folder_id}' in parents and "
        f"name = '{safe_name}' and "
        f"trashed = false"
    )
    res = service.files().list(
        q=q,
        fields="files(id, name, modifiedTime)",
        pageSize=10,
    ).execute()
    files = res.get("files", [])
    return files[0] if files else None

def download_text_file(service, file_id: str) -> str:
    """
    Download file content as text.
    """
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read().decode("utf-8", errors="replace")

def upload_or_update_text(service, folder_id: str, filename: str, content: str) -> str:
    """
    Create or update a .md file in the target folder. Returns fileId.
    """
    existing = find_file_in_folder(service, folder_id, filename)
    media = MediaIoBaseUpload(
        io.BytesIO(content.encode("utf-8")),
        mimetype="text/markdown",
        resumable=False,
    )

    if existing:
        file_id = existing["id"]
        service.files().update(
            fileId=file_id,
            media_body=media,
        ).execute()
        return file_id

    file_metadata = {
        "name": filename,
        "parents": [folder_id],
        "mimeType": "text/markdown",
    }
    created = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id",
    ).execute()
    return created["id"]

# -------- Streamlit UI --------
st.set_page_config(page_title="Daily Notes → Google Drive", layout="wide")
st.title("🗓️ Daily Notes (saved to Google Drive)")
st.caption("Basic per-day note-taking. Notes are stored as .md files in a Google Drive folder you choose.")

# Validate secrets
if "google" not in st.secrets or "folder_id" not in st.secrets["google"] or "service_account_json" not in st.secrets["google"]:
    st.error(
        "Missing Google Drive configuration. Add [google].folder_id and [google].service_account_json to Streamlit secrets."
    )
    st.stop()

folder_id = st.secrets["google"]["folder_id"]

# Date selector
chosen_date = st.date_input(
    "Select day",
    value=max(date.today(), APP_MIN_DATE),
    min_value=APP_MIN_DATE,
)

filename = f"{chosen_date.isoformat()}.md"

# Connect Drive
@st.cache_resource
def cached_drive():
    return get_drive_service()

drive = cached_drive()

# Load note button
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    if st.button("Load from Drive"):
        meta = find_file_in_folder(drive, folder_id, filename)
        if not meta:
            st.session_state["note_text"] = ""
            st.toast("No existing note for this date.", icon="ℹ️")
        else:
            st.session_state["note_text"] = download_text_file(drive, meta["id"])
            st.toast("Loaded from Google Drive ✅", icon="✅")

with colB:
    if st.button("Open Drive folder"):
        st.info("Tip: open your Drive folder in a new tab; this app can’t auto-navigate there.")

# Auto-load when date changes (lightweight)
if st.session_state.get("last_date") != chosen_date:
    meta = find_file_in_folder(drive, folder_id, filename)
    st.session_state["note_text"] = download_text_file(drive, meta["id"]) if meta else ""
    st.session_state["last_date"] = chosen_date

note = st.text_area(
    f"Note for {chosen_date.isoformat()} (Markdown)",
    value=st.session_state.get("note_text", ""),
    height=360,
    placeholder="Write your journal note here...",
)

save_col1, save_col2, save_col3 = st.columns([1, 1, 2])

with save_col1:
    if st.button("Save to Google Drive"):
        file_id = upload_or_update_text(drive, folder_id, filename, note)
        st.session_state["note_text"] = note
        st.success(f"Saved to Drive as {filename}")
        st.caption(f"Drive fileId: {file_id}")

with save_col2:
    st.download_button(
        "Download .md",
        data=note.encode("utf-8"),
        file_name=filename,
        mime="text/markdown",
    )

st.divider()
st.subheader("Preview")
st.markdown(note if note.strip() else "_Nothing yet._")



