import streamlit as st
import pandas as pd
from datetime import datetime
from mistralai import Mistral
from supabase import create_client, Client
from google.oauth2.service_account import Credentials
import gspread
import ast
from openai import OpenAI

from helpers import read_data
from helpers import define_standard_info_mapper
from helpers import plot_ui
from helpers import plot_heatmap
from helpers import read_supabase_documents
from helpers import display_annotated_pdf
from helpers import get_all_reports
from helpers import query_single_report
from helpers import define_popover_title
from helpers import summarize_text_bygpt
from helpers import create_google_auth_credentials
from helpers import get_most_similar_pages
from helpers import read_supabase_pages



# ------------------------------------ SETUP ----------------------------------
st.set_page_config(layout="wide", page_title="SRN CSRD Archive", page_icon="srn-icon.png")
st.markdown("""<style> footer {visibility: hidden;} </style> """, unsafe_allow_html=True)

# Supabase
supabase_url: str = st.secrets["SUPABASE_URL"]
supabase_key: str = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(supabase_url, supabase_key)


# OpenAI
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Google Sheets API
create_google_auth_credentials()
scope = ['https://www.googleapis.com/auth/spreadsheets']
creds = Credentials.from_service_account_file(
    "google-auth-credentials.json",
    scopes=scope
)
google_client = gspread.authorize(creds)
sheet_id = '17pxk3WyL-6Fhyw2WATl_Czn9MVpNk-A-LoWxQzhxopU'
log_spreadsheet = google_client.open_by_key(sheet_id)
log_prompt = log_spreadsheet.worksheet("prompts")
log_users = log_spreadsheet.worksheet("users")

# log_users.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), st.query_params.get("ref", "")])# .../?ref=linkedin_victor logs the referrer

standard_info_mapper = define_standard_info_mapper()

googlesheet = read_data()
hosted_docs = read_supabase_documents(supabase)
pages = read_supabase_pages(supabase)

df = (
    googlesheet
    .merge(hosted_docs, on=['company', 'isin'], how="outer", indicator="_mergeSupabase")
    .dropna(subset=["company", "isin", "country", "sector", "industry"])
    .merge(pages, on=["document_id"], how="outer", indicator="_mergePages")
    .query('_mergePages != "right_only"')
)


if "selected_companies" not in st.session_state:
    st.session_state.selected_companies = set()


# ------------------------------------ WELCOME ----------------------------------
left_main_col, right_main_col = st.columns((0.6, 0.4))
with left_main_col:
    plot_ui("welcome-text", df=df)

with right_main_col:
    # Custom CSS for Bubble Counter
    plot_ui("bubble-counter", df=df)

st.divider()


# ------------------------------------ FILTERS ----------------------------------
st.markdown("### Filters")

col1, col2, col3 = st.columns(3)
with col1:
    country_options = ["All"] + sorted(df["country"].unique())
    selected_countries = st.multiselect("Filter by country", options=country_options, default=["All"], key="tab1_country")

with col2:
    industry_options = ["All"] + sorted(df["sector"].unique())
    selected_industries = st.multiselect("Filter by sector", options=industry_options, default=["All"], key="tab1_industry")

# Apply filtering logic
if "All" in selected_countries:
    filtered_countries = df["country"].unique()
else:
    filtered_countries = selected_countries

if "All" in selected_industries:
    filtered_industries = df["sector"].unique()
else:
    filtered_industries = selected_industries

filtered_df = df[
    df["country"].isin(filtered_countries) &
    df["sector"].isin(filtered_industries)
]

with col3:
    selected_companies = st.multiselect(
        label="Filter by name",
        options=[None] + sorted(df["company"]),
        default=None,
        key="tab1_selectbox"
    )

# If the user selects a company, we filter; otherwise we keep all rows.
if len(selected_companies) != 0:
    filtered_df = filtered_df[filtered_df["company"].isin(selected_companies)]



try:
    tab1, tab2 = st.tabs(["List of reports", "Heatmap of topics reported"])


    # ------------------------------------ TABLE ----------------------------------
    with tab1:

        table = st.dataframe(
            (
                filtered_df
                .assign(company = lambda x: [company if _mergePages == "both" else f"{company}*" for company, _mergePages in zip(x["company"], x["_mergePages"])])
                .loc[:, ["company", "link", "country", "sector", "industry", "publication date", "pages PDF", "auditor"]]
            ),
            column_config={
                "company": st.column_config.Column(width="medium", label="Company"),
                "link": st.column_config.LinkColumn(
                    label="Download",
                    width="small",
                    display_text="Link"
                ),
                "country": st.column_config.Column(label="Country"),
                "sector": st.column_config.Column(width="medium", label="Sector"),
                "industry": st.column_config.Column(width="medium", label="Industry"),
                "publication date": st.column_config.DateColumn(
                    format="DD.MM.YYYY", width="small", label="Published"
                ),
                "pages PDF": st.column_config.NumberColumn(
                    help="Number of pages of the sustainability statement.",
                    label="Pages"
                ),
                "auditor": st.column_config.TextColumn(label="Auditor"),
            },
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            # height=35 * len(filtered_df) + 38,
        )

        query_companies = table.selection.rows
        query_companies_df = filtered_df.iloc[query_companies, :]


    # ----- SEARCH ENGIN -----
    with st.container():
        st.markdown("### Search Engine")
        st.caption(":gray[Reports marked with an asterisk (*) cannot yet be queried. We will upload them soon!]")

        prompt = st.chat_input(define_popover_title(query_companies_df), disabled=query_companies == [] or len(query_companies) > 5)

        if prompt:
            log_prompt.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), prompt, ", ".join(query_companies_df['company'].values)])

            for _, query_document in query_companies_df.iterrows():
                # Define stuff
                query_company_name = query_document['company']
                query_document_id = query_document['document_id']
                query_document_start_page_pdf = int(ast.literal_eval(query_document["pages"])[0])
                query_document_url = f"https://gbixxtefgqebkaviusss.supabase.co/storage/v1/object/public/document-pdfs/{query_document_id}.pdf"

                try:
                    with st.spinner():
                        # Query all pages of the selected document
                        query_report_allpages = (
                            supabase.table("pages")
                            .select("*")
                            .eq("document_id", query_document_id)
                            .execute()
                        ).data

                    similar_pages = get_most_similar_pages(prompt, query_report_allpages, top_pages=3)
                                        
                    if similar_pages == []:
                        st.error(f"We have not processed the report of {query_company_name}.")

                    else:
                        with st.expander(query_company_name, expanded=True):
                            col_expander_response, col_expander_pdf = st.columns([0.35, 0.65])

                            # Left column: Prompt + OpenAI response (@To-Do: switch to Mistral)
                            with col_expander_response:
                                query_results_text = "\n".join([x["content"] for x in similar_pages])

                                with st.chat_message("user"):
                                    st.text(prompt)

                                with st.chat_message("assistant"):
                                    stream = summarize_text_bygpt(
                                        client=openai_client, 
                                        queryText=prompt, 
                                        relevantChunkTexts=query_results_text
                                        )
                                    
                                    gpt_response = st.write_stream(stream)
                                    st.markdown(f"[Access the full report here]({query_document_url})")

                            # Right column: Render relevant PDF pages
                            with col_expander_pdf:
                                pages_to_render = [
                                    int(p["page"]) - query_document_start_page_pdf + 1
                                    for p in sorted(similar_pages, key=lambda x: x["score"], reverse=True)
                                    ]

                                with st.spinner("Downloading and finding the relevant pages", show_time=True):
                                    display_annotated_pdf(
                                        query_document_url,
                                        pages_to_render=pages_to_render
                                        )


                except Exception as e:
                    st.error(f"Could not find any relevant information in the PDF for {query_company_name}.")
                    print(e)
            



# ------------------------------------ HEATMAP ----------------------------------
    with tab2:
        col_tab2_left, col_tab2_right = st.columns([0.5, 0.5])

        with col_tab2_left:
            st.markdown("""##### Explanation \n\n This chart shows simple counts of how often a standard is referenced in the company's sustainability statement. To compute the count, we scan the pages of the sustainability statement and count the occurrences of the standard identifier (e.g., E1, E2, ..., G1).""")

            st.markdown("###### Scaling\n\n")
            st.checkbox(label="Scale the counts by the number of datapoints per standard from IG-3 (to control for longer standards)", key="scale_by_dp")
            scale_by_dp = st.session_state.get("scale_by_dp", False)

            st.markdown("###### Split view")
            split_view = st.radio(label="None", options=("by sector", "by country", "by auditor", "no split"), index=0, horizontal=True, label_visibility="collapsed")


        with col_tab2_right:
            filtered_melted_df = (
                filtered_df
                .loc[:, [
                    'company', "sector", "country", "auditor", "pages PDF", 
                    'e1', 'e2', "e3", "e4", "e5", "s1", "s2", "s3", "s4", "g1"
                    ]
                ]
                .melt(id_vars=["company", "sector", "country", "auditor", "pages PDF"], value_name="hits", var_name="standard")
                .merge(standard_info_mapper)
                .assign(
                    standard=lambda x: x['standard'].str.upper(),
                    hits=lambda x: x["hits"] / x["ig3_dp"] if scale_by_dp else x["hits"],  
                    )
                .sort_values("sector")
                .dropna()
            )

            if filtered_melted_df.empty:
                st.error(f"We have not analyzed this company yet but will do so very soon!", icon="🚨")

            else:
                plot_heatmap(filtered_melted_df, split_view)




# ------------------------------------ ERROR HANDLING ----------------------------------
except Exception as e:
    st.error('This is an error. We are working on a fix. In the meantime, check out our Google Sheet!', icon="🚨")
    print(e)