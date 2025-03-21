import json
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from mistralai import Mistral

from streamlit_pdf_viewer import pdf_viewer

from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx


def read_data() -> pd.DataFrame:
    """
    Read data the SRN CSRD Archive Google Sheet, merge Industry-Sector lookup
    add Standard-Counts dataframes, and return a cleaned DataFrame.
    """
    return (
        pd.read_csv("https://docs.google.com/spreadsheets/d/1Nlyf8Yz_9Fst8rEmQc2IMc-DWLF1fpmBTB7n4FlZwxs/export?format=csv&gid=0", skiprows=2)
        .query("verified == 'yes'")
        .rename(columns={
            'SASB industry \n(SICS® Industries)': "industry",
            })
        .merge(
            # Merge Industry-Sector Lookup from separate sheet
            pd.read_csv(
                "https://docs.google.com/spreadsheets/d/1Nlyf8Yz_9Fst8rEmQc2IMc-DWLF1fpmBTB7n4FlZwxs/export?format=csv&gid=218767986#gid=218767986"
                ).rename(columns={
                    "SICS® Industries": "industry",
                    "SICS® Sector": "sector"
                    }
                ), on="industry", how="left"
            )
        .assign(
            # link = lambda x: [f"{y}#name={z}" for y, z in zip(x["link"], x["company"])],
            # link = lambda x: [f"{y}#download=⬇️" for y, z in zip(x["link"], x["company"])],
            company = lambda x: x["company"].str.strip(),
            )
        .loc[:, ['company', "isin", 'link', 'country', 'sector', 'industry', "publication date", "pages PDF", "auditor"]]
        .dropna()
        # Merge the standard-counts dataframe
        .merge(
            (
                pd.read_csv("https://docs.google.com/spreadsheets/d/1Vj8yau93kmSs_WqnV5w1V_tdU-JlMo-BV6htDvAv1TI/export?format=csv&gid=1792638779#gid=1792638779")
                .assign(
                    isin = lambda x: x["isin"].str.strip(),
                    )
                .query("year == 2024")
                .drop_duplicates(subset=['isin'])
                .drop(["company", "pages", "year", "type"], axis=1)
            ),
            on=["isin"], how="outer", indicator="_mergeHeatmap"
        )
        .query("_mergeHeatmap != 'right_only'")
        .sort_values("publication date", ascending=True)
    )


def define_standard_info_mapper():
    return pd.DataFrame(
        {
            'standard': ['e1', 'e2', 'e3', 'e4', 'e5', 's1', 's2', 's3', 's4', 'g1'],
            'standard2': ['E1 Climate', 'E2 Pollution', 'E3 Water', 'E4 Biodiv', 'E5 Circular', 'S1 Workforce', 'S2 Value chain', 'S3 Communities', 'S4 Consumers', 'G1 Conduct'],
            'standardgroup': ['E', 'E', 'E', 'E', 'E', 'S', 'S', 'S', 'S', 'G'],
            'ig3_dp': [217, 72, 51, 125, 67, 198, 71, 69, 69, 55]
        }
    )

def plot_ui(which: str, df: pd.DataFrame) -> None:

    if which == "bubble-counter":
        return st.markdown(
            f"""
            <div style="
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
            ">
                <p style="
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    width: 100px;
                    height: 100px;
                    background-color: #4200ff;
                    color: white;
                    font-size: 36px;
                    font-weight: bold;
                    border-radius: 50%;
                    text-align: center;
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
                    margin: 0;
                ">
                    {len(df)}
                </p>
                <p style="margin-top: 10px;">CSRD reports so far</p>
                <a href="https://sustainabilityreportingnavigator.com/csrd-first100.pdf" target="_blank" style="color: #4200ff; font-size: 10pt">(Analysis of the first 100 reports)</a>
            </div>
            """,
            unsafe_allow_html=True
        )

    if which == "welcome-text":
        return st.markdown(f"""
                We are crowd-sourcing the collection of CSRD-compliant reports to support prepares and users of sustainability reporting.

                Below, you find a continuously updated list of CSRD-compliant reports for fiscal years starting on 01/01/2024.
                
                Want to make an addition? Feel free to do so [using this Google Sheet](https://docs.google.com/spreadsheets/d/1Nlyf8Yz_9Fst8rEmQc2IMc-DWLF1fpmBTB7n4FlZwxs/edit?gid=1695573594#gid=1695573594) and [follow us on LinkedIn](https://www.linkedin.com/company/sustainability-reporting-navigator/).
                """)

@st.cache_data
def download_pdf(url):
    """Fetch the PDF from a URL and return it as bytes."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load PDF: {e}")
        return None


def plot_heatmap(filtered_melted_df, split_view):

    filtered_melted_df["norm_hits"] = (
                    filtered_melted_df.groupby("company")["hits"]
                    .transform(lambda x: x / x.max() if x.max() != 0 else 0)
                )
    color_field = "norm_hits:Q"
    color_scale = alt.Scale(
        domain=[0, 0.5, 1],
        range=['#ffffff', '#a0a0ff', '#4200ff']
    )


    if split_view != "no split":

        heatmap_faceted = (
            alt.Chart(filtered_melted_df)
            .mark_rect(stroke="lightgray", filled=True)
            .encode(
                x=alt.X(
                    "standard",
                    title=None,
                    axis=alt.Axis(orient="top", labelAngle=0),
                    sort=[
                        'E1', 'E2', 'E3', 'E4', 'E5',
                        'S1', 'S2', 'S3', 'S4',
                        'G1'
                    ]
                ),
                y=alt.Y("company", title=None), 
                color=alt.condition(
                    alt.datum.norm_hits == 0,
                    alt.value('#ffffc5'),
                    alt.Color('norm_hits:Q', scale=color_scale, legend=None)
                    ),
                tooltip=[
                    alt.Tooltip("company", title="Company"),
                    alt.Tooltip("standard2", title="ESRS topic"),
                    alt.Tooltip("hits", title="Referenced", format="d")
                ]
            )
            .properties(width = 400)
            .facet(
                row=alt.Row(
                    "sector:N" if split_view == "by sector" else "country:N" if split_view == "by country" else "auditor:N", 
                    header=alt.Header(
                        orient='top',
                        labelAngle=0,
                        title=""
                    )
                )
            )
            .resolve_scale(
                x="independent",
                y='independent',
                color="shared"
            )
        )
        
        return st.altair_chart(heatmap_faceted)

    else:

        heatmap = (
            alt.Chart(filtered_melted_df)
            .mark_rect(stroke="lightgray", filled=True)
            .encode(
                x=alt.X(
                    "standard",
                    title=None,
                    axis=alt.Axis(orient="top", labelAngle=0),
                    sort=[
                        'E1', 'E2', 'E3', 'E4', 'E5',
                        'S1', 'S2', 'S3', 'S4',
                        'G1'
                    ]
                ),
                y=alt.Y("company", title=None), 
                color=alt.condition(
                    alt.datum.norm_hits == 0,
                    alt.value('#ffffc5'),
                    alt.Color('norm_hits:Q', scale=color_scale, legend=None)
                    ),
                tooltip=[
                    alt.Tooltip("company", title="Company"),
                    alt.Tooltip("standard2", title="ESRS topic"),
                    alt.Tooltip("hits", title="Referenced", format="d")
                ]
            )
            .properties(width = 400)
        )
        
        return st.altair_chart(heatmap)

@st.cache_data
def get_all_reports() -> pd.DataFrame:
    """ Get all available reports from the Sunhat API """
    all_reports = []
    currentPage = 1
    pageSize = 50

    print("Fetching reports from Sunhat API...")

    while True:
        response = requests.get(
            "https://sunhat-api.onrender.com/sustainability-reports/reports",
            headers={"Content-Type": "application/json"},
            params={"pageSize": pageSize, "page": currentPage},
        ).json()

        all_reports.extend(response.get("data"))
        
        pagination = response.get("pagination")
        if pagination.get("nextPage") is None:
            break

        currentPage += 1 

    return (
        pd.DataFrame(all_reports)
        .assign(
            companyName=lambda x: x["company"].apply(lambda y: y["name"]),
            isin=lambda x: x["company"].apply(lambda y: y["isin"])
            )
        .loc[:, ['id', 'companyName', 'isin', 'link']]
        )


def define_popover_title(query_companies_df) -> str:
    query_companies_names = query_companies_df['company'].values
    """ Define the title for the popover """
    if len(query_companies_names) == 0:
        return "Select companies from the table by selecting the box to the left of the name"
    elif len(query_companies_names) > 5:
        return f"You can only select a maximum of five companies ({len(query_companies_names)} selected)"
    elif len(query_companies_names) == 1:
        return f"Search in the report of {query_companies_names[0]}"
    elif len(query_companies_names) > 1:
        return f"Search in the reports of {', '.join(query_companies_names[:-1])}, and {query_companies_names[-1]}"


def query_single_report(reportId, prompt, numberOfReturnedChunks=5):
    """ Query a single report using the Sunhat API
    Args:
        reportId: str, the UUID report id
        prompt: str, the text query to be executed
        numberOfReturnedChunks: int, the number of chunks to be returned
            @ToDo: Implement pagination for returned chunks (but don't really need it)
    """
    return requests.post(
        "https://sunhat-api.onrender.com/sustainability-reports/query",
        headers={"Content-Type": "application/json"},
        json={ 
            "reportId": reportId,
            "query": prompt, 
            "pageSize": numberOfReturnedChunks, 
            }
    )


def summarize_text_bygpt(client, queryText, relevantChunkTexts):
    """ Summarize the text using GPT-4o-mini """    
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in gathering information from sustainability reports."},
            {"role": "user", "content": f"Answer diligently on this question {queryText} from the following texts of the report:"},
            {"role": "user", "content": relevantChunkTexts.replace("\n", " ").replace("\t", " ")},
            {"role": "user", "content": f"Be concise and provide the most relevant information from the texts only. Do not use the internet or general knowledge."},
        ],
        stream=True
        )


def display_annotated_pdf(query_report_link, similar_pages):
    return pdf_viewer(
        input=download_pdf(query_report_link),
        # annotations=query_results_annotations,
        height=800,
        pages_to_render=[
            int(p["page"]) 
            for p in sorted(similar_pages, key=lambda x: x["score"], reverse=True)
            ],
        )


def get_remote_ip() -> str:
    """Get remote ip."""

    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return None

        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return None
    except Exception as e:
        return None

    return session_info.request.remote_ip


def create_google_auth_credentials():
    with open("google-auth-credentials.json", "w") as f:
        json.dump(
            {
                "type": "service_account",
                "project_id": st.secrets["GOOGLE_project_id"],
                "private_key_id": st.secrets["GOOGLE_private_key_id"],
                "private_key": st.secrets["GOOGLE_private_key"],
                "client_email": st.secrets["GOOGLE_client_email"],
                "client_id": st.secrets["GOOGLE_client_id"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": st.secrets["GOOGLE_client_x509_cert_url"],
                "universe_domain": "googleapis.com"
            },
            f,
            indent=4
        )


def read_supabase_documents(supabase):
    return (
        pd.DataFrame(
            (
                supabase
                .from_("documents")
                .select("id, company_id, year, type, pages, companies(id, name, isin)")
                .execute()
            )
            .data
        )
        .assign(
            company = lambda x: x['companies'].apply(lambda y: y['name']),
            isin = lambda x: x['companies'].apply(lambda y: y['isin'])
        )
        .drop("companies", axis=1)
        .rename(columns={
            'id': 'document_id'
        })
    )


def get_most_similar_pages(prompt: str, pages: list, topk=5):
    """ Embed prompt with Mistral, compare with all supplied pages and return topk """
    client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
    embeddings_response = client.embeddings.create(
        model="mistral-embed",
        inputs=prompt
    )
    prompt_emb = embeddings_response.data[0].embedding

    for page in pages:
        if len(page["content"].strip()) < 500:
            page["score"] = 0
        else:
            distance = cosine_similarity([literal_eval(page["embedding"])], [prompt_emb])
            page["score"] = distance[0][0]


    pages = sorted(pages, key=lambda x: x["score"], reverse=True)
    pages = pages[:topk]
    
    return pages


def read_supabase_pages(supabase):
    return (
        pd.DataFrame(
            (
                supabase
                .from_("unique_pages")
                .select("document_id")
                .execute()
            )
            .data
        )
    )