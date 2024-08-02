import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
import re
from bs4 import BeautifulSoup
from transformers import pipeline
import warnings
import spacy
import requests
import google.generativeai as genai

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
genai.configure(api_key="AIzaSyAyRBWQ016M7-GJ65NQ9szFk-TkPHCje_U")


def Title(link):
    try:
        res = requests.get(link)
        html_content = res.text
        soup = BeautifulSoup(html_content, "html.parser")
        video_title = soup.find("meta", property="og:title")["content"]
        return video_title
    except Exception as e:
        return f"Error fetching title: {e}"


def PreProcess(text, link):
    nlp_spacy = spacy.load("en_core_web_lg")
    title = f"Title or main topic of the video is {Title(link)}"
    doc = nlp_spacy(text)
    tokens = [token.text for token in doc if not token.is_punct and not token.is_stop]
    tokens.append(title)
    return " ".join(tokens)


def QNA(question, context):
    model_name = "deepset/roberta-base-squad2"
    nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)
    QA_input = {"question": question, "context": context}
    try:
        res = nlp(QA_input)
        return str(res["answer"])
    except Exception as e:
        return f"Error in QNA: {e}"


def generate_extended_answer(prompt, max_tokens, temperature):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content([prompt])
        answer = response.text
        if max_tokens:
            answer = " ".join(answer.split()[:max_tokens])
        return answer
    except Exception as e:
        return f"Error generating answer: {e}"


with st.sidebar:
    st.subheader("Settings")
    max_tokens = st.number_input(
        "Maximum Tokens", min_value=1, max_value=5000, value=500
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)


def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry["text"] for entry in transcript])
        return transcript_text
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        return str(e)


def convert_to_markdown(answer):
    return f"# Extended Answer\n\n{answer}"


def markdown_to_text(markdown):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", markdown)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"#(.*?)\n", r"\1\n", text)
    text = re.sub(r"^\s*[*\-]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n+", "\n", text).strip()
    return text


def main():
    st.title("YouTube Transcript Q/A Chatbot")
    if "output_placeholder" not in st.session_state:
        st.session_state.output_placeholder = ""

    link = st.text_input("Enter YouTube Link")

    transcript = ""
    if link:
        try:
            video_id = re.findall(r"v=([A-Za-z0-9_-]+)", link)[0]
            transcript = fetch_transcript(video_id)
            if transcript:
                st.success("Transcript fetched successfully!")
            else:
                st.warning(
                    "No transcript found or transcript is disabled for this video."
                )
        except IndexError:
            st.error("Invalid YouTube link. Please enter a valid YouTube link.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    col1, col2 = st.columns(2)

    with col1:
        question = st.text_area("Question:", height=300)

    with col2:
        output = st.text_area(
            "Answer:",
            value=st.session_state.output_placeholder,
            height=300,
        )

    with col1:
        if st.button("Process"):
            if link and transcript:
                context = PreProcess(transcript, link)
                qa_answer = QNA(question, context)
                extended_prompt = f'''Give answers only based on the following question and context, extend the given answer: {qa_answer}\n\nQuestion: {question}\nContext: {context} and if the question is not realted to our Context then tell the me that, "please enter valid question related to your video"  '''
                extended_answer = generate_extended_answer(
                    extended_prompt, max_tokens, temperature
                )
                st.session_state.output_placeholder = markdown_to_text(extended_answer)
                st.experimental_rerun()
            else:
                st.warning("No link is entered")

    with col2:
        if (
            "output_placeholder" in st.session_state
            and st.session_state.output_placeholder
        ):
            # markdown_content = convert_to_markdown(st.session_state.output_placeholder)
            st.download_button(
                label="Export",
                data=markdown_to_text(st.session_state.output_placeholder),
                file_name=f"{question}_file.txt",
                mime="text",
            )


if __name__ == "__main__":
    main()