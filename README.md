# ChatScholar: AI-Powered Document Analysis and Essay Grading

ChatScholar is a comprehensive web application designed to assist students, educators, and professionals by leveraging the power of Generative AI. It offers two core functionalities: an interactive Q&A system for PDF documents and an automated essay grading tool, both powered by Google's Gemini.

## 🚀 Overview

This project aims to streamline the process of information retrieval from dense documents and provide objective, criteria-based feedback on written work. Whether you're a student studying from a textbook, a researcher analyzing papers, or a writer looking for feedback, ChatScholar provides the tools you need to work smarter.

## 📸 Screenshots

Here is a visual walkthrough of the ChatScholar application.

**1. Landing Page**
*The main entry point where users can choose between "Q&A with PDF" and "Essay Grading".*
![Screenshot of the application's landing page showing two main options: Q&A with PDF and Essay Grading.](https://github.com/AnuragSinghDhami/ChatScholar/blob/main/static/phot1.png)

**2. Uploading a PDF for Q&A**
*The user interface for uploading a PDF document to the Q&A system.*
![Screenshot of the PDF upload interface. A user is selecting a PDF file from their local machine to start a Q&A session.](https://github.com/AnuragSinghDhami/ChatScholar/blob/main/static/phot2.png)

**3. Receiving Answers from the PDF**
*An interactive chat interface where the user asks a question and the AI provides an answer sourced from the document.*
![Screenshot of the Q&A chat window. The user has asked a question, and the AI has responded with a relevant answer extracted from the uploaded PDF.](https://github.com/AnuragSinghDhami/ChatScholar/blob/main/static/phot3.png)

**4. Defining the Essay Grading Rubric**
*The setup screen for the Essay Grading tool, where the user defines the criteria and their respective weights (e.g., Grammar: 30%, Vocabulary: 30%, Structure: 40%).*
![Screenshot showing the interface for setting up the essay grading rubric. There are sliders or input fields for Grammar, Vocabulary, and other criteria.](https://github.com/AnuragSinghDhami/ChatScholar/blob/main/static/phot4.png)

**5. Getting the Essay Graded**
*The final output screen. The user has submitted an essay, and Gemini has provided a score breakdown and detailed feedback based on the predefined rubric.*
![Screenshot displaying the results of an essay grade. It shows a final score, a breakdown by criteria, and specific feedback and suggestions for improvement from the Gemini model.](https://github.com/AnuragSinghDhami/ChatScholar/blob/main/static/phot5.png)


## ✨ Features

The application is divided into two main modules:

### 1. Q&A with PDF 📖

This feature allows users to have a conversation with their documents. Instead of manually searching through hundreds of pages, you can simply ask questions and get concise, relevant answers instantly.

-   **Upload & Process**: Securely upload any PDF file. The system processes the text and prepares it for querying.
-   **Natural Language Queries**: Ask questions in plain English, just like you would ask a person.
-   **Contextual Answers**: The AI provides answers based solely on the content of the uploaded document, ensuring accuracy and relevance.
-   **Source Citing**: (Optional Feature) The system can point to the page number or section of the PDF from where the information was retrieved.

### 2. Essay Grading ✍️

Receive instant, unbiased, and detailed feedback on your writing. This tool is perfect for self-assessment and improving writing skills.

-   **Customizable Rubric**: Define your own grading criteria. You can set the weight for different aspects such as:
    -   Grammar & Syntax
    -   Vocabulary & Diction
    -   Structure & Coherence
    -   Argument Strength
    -   Clarity
-   **Submit Your Essay**: Type directly into the editor or upload a text file.
-   **Instant AI Feedback**: Gemini analyzes the essay based on your custom rubric, providing a detailed score breakdown and constructive suggestions for improvement.

## 💻 Technology Stack

-   **Frontend**: HTML, Tailwind CSS
-   **Backend**: Python (Flask )
-   **AI Model**: Google Gemini 2.5 flash
-   **PDF Processing**: LangChain, PyPDF2
-   **Vector Database**:  FAISS

## 🛠️ How to Use

### For Q&A with PDF:
1.  Navigate to the **Q&A with PDF** section from the landing page.
2.  Click the 'Upload' button and select the PDF file you want to analyze.
3.  Once the file is processed, a chat box will appear.
4.  Type your question into the chat box and press Enter.
5.  Receive an AI-generated answer based on the document's content.

### For Essay Grading:
1.  Select **Essay Grading** from the landing page.
2.  Define your grading rubric. Adjust the sliders or input fields to assign weights to different criteria (e.g., Grammar, Vocabulary, etc.).
3.  Write your essay in the provided text editor or upload a `.txt` file.
4.  Click the 'Grade My Essay' button.
5.  View your detailed score and the constructive feedback provided by Gemini.
