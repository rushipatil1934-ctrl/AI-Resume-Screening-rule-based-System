"""
Sample data generator: creates realistic resumes + JDs for testing.
"""

SAMPLE_JD_ML_ENGINEER = """
Job Title: Senior Machine Learning Engineer

About the Role:
We are looking for a talented ML Engineer to join our AI team and help build
production-grade machine learning systems at scale.

Requirements (Must Have):
- 4+ years of experience in machine learning or data science
- Strong proficiency in Python, TensorFlow or PyTorch
- Experience with NLP, deep learning, and transformer models
- Expertise in scikit-learn, pandas, numpy
- Experience deploying models on AWS or GCP
- Proficiency with Docker, Kubernetes
- Strong understanding of SQL and NoSQL databases
- Experience with Git and CI/CD pipelines
- Excellent problem solving and analytical skills

Preferred Qualifications:
- Master's degree or PhD in Computer Science, AI, or related field
- Experience with LLMs, RAG pipelines, and generative AI
- Knowledge of Kafka, Spark, Airflow for data pipelines
- Experience with MLflow or similar experiment tracking
- Strong communication and leadership skills
- Agile/Scrum experience
"""

SAMPLE_RESUMES = [
    {
        "name": "Alice Chen",
        "text": """
Alice Chen
alice.chen@email.com | +1-555-0101 | linkedin.com/in/alicechen

SUMMARY
Senior ML Engineer with 6 years of experience building production machine learning
systems. Expert in Python, TensorFlow, PyTorch, and NLP. Passionate about LLMs
and generative AI.

EXPERIENCE
Senior Machine Learning Engineer | TechCorp AI | 2020 - Present
- Led development of NLP pipeline processing 10M documents/day using BERT and transformers
- Built RAG pipeline reducing hallucination by 45% using LangChain and GPT
- Deployed models on AWS SageMaker with Docker and Kubernetes orchestration
- Implemented CI/CD pipelines with GitHub Actions and Jenkins
- Mentored 4 junior engineers; led Agile sprint planning

ML Engineer | DataSolve Inc | 2018 - 2020
- Developed classification and regression models using scikit-learn, XGBoost
- Built real-time feature engineering pipelines with Kafka and Spark
- Managed PostgreSQL and MongoDB databases for ML feature store
- Used Airflow for orchestrating training pipelines

EDUCATION
Master of Science in Computer Science (AI) | Stanford University | 2018
Bachelor of Technology in Computer Engineering | IIT Delhi | 2016

SKILLS
Languages: Python, SQL, Bash, Scala
ML/AI: TensorFlow, PyTorch, scikit-learn, Keras, HuggingFace, LangChain, BERT, GPT, LLM
Cloud: AWS, GCP, Docker, Kubernetes, CI/CD
Data: pandas, numpy, Spark, Kafka, Airflow
Databases: PostgreSQL, MongoDB, Redis
Tools: Git, Jupyter, MLflow, Jira
Soft Skills: Leadership, Communication, Problem Solving, Agile, Scrum

PROJECTS
- Open-source NLP toolkit (2,000+ GitHub stars): transformer-based multilingual NER
- Kaggle Competition: Top 3% in NLP challenge (BERT fine-tuning)

CERTIFICATIONS
- AWS Certified Machine Learning Specialty
- Google Professional ML Engineer
"""
    },
    {
        "name": "Bob Martinez",
        "text": """
Bob Martinez
bob.martinez@gmail.com | 555-0202

OBJECTIVE
Data Scientist with 3 years experience seeking ML engineer role.

WORK HISTORY
Data Scientist | Analytics Co | 2021 - Present
- Built predictive models using scikit-learn and Python
- Analyzed datasets with pandas and numpy
- Created dashboards using Tableau and Power BI
- Some experience with random forest and gradient boosting models
- Used SQL for data extraction from MySQL and PostgreSQL

Junior Analyst | StartupXYZ | 2019 - 2021
- Excel reporting and basic Python scripting
- Statistical analysis

EDUCATION
Bachelor of Science in Statistics | State University | 2019

TECHNICAL SKILLS
Python, pandas, numpy, scikit-learn, SQL, MySQL, Tableau, Excel, Git, Jupyter

SOFT SKILLS
Teamwork, Communication, Analytical Thinking
"""
    },
    {
        "name": "Carol Singh",
        "text": """
Carol Singh
carol.singh@techmail.com

PROFILE
PhD researcher transitioning to industry ML engineering. 7 years research + 2 years industry.

EXPERIENCE
AI Research Scientist | ResearchLab | 2022 - Present
- Published 8 papers on transformer architectures and attention mechanisms
- Developed novel neural network architectures using PyTorch and TensorFlow
- Fine-tuned BERT, GPT models for domain-specific NLP tasks
- Implemented deep learning pipelines with distributed training on GCP
- Used Docker containers for reproducible research environments
- Collaborated with engineering teams on model deployment

PhD Research Fellow | MIT | 2018 - 2022
- Dissertation: "Efficient Transformer Models for Low-Resource NLP"
- Built custom LSTM, CNN, RNN architectures from scratch
- Extensive use of Python, NumPy, pandas, matplotlib for research

Software Engineer Intern | Google | 2017
- Worked on production ML systems using TensorFlow and Kubernetes
- Contributed to CI/CD pipeline improvements

EDUCATION
PhD Computer Science (NLP/ML) | MIT | 2022
Bachelor of Engineering, Computer Science | IIT Bombay | 2017

SKILLS
ML: PyTorch, TensorFlow, scikit-learn, HuggingFace, Keras, BERT, GPT, LLM, RAG, LangChain
Languages: Python, Java, C++, SQL, Bash
Cloud: GCP, AWS, Docker, Kubernetes
Data: pandas, numpy, Spark, PostgreSQL, MongoDB
Tools: Git, Jupyter, Airflow, MLflow, Jira
Soft: Research, Leadership, Communication, Problem Solving, Analytical

CERTIFICATIONS
- Google Professional Data Engineer
- Deep Learning Specialization (Coursera / Andrew Ng)
"""
    },
    {
        "name": "David Kim",
        "text": """
David Kim | david.kim@email.com

EXPERIENCE
Software Developer | WebDev Agency | 2020 - Present
- Built React and Node.js web applications
- REST API development with Django and Flask
- MySQL database management
- Some Python scripting and data analysis with pandas

Freelance Developer | 2018 - 2020
- WordPress websites and JavaScript apps
- HTML, CSS, Bootstrap

EDUCATION
Bachelor of Science, Information Technology | City College | 2018

SKILLS
JavaScript, React, Vue, Node.js, HTML, CSS, Python, Django, Flask,
MySQL, Git, Docker, AWS (basic)
"""
    },
   
    {
        "name": "Eva Patel",
        "text": """
Eva Patel
eva.patel@ml.com | LinkedIn: eva-patel-ml

SUMMARY
ML Engineer with 5 years experience specializing in NLP and production ML systems.
Strong background in Python, deep learning, and cloud deployment.

EXPERIENCE
Machine Learning Engineer | AIStartup | 2021 - Present
- Built NLP models (BERT, RoBERTa) for text classification and NER using HuggingFace
- Deployed models as microservices using FastAPI, Docker, and Kubernetes on AWS
- Implemented MLflow for experiment tracking and model registry
- Built data pipelines with Apache Airflow and Kafka
- Collaborated with product team using Agile/Scrum methodology
- Strong SQL and NoSQL (MongoDB, Redis) database skills

Data Scientist | ConsultingFirm | 2019 - 2021
- Developed classification and regression models with scikit-learn, XGBoost
- Feature engineering and EDA with pandas, numpy, matplotlib
- Deployed models on GCP using Cloud Run and BigQuery

EDUCATION
Master of Science, Data Science | Carnegie Mellon University | 2019
Bachelor of Technology, CS | BITS Pilani | 2017

SKILLS
ML/AI: scikit-learn, TensorFlow, PyTorch, HuggingFace, LangChain, LLM, NLP
Languages: Python, SQL, Bash
Cloud: AWS, GCP, Docker, Kubernetes, CI/CD
Data Engineering: Kafka, Airflow, Spark, pandas, numpy
Databases: PostgreSQL, MongoDB, Redis, BigQuery
Tools: MLflow, Git, Jupyter, Jira, Confluence
Soft Skills: Communication, Leadership, Problem Solving, Agile, Scrum

PROJECTS
- Built open-source RAG chatbot with LangChain + GPT-4
- Kaggle: Top 5% NLP competition

CERTIFICATIONS
- AWS Certified ML Specialty
- TensorFlow Developer Certificate
"""
    },

    {
       """ 
Rushikesh Patil
leetcode.com/Rushikeshpatil 7 | 
rushi.patil1934@gmail.com | 
Summary
github.com/rushipatil1934-ctrl | 
Bengaluru, India
Computer Science Engineering student (Class of 2027) specializing in Machine Learning, Deep Learning, and Software
Engineering. Experienced in building end-to-end ML pipelines using Python, Scikit-learn, and XGBoost across healthcare, finance,
and HR domains. Familiar with LLMs and RAG architectures. Proficient in DSA and OOP with 120+ LeetCode problems
solved.
Education
Garden City University
B.Tech in Computer Science Engineering | CGPA: 7.8/10
Bengaluru, Karnataka
Aug 2023– Jun 2027
• Coursework: Data Structures & Algorithms, Machine Learning, Deep Learning, DBMS, Operating Systems, OOP, Computer
Networks
Technical Skills
Languages: Python, C++, JavaScript, SQL
ML/AI: Scikit-learn, XGBoost,svm, TensorFlow, Keras, NLTK, spaCy , NLP
GenAI: Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), Deep Learning, Neural Networks
Data & Tools: Pandas, NumPy, Matplotlib, Seaborn, Feature Engineering, Git, GitHub, AWS (basics), MySQL, HTML, CSS
Projects
IntelliHire– AI Resume Screening System | Python, NLP, Scikit-learn, TF-IDF
Jan 2026– june 2026
• Processed 10,000+ resumes through an NLP pipeline automating candidate evaluation and reducing screening time by 50%.
• Achieved 87% accuracy across 6 job roles using an ensemble of Logistic Regression, SVM, and Gradient Boosting with
cross-validation.
• Engineered TF-IDF features with keyword extraction and skill-match scoring to rank candidates against job descriptions.
ReadmitIQ– Diabetic Patient Readmission Predictor | Python, XGBoost, Scikit-learn, Matplotlib
Jun 2026
• Built a clinical ML pipeline on a 100,000-record dataset predicting 30-day readmission risk with 80%+ accuracy using
XGBoost.
• Performed EDA, feature engineering, and SMOTE oversampling; compared XGBoost, Random Forest, and Logistic Regression
via ROC-AUC scoring.
• Visualized top clinical risk factors (HbA1c, prior admissions, medication count) using Matplotlib to support data-driven
insights.
CreditGate– Loan Approval Prediction System | Python, Scikit-learn, Pandas, Random Forest
Jun 2026
• Processed 5,000+ loan records with data cleaning and feature engineering; achieved 85%+ accuracy via GridSearchCV
tuning.
• Benchmarked Logistic Regression, Decision Tree, and Random Forest; evaluated using Accuracy, Precision, Recall, and
F1-Score.
• Identified income, credit history, and loan amount as key approval drivers through exploratory data analysis.
Certifications
Introduction to Data Science | Cisco Networking Academy | Verify
Deep Learning & Neural Networks with Keras | IBM / Coursera | Verify
Machine Learning Specialization | DeepLearning.AI / Coursera (Andrew Ng)
Achievements & Activities
Jan 2025
Competitive Programming
• Solved 120+ DSA problems on LeetCode (Arrays, DP, Trees, Graphs); peak contest rating 1450+.
• Participates regularly in LeetCode Weekly Contests to sharpen problem-solving under time constraints.
Open Source & Self-Learning
• Published 3 end-to-end ML projects on GitHub with clean documentation and reproducible notebooks.
2025– Present
2026– Present
• Actively upskilling in Generative AI, LLMs, and RAG through hands-on projects and online courses."""
    },
]
