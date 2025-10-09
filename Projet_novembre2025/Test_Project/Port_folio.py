import streamlit as st


#Barre de Menu
with st.sidebar:
    st.title("Porte-Folio:")
    #st.image(caption="No image",width=150)
    st.title("Coordonnées:\n\n  📞: +33 07 44 51 67 92 \n\n  🏙️: Paris et Péripheries \n\n 📧: gerardtonfack0@gmail.com")
    st.write("Like and Share: 🌐")
    st.title("Sommaire:")
    st.page_link(page="Port_folio.py",label="Home",icon="🏚️")
    st.page_link(page="Pages/Salary_engineer.py",label="Salary Engineer",icon="🚕")
    st.page_link(page="Pages/Call_center.py",label="Call Center",icon="🚕")
    st.page_link(page="Pages/Student_Perf.py",label="Student Performance",icon="🚕")
    st.page_link(page="Pages/Games_review.py",label="Games Review",icon="🚕")
    
#A Propos de Moi
st.title("Dzeugueu Claude: Recherche  Stage Technique Data Analyste 📉💰")
st.title("About me 👨‍💻:")
st.write("Passionné par l'informatique, je souhaite mettre ::building à profit mes compétences dans le milieu de l'entreprise"
          "principalement dans le secteur de l'innovation toujours curieux des nouvelles avancées dans la data et l'IA;"
           " je voudrais mettre en avant mon profil de futur d'ingénieur en Business Analytics & Intelligence"
           "motivé par l'expérience de cette opportunité, je suis prêt à relever les responsabilités qui me seront confiées")
st.title("💻 Competences Techniques:")
st.write("* Langages: C, Java, Python, R, HTML/CSS, JavaScript\n"
         "* Frameworks: Bootstrap, Pandas, Streamlit, matplotlib, folium, Pandas\n"
         "* Logiciels: Docker, PowerBI, Github, JupiterNotebook, Excel"
        )
st.title("💡Softs Skills:")
st.write("* Travail d'equipe\n"
         "* Creativite\n"
         "* Curiosite intellectuelle\n"
         "* Sens de l'ethique\n"
         "* Adaptbilité\n"
        )
st.title("🏗️ Projets Academiques:")
st.write("* Dashboard dynamique sur le bilan d'Amazon France durant l'année 2021 \n"
         "* Site marchand dédié aux entreprises, proposant un catalogue de fournitures de bureau et une solution de facturation intégrée\n"
         "* Digitalisation de l'infrastructure réseau\n"
        )
st.title("💛 centres d'intérêt:")
st.write("* 🏯Anime / Manga \n"
         "* 🤖Technologie\n"
         "* 🏊Natation\n"
         "* 🧑‍🍳Cuisine\n"
        )

