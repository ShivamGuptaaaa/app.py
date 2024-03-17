import streamlit as st
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords') 

#loading model 
import pickle

# Load 'clf.pkl'

with open('NLP/clf.pkl', 'rb') as f:
    clf = pickle.load(f, encoding='latin1')

with open('NLP/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f, encoding='latin1')


def CleanResume(txt):
        CleanTxt = re.sub('http\S+\s',' ',txt)
        CleanTxt = re.sub('RT|cc',' ',CleanTxt)
        CleanTxt = re.sub('#\S+\s',' ',CleanTxt)
        CleanTxt = re.sub('@\S+',' ',CleanTxt)
        CleanTxt = re.sub('[%s]' %re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',CleanTxt)
        CleanTxt = re.sub(r'[^\x00-\x7f]',' ',CleanTxt)
        CleanTxt = re.sub('\s+',' ',CleanTxt) 
        return CleanTxt
#web App
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])
   
    
    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')  

        
        Cleaned_resume = CleanResume(resume_text)
        input_feature = tfidf.transform([Cleaned_resume])
        prediction_id = clf.predict(input_feature)[0]
        st.write(prediction_id)
        
        #Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write("Predicted Category:", category_name)

        
#python main
if __name__ == "__main__":
    main()

