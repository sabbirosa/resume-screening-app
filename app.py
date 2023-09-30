from PyPDF2 import PdfReader

from model import clean_resume, clf, tfidf

reader = PdfReader('advocate_resume.pdf')

my_resume = ""

for i in range(len(reader.pages)):

    page = reader.pages[i]
    page_text = page.extract_text()
    my_resume += page_text

cleaned_resume = clean_resume(my_resume)
input_features = tfidf.transform([cleaned_resume])
prediction_id = clf.predict(input_features)[0]

category_mapping = {0: 'Advocate', 1: 'Arts', 2: 'Automation Testing', 3: 'Blockchain', 4: 'Business Analyst', 5: 'Civil Engineer', 6: 'Data Science', 7: 'Database', 8: 'DevOps Engineer', 9: 'DotNet Developer', 10: 'ETL Developer', 11: 'Electrical Engineering',
                    12: 'HR', 13: 'Hadoop', 14: 'Health and fitness', 15: 'Java Developer', 16: 'Mechanical Engineer', 17: 'Network Security Engineer', 18: 'Operations Manager', 19: 'PMO', 20: 'Python Developer', 21: 'SAP Developer', 22: 'Sales', 23: 'Testing', 24: 'Web Designing'}

category_name = category_mapping.get(prediction_id, "Unknown")

print("Predicted Category:", category_name)
