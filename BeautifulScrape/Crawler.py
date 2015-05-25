from bs4 import BeautifulSoup
import re
import requests
import pandas as pd

def GenBs4(f):
    page  = requests.get(f).text
    return BeautifulSoup(page)     
    
def Scrape(rd):
    Temp_Names = []
    Temp_Title = []
    Temp_URL = []
    tag = rd.findAll('ul')
    p = str(tag[17:39]).split('<li>')
    for i in range(1,221):
        Temp_Names.append(str(p[i]).splitlines()[0][p[i].index(">")+1:-9])
        Temp_URL.append(str(p[i]).splitlines()[0][p[i].index('h')+6:p[i].index(">")-1])
        Temp_Title.append(str(p[i]).splitlines()[1][:-5])
    return Temp_Names, Temp_URL, Temp_Title    

def GenDept(x):
    first_link = x.find("h3",{'class':'faculty-departments'})
    New_String = re.sub("<.*?>", "", str(first_link))
    return New_String.strip()

def GenEdu(x):
    GSchool = []
    root = x.find(text="Education")
    if root == None:
        return "Education Not Found"
    else:
        for x in root.parent.next_siblings:
            GSchool.append(re.sub("<.*?>", "", str(x)).replace('\xc2\xa0', '').replace('\xe2\x80','').replace('\x99\xc3','').replace('\xb3\xbc','').strip().replace('\n',"|"))
    GSchool = [x for x in GSchool if x]
    Text = ','.join(GSchool)
    return Text[:256].strip()
   
def main():
    SCD = re.compile('Sc.D|ScD|S.D|D.Sc')
    PHD = re.compile('Ph.D.|PhD|Doctoral|doctoral')
    DRPH = re.compile('Dr.P.H|DrPH|DR.PH')
    MD = re.compile('MD|M.D|M.D.')    
    Temp = []
    MatrixDict = {}
    MatrixDict['firstname'] = []
    MatrixDict['lastname'] = []
    MatrixDict['department'] = [] 
    MatrixDict['grad_school'] = []
    MatrixDict['Grad_Yr'] = []
    MatrixDict['title'] = []
    MatrixDict['unversity'] = []
    MatrixDict['highest_degree'] = []  
    url = "http://www.hsph.harvard.edu/faculty/"
    soup_expatistan = GenBs4(url)
    Names,URL,Title = Scrape(soup_expatistan)
    for x in Names:
        MatrixDict['firstname'].append(x[0:x.index(' ')])
        MatrixDict['lastname'].append(x[x.index(' '):])
    for x in Title:
        MatrixDict['title'].append(x)
    for x in URL:
        Temp.append(GenEdu(GenBs4(x)))
        MatrixDict['department'].append(GenDept(GenBs4(x)))
    for x in Temp:
        if (any(char.isdigit() for char in x)) == True:
            MatrixDict['Grad_Yr'].append(sorted(re.findall('\d+',x), reverse=True)[0])
        else:
            MatrixDict['Grad_Yr'].append(0)       
        MatrixDict['grad_school'].append(re.sub("\d+,", "", str(x)).strip())
    for x in Temp:
        if SCD.match(x):
            MatrixDict['highest_degree'].append("Sc.D.")
        elif PHD.match(x):
            MatrixDict['highest_degree'].append("Ph.D.")
        elif DRPH.match(x):
            MatrixDict['highest_degree'].append("Dr.P.H")
        elif MD.match(x):
            MatrixDict['highest_degree'].append("M.D.")
        else:
            MatrixDict['highest_degree'].append("Degree Not Found")
    for x,i in enumerate(Temp):
        MatrixDict['unversity'].append("Havard University")
    csv = pd.DataFrame(MatrixDict, columns=['firstname','lastname','unversity','department','grad_school','title','highest_degree','Grad_Yr'])
    csv.to_csv("Elton_Assign1_infx575.csv")       
        
if __name__ == "__main__":
    main()
 
 
 