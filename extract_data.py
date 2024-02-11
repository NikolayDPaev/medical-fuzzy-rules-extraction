from bs4 import BeautifulSoup
import os
import datetime

def text(a, tag: str) -> str | None:
    t = a.find(tag)
    if t != None:
        return t.text
    return None

def date(date: str) -> tuple[int, int, int]:
    date_list = date.split('-')
    return int(date_list[0]), int(date_list[1]), int(date_list[2])

def birthdate_from_egn(egn: str):
    year= 1900 + int(egn[:2])
    month = int(egn[2:4])
    if month > 40:
        year += 100
        month -= 40
    day = int(egn[4:6])
    return year, month, day

def get_visit_type(visit_tag):
    if text(visit_tag, 'Consult') == '1':
        return 'consult'
    if text(visit_tag, 'disp') == '1':
        return 'disp'
    if text(visit_tag, 'rpHosp') == '1':
        return 'rpHosp'
    if text(visit_tag, 'LKK') == '1':
        return 'LKK'
    if text(visit_tag, 'Telk') == '1':
        return 'Telk'

def get_data_from_file(file: str):
    with open(file, 'r', encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "xml")
    for a in soup.findAll('AmbList'):
        try:
            data = {
                "date": datetime.datetime(*date(a.find('dataAl').text)),
                "paid": bool(int(a.find('Pay').text)),
                "sex": "m" if int(a.find('EGN').text[8])%2 == 0 else "f",
                "birth": datetime.datetime(*birthdate_from_egn(a.find('EGN').text)),
                "insurance": bool(text(a, 'IsHealthInsurance')),
                "main_diag": text(a.find('MainDiag'), 'MKB'),
                "diag": [text(p, 'MKB') for p in a.findAll('Diag')],
                "procedures": [text(p, 'kodP') for p in a.findAll('Procedure')],
                "anamnesis": text(a, 'Anamnesa'),
                "diabetes": True if 'диабет' in text(a, 'Anamnesa').lower() else False,
                "h_state": text(a, 'HState'),
                "visit_type": get_visit_type(a.find('VisitFor'))
            }
        except:
            continue
        yield data

def get_data_from_directory(dir: str):
    return (
        patient_data
            for root, _, files in os.walk(dir)
                for filename in files
                    if filename.endswith('.xml')
                        for patient_data in get_data_from_file(os.path.join(root, filename))
        )