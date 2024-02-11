import re
from dateutil.relativedelta import relativedelta
from sklearn.neural_network import MLPRegressor

def string_or_none(match_obj): return match_obj.group() if match_obj != None else None

def parse_h_state(h_state: str) -> dict[str, str | None]:
    start_id_of_left = h_state.find('ВОС')
    parsed = {
        'right_native': string_or_none(re.search('(?<=ВОД=)(\\d\\.\\d|1/~)', h_state)),
        'correction_right_sphere': string_or_none(re.search('(?<=с)([\\+\\-]\\d\\.\\d*)(?=дсф)', h_state[:start_id_of_left])),
        'correction_right_cylinder': string_or_none(re.search('(?<=с)([\\+\\-]\\d\\.\\d*)(?=дц)', h_state[:start_id_of_left])),
        'right_cylinder_degrees': string_or_none(re.search('(\\d\\d*)(?=гр)', h_state[:start_id_of_left])),
        'corrected_right': string_or_none(re.search('(?<=дсф=)(\\d\\.\\d*)|(?<=гр\\.=)(\\d\\.\\d*)', h_state[:start_id_of_left])),
        'left_native': string_or_none(re.search('(?<=ВОС=)(\\d\\.\\d|1/~)', h_state)),
        'correction_left_sphere': string_or_none(re.search('(?<=с)([\\+\\-]\\d\\.\\d*)(?=дсф)', h_state[start_id_of_left:])),
        'correction_left_cylinder': string_or_none(re.search('(?<=с)([\\+\\-]\\d\\.\\d*)(?=дц)', h_state[start_id_of_left:])),
        'left_cylinder_degrees': string_or_none(re.search('(\\d\\d*)(?=гр)', h_state[start_id_of_left:])),
        'corrected_left': string_or_none(re.search('(?<=дсф=)(\\d\\.\\d*)|(?<=гр\\.=)(\\d\\.\\d*)', h_state[start_id_of_left:])),
    }
    return parsed

def add_age(patient_data):
    patient_age = relativedelta(patient_data['date'], patient_data['birth'])
    patient_data['age'] = patient_age.years
    return patient_data

def clean_cylinder_nones(h_state):
    if h_state['correction_right_sphere'] != None and h_state['correction_right_cylinder'] == None:
        h_state['correction_right_cylinder'] = 0
    if h_state['correction_left_sphere'] != None and h_state['correction_left_cylinder'] == None:
        h_state['correction_left_cylinder'] = 0
    return h_state

def clean_sphere_nones(h_state):
    if h_state['correction_right_sphere'] == None and h_state['correction_right_cylinder'] != None:
        h_state['correction_right_sphere'] = 0
    if h_state['correction_left_sphere'] == None and h_state['correction_left_cylinder'] != None:
        h_state['correction_left_sphere'] = 0
    return h_state

def copy_native_to_corrected(h_state):
    if h_state['correction_right_sphere'] == None and h_state['correction_right_cylinder'] == None:
        h_state['corrected_right'] = h_state['right_native']
    if h_state['correction_left_sphere'] == None and h_state['correction_left_cylinder'] == None:
        h_state['corrected_left'] = h_state['left_native']
    return h_state

from sklearn.neighbors import KNeighborsRegressor

def train_regressor_for_native_vision(data):
    data = [clean_cylinder_nones(d) for d in data]
    visions = [
            (0., 0., 0., 0.),
            (1, 0., 0., 1.),
            (0.9, -0.5, 0., 1.),
            (0.5, -1, 0., 1.),
            (0.15, -1.5, 0., 1.),
            (0.1, -2, 0., 1.),
            (0.07, -3, 0., 1.),
            (0.06, -4, 0., 1.),
            (0.05, -5, 0., 1.),
            (0.04, -6, 0., 1.),
        ]
    visions += [(d['right_native'], d['correction_right_sphere'], d['correction_right_cylinder'], d['corrected_right']) for d in data]
    visions += [(d['left_native'], d['correction_left_sphere'], d['correction_left_cylinder'], d['corrected_left']) for d in data]
    samples = [v[:3] for v in visions if all(value != None for value in v) and v[3] == 1.0]
    X = [s[1:] for s in samples]
    y = [s[0] for s in samples]
    regressor = KNeighborsRegressor(n_neighbors=2).fit(X, y)
    return lambda x: min(max(0, round(regressor.predict([x])[0], 3)), 1)

def preprocess(data):
    data = [add_age(d) for d in data]
    # data = [d for d in data if not d['main_diag'].startswith('H40') and not d['main_diag'].startswith('H36')]
    for d in data:
        parsed_h_state = parse_h_state(d['h_state'])
        num_h_state = {k: None if v == None else float(v) if v != '1/~' else 0 for k, v in parsed_h_state.items()}
        d.update(num_h_state)
        del d['h_state']
        del d['anamnesis']
        del d['right_cylinder_degrees']
        del d['left_cylinder_degrees']
        del d['birth']
        del d['date']
        del d['insurance']
        del d['procedures']
        del d['visit_type']
        del d['paid']


    regressor = train_regressor_for_native_vision(data)
    data = [clean_cylinder_nones(d) for d in data]
    data = [clean_sphere_nones(d) for d in data]
    # add missing native vision
    for d in data:
        try:
            if d['right_native'] == None and d['corrected_right'] == 1.0:
                d['right_native'] = regressor((d['correction_right_sphere'], d['correction_right_cylinder']))
            if d['left_native'] == None and d['corrected_left'] == 1.0:
                d['left_native'] = regressor((d['correction_left_sphere'], d['correction_left_cylinder']))
        except:
            del d
    # if no correction copy native to corrected
    data = [copy_native_to_corrected(d) for d in data]
    # assert all have corrected and native
    data = [d for d in data if d['corrected_right'] != None and d['right_native'] != None and d['corrected_left'] != None and d['left_native'] != None]
    # fill the missing corrections with 0
    data = [{k:0 if v == None else v for k, v in d.items()} for d in data]

    return data