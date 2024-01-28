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
        h_state['correction_right_cylinder'] = '0'
    if h_state['correction_left_sphere'] != None and h_state['correction_left_cylinder'] == None:
        h_state['correction_left_cylinder'] = '0'
    return h_state

def train_regressor_for_native_vision(data, architecture=(5)):
    h_states = [clean_cylinder_nones(parse_h_state(d['h_state'])) for d in data]
    visions = [(d_state['right_native'], d_state['correction_right_sphere'], d_state['correction_right_cylinder'], d_state['corrected_right']) for d_state in h_states]
    visions += [(d_state['left_native'], d_state['correction_left_sphere'], d_state['correction_left_cylinder'], d_state['corrected_left']) for d_state in h_states]
    samples = [[float(value) if value != '1/~' else 0 for value in v] for v in visions if all(value != None for value in v)]
    X = [s[1:] for s in samples]
    y = [s[0] for s in samples]
    regressor = MLPRegressor(architecture, random_state=1, max_iter=500).fit(X, y)
    return lambda x: min(max(0, round(regressor.predict([x])[0], 3)), 1)

def preprocess(data):
    data = [add_age(d) for d in data]
    clean_diag = [d for d in data if not d['main_diag'].startswith('H40') and not d['main_diag'].startswith('H36')]
    for d in clean_diag:
        parsed_h_state = clean_cylinder_nones(parse_h_state(d['h_state']))
        d.update(parsed_h_state)
        del d['h_state']
        del d['anamnesis']
        del d['right_cylinder_degrees']
        del d['left_cylinder_degrees']
    return data