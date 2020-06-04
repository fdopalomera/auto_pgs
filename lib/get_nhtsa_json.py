import requests
import json
from time import time


def get_nhtsa_json(vin_list, i):

    start_time = time()
    url = 'https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/'

    vins_string = ';'.join(vin_list)
    post_fields = {'format': 'json', 'data': vins_string}
    r = requests.post(url=url, data=post_fields)
    # Filtar solo resultados
    car_list = json.loads(r.text)['Results']
    car_text = json.dumps(car_list)
    json_text = car_text[1:-1] + ', '

    # Calcular tiempo del request
    duration = round(time() - start_time, 1)
    print(f'{i}: {duration}s')

    return json_text


