import requests
import json

BACKEND_BASE_ULR = "https://3eff41c5-b8bf-4056-94f9-e2712b151c0a.mock.pstmn.io/"
SOLVE_URI = BACKEND_BASE_ULR + "solve"


def toJSON(el):
    return json.dumps(el, default=lambda o: o.__dict__,
                      sort_keys=True, indent=4)


def solver_request(model):
    request_body = model
    print(request_body)
    request = requests.post(SOLVE_URI, json=request_body)
    return request.json() if request.status_code == requests.codes.ok else request.status_code
