import requests

BACKEND_BASE_ULR = "https://3eff41c5-b8bf-4056-94f9-e2712b151c0a.mock.pstmn.io/"
SOLVE_URI = BACKEND_BASE_ULR + "solve"


def solver_request(model):
    request = requests.post(SOLVE_URI, json=model)
    return request.json() if request.status_code == requests.codes.ok else request.status_code
