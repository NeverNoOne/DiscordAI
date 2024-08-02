

def generate_response(transscript:str):
    responses = {
        "hallo":"Moin",
        "Wie geht's":"Ich bin ein Bot mir geht es immer gut!",
        "Tschüss":"Rein gehauen!"
    }

    for key in responses:
        if key in transscript.lower():
            return responses[key]
    return "Was laberst du?"