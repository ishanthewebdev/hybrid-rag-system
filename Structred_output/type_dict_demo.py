from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int

new_person:Person={'name':'ishan','age':25}
print(new_person)    