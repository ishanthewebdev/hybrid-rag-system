from pydantic import BaseModel,EmailStr
from typing import Optional #jese typed dict me annoted hota tha isme optional hai
#as typeddic lacks validation , pydantic provides validation and we cannot pass invalied type , if we have defined a str , we should passa str only

class Student(BaseModel):
    name:str
   # name:str ='nitish' here we are setting a default value so agar kuch pass nhi karnege to by default ye print ho jayega
    age:Optional[int]=None #kuch na kuch default value dena hi padhta hai if we are setting things as optional
    email: EmailStr #for email validation


new_student= {'name': 'ishan','age':24}  #we cant pass int here

student=Student(**new_student)
# same we can do with pydantic
#we have to write a review class here 
# structred_model= model.with_structured_output(Review)

# result = structred_model.invoke("""  the hardware is great, but the software feels quite a bit laggy, it has a great UI but the slow user expirence. Hoping a software update can fix it , it has too many pre- installed apps that i cant remove
# """)

# print(result)