from flask import Flask , render_template, request, redirect
from db import DATABASE

from my_api import NLPAnalyzer




app = Flask(__name__)

dbo = DATABASE()
api = NLPAnalyzer()


@app.route("/") # Shows login page

def index(): 

    return render_template("login.html")

# shows register page

@app.route("/register")

def register():

    return render_template("register.html")



# making a file which accepts data from register page --- IMP:: When Submit button is pressed


@app.route("/perform_registeration", methods=["post"])

def perform_registeration():

    name_data = request.form.get("user_name")
    email_data = request.form.get("user_email")
    password_data = request.form.get("user_password")

    response = dbo.insert(name=name_data,email=email_data,password=password_data)

    if response==1:

        return render_template("login.html", message="Registeration Succesfull, Kindly login to proceed.")
    
    else:

        return render_template("register.html", message="Email already exists. You can login here.")



@app.route("/perform_login", methods=["post"])

def perform_login():

    email_data = request.form.get("user_email")
    password_data = request.form.get("user_password")

    response = dbo.search(email=email_data,password=password_data)

    if response==1:

        return redirect("/profile")
    
    else:

        return render_template("login.html", message="Incorrect email/password")
    

@app.route("/profile")

def profile():

    return render_template("profile.html")




@app.route("/abuse")

def abuse():

    return render_template("abuse.html")


@app.route("/perform_abuse_detection", methods=["post"])

def perform_abuse_detection():

    text_data = request.form.get("abuse_text")

    response = api.abuse_detection(text=text_data)

    return render_template("abuse_result.html", message=response)


@app.route("/sentiment_analysis")

def sentiment_analysis():

    return render_template("sentiment.html")


@app.route("/perform_sentiment_analysis", methods=["post"])

def perform_sentiment_analysis():

    text_data = request.form.get("sentiment_text")

    response = api.sentiment_analysis(text=text_data)

    return render_template("sentiment_result.html", message = response)


@app.route("/spam")

def spam():

    return render_template("spam.html")


@app.route("/perform_spam_detection", methods=["post"])

def perform_spam_detection():

    text_data = request.form.get("spam_text")

    response = api.spam_detection(text=text_data)

    response.pop("label")
    print(response)

    return render_template("spam_result.html", message=response)














    











app.run(debug=True)